import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.kv_transfer import MappedPrefixTransferReq, MappedTransferStatus
from prism_infer.engine.prefix_cache import PrefixCacheService, PrefixOperationStatus
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams


@pytest.fixture
def small_block():
    previous = Sequence.block_size
    Sequence.block_size = 4
    try:
        yield
    finally:
        Sequence.block_size = previous


def _source(small_block):
    manager = BlockManager(8, 4, instance_id="src", instance_epoch="se")
    sequence = Sequence(list(range(9)), SamplingParams())
    manager.allocate(sequence, 0)
    sequence.num_scheduled_tokens = sequence.num_tokens
    manager.hash_blocks(
        sequence, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    cache = torch.arange(2 * 1 * 8 * 4 * 1 * 1, dtype=torch.float32).reshape(2, 1, 8, 4, 1, 1)
    return PrefixCacheService(manager, cache), sequence


def test_remote_prepare_copy_commit_and_suffix_boundary(small_block):
    source, sequence = _source(small_block)
    target_manager = BlockManager(8, 4, instance_id="dst", instance_epoch="de")
    target_cache = torch.zeros_like(source.kv_cache)
    target = PrefixCacheService(target_manager, target_cache)
    expected = [
        (source.block_manager.blocks[bid].hash, source.block_manager.blocks[bid].token_ids)
        for bid in sequence.block_table[:2]
    ]
    src_blocks = source.resolve_prefix(
        "op", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    prepared = target.prepare(
        "op", "r", mode="remote_transfer", block_count=2,
        token_ids=list(range(9)), sampling_params=SamplingParams(),
    )
    request = MappedPrefixTransferReq(
        "op", "r", "src", "se", "dst", "de", src_blocks,
        prepared.dst_block_ids, "ns", "compat", "text",
    )
    assert target.transfer_from(source, request) == MappedTransferStatus.COMPLETED
    committed = target.commit(
        "op", namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text", cached_prefix_tokens=8,
    )
    assert committed.num_cached_tokens == 8
    assert committed.token_ids[8:] == [8]
    for src, dst in zip(src_blocks, prepared.dst_block_ids):
        assert torch.equal(source.kv_cache[:, :, src], target.kv_cache[:, :, dst])
    assert target.status("op") == PrefixOperationStatus.COMMITTED
    assert source.unpin("op") is True


def test_remote_prepare_oom_rolls_back(small_block):
    target = PrefixCacheService(BlockManager(1, 4, instance_id="dst"))
    assert target.prepare(
        "op", "r", mode="remote_transfer", block_count=2,
        token_ids=list(range(9)), sampling_params=SamplingParams(),
    ) is None
    assert len(target.block_manager.free_block_ids) == 1


def test_resource_counts_are_read_from_infer_registries(small_block):
    source, sequence = _source(small_block)
    expected = [(source.block_manager.blocks[sequence.block_table[0]].hash, list(range(4)))]
    source.resolve_prefix(
        "pin", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    target = PrefixCacheService(BlockManager(4, 4, instance_id="dst"))
    target.prepare(
        "pending", "r", mode="remote_transfer", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    assert source.resource_counts()["transfer_pins"] == 1
    assert target.resource_counts()["pending_allocations"] == 1

    source.prepare(
        "pin", "local", mode="local_reuse", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    assert source.resource_counts()["pending_allocations"] == 0


def test_abort_pending_is_idempotent_and_commit_is_irreversible(small_block):
    target = PrefixCacheService(BlockManager(4, 4, instance_id="dst"))
    target.prepare(
        "op", "r", mode="remote_transfer", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    assert target.abort("op") == PrefixOperationStatus.ABORTED
    assert target.abort("op") == PrefixOperationStatus.ABORTED
    assert len(target.block_manager.free_block_ids) == 4


def test_missing_tensor_backend_fails_unknown_without_commit(small_block):
    source, sequence = _source(small_block)
    source.kv_cache = None
    target = PrefixCacheService(BlockManager(4, 4, instance_id="dst", instance_epoch="de"))
    expected = [(source.block_manager.blocks[sequence.block_table[0]].hash, list(range(4)))]
    src = source.resolve_prefix(
        "op", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    prepared = target.prepare(
        "op", "r", mode="remote_transfer", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    request = MappedPrefixTransferReq(
        "op", "r", "src", "se", "dst", "de", src,
        prepared.dst_block_ids, "ns", "compat", "text",
    )
    assert target.transfer_from(source, request) == MappedTransferStatus.UNKNOWN
    with pytest.raises(ValueError, match="not complete"):
        target.commit(
            "op", namespace="ns", kv_compatibility_id="compat",
            request_context_digest="text", cached_prefix_tokens=4,
        )


def test_local_reuse_commit_converts_pin_to_sequence_ref(small_block):
    service, sequence = _source(small_block)
    expected = [
        (service.block_manager.blocks[bid].hash, service.block_manager.blocks[bid].token_ids)
        for bid in sequence.block_table[:2]
    ]
    blocks = service.resolve_prefix(
        "local", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    operation = service.prepare(
        "local", "r", mode="local_reuse", block_count=2,
        token_ids=list(range(9)), sampling_params=SamplingParams(),
    )
    assert operation.dst_block_ids == ()
    committed = service.commit(
        "local", namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text", cached_prefix_tokens=8,
    )
    assert tuple(committed.block_table) == blocks
    assert "local" not in service.block_manager._transfer_pins
    for block_id in blocks:
        assert service.block_manager.blocks[block_id].ref_count == 2


def test_watchdog_aborts_only_unstarted_prepared_operation(small_block):
    target = PrefixCacheService(BlockManager(4, 4, instance_id="dst"))
    operation = target.prepare(
        "op", "r", mode="remote_transfer", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    expired = target.expire_unstarted(1.0, now=operation.created_at + 2.0)
    assert expired == ["op"]
    assert target.status("op") == PrefixOperationStatus.ABORTED
    assert len(target.block_manager.free_block_ids) == 4


def test_watchdog_does_not_free_unknown_handed_off_transfer(small_block):
    source, sequence = _source(small_block)
    source.kv_cache = None
    target = PrefixCacheService(BlockManager(
        4, 4, instance_id="dst", instance_epoch="de"
    ))
    expected = [(source.block_manager.blocks[sequence.block_table[0]].hash, list(range(4)))]
    src = source.resolve_prefix(
        "op", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    operation = target.prepare(
        "op", "r", mode="remote_transfer", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    request = MappedPrefixTransferReq(
        "op", "r", "src", "se", "dst", "de", src,
        operation.dst_block_ids, "ns", "compat", "text",
    )
    assert target.transfer_from(source, request) == MappedTransferStatus.UNKNOWN
    assert target.expire_unstarted(1.0, now=operation.created_at + 2.0) == []
    assert len(target.block_manager.used_block_ids) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_mapped_block_copy_parity(small_block):
    source, sequence = _source(small_block)
    source.kv_cache = source.kv_cache.cuda()
    target = PrefixCacheService(
        BlockManager(4, 4, instance_id="dst", instance_epoch="de"),
        torch.zeros_like(source.kv_cache),
    )
    expected = [(source.block_manager.blocks[sequence.block_table[0]].hash, list(range(4)))]
    src = source.resolve_prefix(
        "gpu", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    prepared = target.prepare(
        "gpu", "r", mode="remote_transfer", block_count=1,
        token_ids=list(range(5)), sampling_params=SamplingParams(),
    )
    request = MappedPrefixTransferReq(
        "gpu", "r", "src", "se", "dst", "de", src,
        prepared.dst_block_ids, "ns", "compat", "text",
    )
    assert target.transfer_from(source, request) == MappedTransferStatus.COMPLETED
    torch.cuda.synchronize()
    assert torch.equal(
        source.kv_cache[:, :, src[0]],
        target.kv_cache[:, :, prepared.dst_block_ids[0]],
    )
