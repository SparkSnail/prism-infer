import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.prefix_cache import PrefixCacheService
from prism_infer.engine.sequence import Sequence
from prism_infer.sampling_params import SamplingParams
from prism_serve.router.fingerprint import PromptFingerprint
from prism_serve.router.loader import load_cached_prefix
from prism_serve.router.protocol import CachedPrefixDecision
from prism_serve.router.rpc import InProcessPrefixCacheRPC


@pytest.fixture
def small_block():
    previous = Sequence.block_size
    Sequence.block_size = 4
    try:
        yield
    finally:
        Sequence.block_size = previous


@pytest.mark.asyncio
async def test_real_services_execute_cross_layer_mapped_load(small_block):
    source_manager = BlockManager(8, 4, instance_id="src", instance_epoch="se")
    source_seq = Sequence(list(range(10)), SamplingParams())
    source_manager.allocate(source_seq, 0)
    source_seq.num_scheduled_tokens = source_seq.num_tokens
    source_manager.hash_blocks(
        source_seq, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    source_cache = torch.arange(64, dtype=torch.float32).reshape(2, 1, 8, 4, 1, 1)
    target_cache = torch.zeros_like(source_cache)
    source = PrefixCacheService(source_manager, source_cache)
    target = PrefixCacheService(
        BlockManager(8, 4, instance_id="dst", instance_epoch="de"), target_cache
    )
    rpc = InProcessPrefixCacheRPC({"src": source, "dst": target})
    fingerprint = PromptFingerprint.create(
        namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text", token_ids=list(range(10)), block_size=4,
    )
    plan = await load_cached_prefix(
        rpc, req_id="r", operation_id="op", fingerprint=fingerprint,
        sampling_params={},
        decision=CachedPrefixDecision("src", "se", "dst", 2, 8, 64, 1.0),
        target_epoch="de",
    )
    committed = target._operations["op"].sequence
    assert plan.cached_prefix_tokens == committed.num_cached_tokens == 8
    assert committed.token_ids[8:] == [8, 9]
    assert source_manager._transfer_pins == {}

