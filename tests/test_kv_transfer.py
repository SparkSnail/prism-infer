import pytest
import torch
from collections import defaultdict

from prism_infer.engine.kv_transfer import (
    TransferReq, ChunkedBlock, KVBlockPusher, KVReceiver, NCCLTransport,
    _calc_block_bytes,
)

# ── helpers ────────────────────────────────────────────────────────────────────

NUM_BLOCKS, NUM_LAYERS, KV_HEADS, HEAD_DIM, BLOCK_SIZE = 10, 2, 2, 4, 4


def _fake_kv(num_blocks=NUM_BLOCKS):
    return torch.zeros(2, NUM_LAYERS, num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM)


def _make_pusher(transport=None, max_bytes_inflight=None,
                 max_blocks_per_peer=64, num_blocks=NUM_BLOCKS):
    kv = _fake_kv(num_blocks)
    bb = _calc_block_bytes(kv, BLOCK_SIZE)
    if max_bytes_inflight is None:
        max_bytes_inflight = bb * num_blocks * 2  # large enough to not trigger by default

    if transport is None:
        class Immediate:
            def send_async(self, dst, chunk, on_complete): on_complete()
            def send_batch_async(self, dst, chunks, on_complete): on_complete()
            def ack(self, *args, **kwargs): pass
        transport = Immediate()

    return KVBlockPusher(transport, kv, BLOCK_SIZE,
                         max_bytes_inflight, max_blocks_per_peer)


def _make_req(block_table, block_hint=None, on_fail="recompute"):
    return TransferReq(
        op_id="op-1", seq_id="s-1",
        src_instance="p-0", dst_instance="dc-1",
        block_table=block_table, block_hint=block_hint or [],
        on_fail=on_fail,
    )


def test_delta_filters_block_hint():
    hint = {0, 1}
    delta = [b for b in [0, 1, 2, 5, 6] if b not in hint]
    assert delta == [2, 5, 6]


def test_delta_empty_all_cached():
    sent = []
    class R:
        def send_async(self, dst, chunk, on_complete): sent.append(chunk)
        def send_batch_async(self, dst, chunks, on_complete): sent.extend(chunks)
        def ack(self, op_id, dst, bytes_sent=0): pass
    pusher = _make_pusher(transport=R())
    pusher.transfer(_make_req([0, 1, 2], block_hint=[0, 1, 2]))
    assert sent == []


def test_coalesce_merges_adjacent():
    p = _make_pusher()
    chunks = p._coalesce([2, 3, 4, 7, 8], "s", "o")
    assert len(chunks) == 2
    assert chunks[0].block_ids == [2, 3, 4]
    assert chunks[1].block_ids == [7, 8]


def test_coalesce_single_block():
    p = _make_pusher()
    chunks = p._coalesce([5], "s", "o")
    assert len(chunks) == 1 and chunks[0].block_ids == [5]


def test_coalesce_all_non_adjacent():
    p = _make_pusher()
    chunks = p._coalesce([0, 2, 4, 6], "s", "o")
    assert len(chunks) == 4


def test_coalesce_respects_max_bytes_inflight():
    """Deadlock guard: merged chunk must not exceed max_bytes_inflight."""
    kv = _fake_kv()
    bb = _calc_block_bytes(kv, BLOCK_SIZE)
    max_inf = int(bb * 1.5)  # allows at most 1 block per chunk

    class D:
        def send_async(self, dst, c, cb): cb()
        def send_batch_async(self, dst, cs, cb): cb()
        def ack(self, *a): pass

    p = KVBlockPusher(D(), kv, BLOCK_SIZE, max_inf, 64)
    chunks = p._coalesce([5, 6, 7], "s", "o")
    for c in chunks:
        assert c.size_bytes <= max_inf, f"chunk {c.block_ids} exceeds limit"
    assert len(chunks) > 1


def test_nccl_block_slices_shape_and_count():
    """_block_slices must produce one contiguous tensor per block,
    matching the per-block irecv ops issued by recv_kv on the destination."""
    kv = _fake_kv()
    transport = NCCLTransport(pd_group=None, decode_rank=1, kv_cache=kv)

    slices = transport._block_slices([2, 3, 4])

    assert len(slices) == 3
    expected_shape = kv[:, :, 0, :, :, :].shape
    for s in slices:
        assert s.shape == expected_shape
        assert s.is_contiguous()


def test_flow_control_defers_overflow():
    """When bytes_inflight would exceed the cap, chunks enter the deferred queue."""
    sent = []; completions = []

    class R:
        def send_async(self, dst, c, on_complete): sent.append(c); completions.append(on_complete)
        def send_batch_async(self, dst, cs, on_complete): sent.extend(cs); completions.append(on_complete)
        def ack(self, op_id, dst, bytes_sent=0): pass

    kv = _fake_kv()
    bb = _calc_block_bytes(kv, BLOCK_SIZE)
    p = KVBlockPusher(R(), kv, BLOCK_SIZE, int(bb * 1.5), 64)
    p.transfer(_make_req([0, 1, 2]))
    assert len(sent) == 1
    assert len(p.deferred["dc-1"]) > 0


def test_deferred_flushed_after_completion():
    sent = []; completions = []

    class R:
        def send_async(self, dst, c, on_complete): sent.append(c); completions.append(on_complete)
        def send_batch_async(self, dst, cs, on_complete): sent.extend(cs); completions.append(on_complete)
        def ack(self, op_id, dst, bytes_sent=0): pass

    kv = _fake_kv()
    bb = _calc_block_bytes(kv, BLOCK_SIZE)
    p = KVBlockPusher(R(), kv, BLOCK_SIZE, int(bb * 1.5), 64)
    p.transfer(_make_req([0, 1]))
    assert len(sent) == 1
    completions[0]()
    assert len(sent) == 2


def test_receiver_ready_after_mark():
    r = KVReceiver()
    r.expect("s1", [0, 1])
    assert not r.is_ready("s1")
    r.mark_received("s1")
    assert r.is_ready("s1")


def test_receiver_consume_clears():
    r = KVReceiver()
    r.mark_received("s1")
    r.consume_ready("s1")
    assert not r.is_ready("s1")


def test_receiver_unknown_not_ready():
    assert not KVReceiver().is_ready("x")


def test_on_fail_default_is_recompute():
    req = TransferReq(op_id="x", seq_id="s", src_instance="p",
                      dst_instance="d", block_table=[0], block_hint=[])
    assert req.on_fail == "recompute"


def test_on_fail_explicit_fail():
    req = TransferReq(op_id="x", seq_id="s", src_instance="p",
                      dst_instance="d", block_table=[0], block_hint=[],
                      on_fail="fail")
    assert req.on_fail == "fail"


def test_calc_block_bytes():
    kv = torch.zeros(2, 28, 10, 256, 8, 128)  # Qwen3-7B params (float32)
    assert _calc_block_bytes(kv, 256) == 2 * 28 * 256 * 8 * 128 * 4


class _DelayedSendTransport:
    def __init__(self):
        self.completions = []
        self.acks = []

    def send_batch_async(self, dst, chunks, on_complete):
        self.completions.append(on_complete)

    def ack(self, op_id, dst, bytes_sent):
        self.acks.append((op_id, dst, bytes_sent))

    def has_pending(self):
        return bool(self.completions)

    def complete_one(self):
        self.completions.pop(0)()


def _make_prefill_connector(transport):
    from types import SimpleNamespace

    from prism_infer.engine.block_manager import BlockManager
    from prism_infer.engine.kv_connector import PrefillConnector

    block_manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    pusher = KVBlockPusher(
        transport=transport,
        kv_cache=_fake_kv(num_blocks=4),
        block_size=BLOCK_SIZE,
    )
    connector = PrefillConnector(
        pusher=pusher,
        config=SimpleNamespace(
            instance_id="prefill-0",
            pd_decode_addr="decode-0",
            kv_transfer_on_fail="recompute",
        ),
        block_manager=block_manager,
    )
    return connector, pusher, block_manager


def test_prefill_transfer_retain_survives_sequence_cleanup():
    from prism_infer.engine.sequence import Sequence, SequenceStatus
    from prism_infer.sampling_params import SamplingParams

    transport = _DelayedSendTransport()
    connector, _, block_manager = _make_prefill_connector(transport)
    seq = Sequence([1, 2, 3, 4], SamplingParams())
    block_manager.allocate(seq, 0)
    block_id = seq.block_table[0]
    seq.num_cached_tokens = seq.num_prompt_tokens

    connector.on_prefill_done(seq)
    assert seq.status == SequenceStatus.FINISHED
    block_manager.deallocate(seq)

    assert block_id in block_manager.used_block_ids
    assert block_manager.blocks[block_id].ref_count == 1
    assert connector.has_pending()

    transport.complete_one()

    assert block_id not in block_manager.used_block_ids
    assert block_manager.blocks[block_id].ref_count == 0
    assert not connector.has_pending()


def test_prefill_transfer_submit_failure_rolls_back_retain_and_pending():
    from prism_infer.engine.sequence import Sequence, SequenceStatus
    from prism_infer.sampling_params import SamplingParams

    class FailingTransport:
        def send_batch_async(self, dst, chunks, on_complete):
            raise RuntimeError("submit failed")

        def ack(self, op_id, dst, bytes_sent):
            raise AssertionError("failed submit must not ack")

        def has_pending(self):
            return False

    connector, pusher, block_manager = _make_prefill_connector(FailingTransport())
    seq = Sequence([1, 2, 3, 4], SamplingParams())
    block_manager.allocate(seq, 0)
    block_id = seq.block_table[0]
    seq.num_cached_tokens = seq.num_prompt_tokens

    with pytest.raises(RuntimeError, match="submit failed"):
        connector.on_prefill_done(seq)

    assert seq.status == SequenceStatus.WAITING
    assert block_manager.blocks[block_id].ref_count == 1
    assert not block_manager._pd_transfer_retains
    assert not pusher.has_pending()


def test_zero_delta_completion_leaves_no_pending_state():
    completed = []
    pusher = _make_pusher()
    req = _make_req(block_table=[0, 1], block_hint=[0, 1])

    pusher.transfer(req, on_complete=lambda: completed.append(req.op_id))

    assert completed == [req.op_id]
    assert not pusher.has_pending()


def test_engine_empty_step_polls_pending_connector_before_finishing():
    from prism_infer.engine.llm_engine import LLMEngine

    class EmptyScheduler:
        def schedule(self):
            return [], False

        def is_finished(self):
            return True

    class PendingConnector:
        def __init__(self):
            self.pending = True
            self.poll_calls = 0

        def poll(self):
            self.poll_calls += 1
            self.pending = False

        def has_pending(self):
            return self.pending

    engine = object.__new__(LLMEngine)
    engine.scheduler = EmptyScheduler()
    engine.kv_connector = PendingConnector()

    assert not engine.is_finished()
    assert engine.step() == ([], 0)
    assert engine.kv_connector.poll_calls == 1
    assert engine.is_finished()


def test_nccl_poll_preserves_work_enqueued_by_completion_callback():
    class Work:
        def __init__(self, completed):
            self.completed = completed

        def is_completed(self):
            return self.completed

    transport = object.__new__(NCCLTransport)
    next_entry = ([Work(False)], [], [], lambda: None)
    transport._pending = [
        ([Work(True)], [], [], lambda: transport._pending.append(next_entry))
    ]

    transport.poll_completions()

    assert transport._pending == [next_entry]
