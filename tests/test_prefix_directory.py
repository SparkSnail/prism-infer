import time
import threading

import pytest

from prism_infer.engine.block_manager import BlockManager, FullReportRequired
from prism_infer.engine.sequence import Sequence


@pytest.fixture
def small_block():
    previous = Sequence.block_size
    Sequence.block_size = 4
    try:
        yield
    finally:
        Sequence.block_size = previous


def _cached_manager(capacity=16, lease=30.0):
    manager = BlockManager(
        8, 4, instance_id="d0", instance_epoch="e1",
        prefix_event_log_capacity=capacity, prefix_consumer_lease_s=lease,
    )
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = seq.num_tokens
    manager.hash_blocks(
        seq, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    return manager, seq


def test_full_report_peek_ack_is_replayable_per_consumer(small_block):
    manager, seq = _cached_manager()
    report_a = manager.full_report_and_register("a", "ga")
    report_b = manager.full_report_and_register("b", "gb")
    assert report_a.snapshot_seq_no == report_b.snapshot_seq_no == 2
    manager.deallocate(seq)
    manager._evict_one()
    events_a = manager.peek_events("a", "ga", 2)
    assert [event.kind for event in events_a] == ["evicted"]
    assert manager.peek_events("a", "ga", 2) == events_a
    manager.ack_events("a", "ga", 3)
    assert [event.seq_no for event in manager.peek_events("b", "gb", 2)] == [3]


def test_expired_generation_requires_new_full_report(small_block):
    manager, _ = _cached_manager(lease=0.001)
    manager.full_report_and_register("a", "g1")
    time.sleep(0.005)
    with pytest.raises(FullReportRequired):
        manager.peek_events("a", "g1", 2)
    with pytest.raises(ValueError):
        manager.full_report_and_register("a", "g1")
    assert manager.full_report_and_register("a", "g2").instance_epoch == "e1"


def test_resolve_pin_prevents_eviction_and_unpin_is_idempotent(small_block):
    manager, seq = _cached_manager()
    expected = [(manager.blocks[bid].hash, manager.blocks[bid].token_ids) for bid in seq.block_table[:2]]
    pinned = manager.resolve_and_pin_prefix(
        "op1", expected, namespace="ns", kv_compatibility_id="compat",
        request_context_digest="text",
    )
    manager.deallocate(seq)
    assert pinned is not None
    assert manager._num_available() == 6
    assert manager.unpin_prefix("op1") is True
    assert manager.unpin_prefix("op1") is False
    assert manager._num_available() == 8


def test_resolve_rejects_compatibility_mismatch_without_partial_pin(small_block):
    manager, seq = _cached_manager()
    expected = [(manager.blocks[seq.block_table[0]].hash, [1, 2, 3, 4])]
    assert manager.resolve_and_pin_prefix(
        "op1", expected, namespace="ns", kv_compatibility_id="wrong",
        request_context_digest="text",
    ) is None
    assert manager._transfer_pins == {}


def test_full_report_cannot_observe_evicted_metadata_after_watermark(small_block):
    manager = BlockManager(1, 4, instance_id="d0", instance_epoch="e1")
    seq = Sequence([1, 2, 3, 4])
    manager.allocate(seq, 0)
    seq.num_scheduled_tokens = seq.num_tokens
    manager.hash_blocks(seq, namespace="ns", kv_compatibility_id="c")
    manager.deallocate(seq)
    block = manager.blocks[0]
    original_reset = block.reset
    entered = threading.Event()
    release = threading.Event()

    def blocked_reset():
        entered.set()
        release.wait(timeout=2)
        original_reset()

    block.reset = blocked_reset
    allocator = threading.Thread(target=manager._allocate_block)
    allocator.start()
    assert entered.wait(timeout=1)
    result = []
    reporter = threading.Thread(
        target=lambda: result.append(manager.full_report_and_register("g", "gen"))
    )
    reporter.start()
    reporter.join(timeout=0.05)
    assert reporter.is_alive(), "full report escaped the eviction/reset lock"
    release.set()
    allocator.join(timeout=1)
    reporter.join(timeout=1)
    assert result[0].locations == ()
