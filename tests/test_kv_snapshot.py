import time
import pytest

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.sequence import Sequence, SequenceStatus
from prism_infer.engine.kv_snapshot import (
    SnapHandle,
    SnapshotReq,
    MigrationStatus,
    MigrationWatchdog,
    snapshot_sequence,
    incremental_snapshot,
    apply_snapshot,
    pre_alloc_blocks,
    commit_migration,
    free_pre_alloc,
    reset_to_waiting,
    resume_after_abort,
    MIGRATION_IN_TIMEOUT_S,
)
from prism_infer.sampling_params import SamplingParams


def _make_seq(num_tokens: int = 8, num_cached: int = 8,
              num_scheduled: int = 0) -> Sequence:
    sp = SamplingParams()
    seq = Sequence(list(range(num_tokens)), sp)
    seq.num_cached_tokens = num_cached
    seq.num_scheduled_tokens = num_scheduled
    block_size = Sequence.block_size
    num_blocks = (num_tokens + block_size - 1) // block_size
    seq.block_table = list(range(num_blocks))
    return seq


class _MockBlockManager:
    """Minimal BlockManager mock with the same public surface as the real one."""
    def __init__(self, total_blocks: int = 64):
        self._free = list(range(total_blocks))
        self.freed = []

    def _allocate_block(self):
        return self._free.pop(0) if self._free else None

    def release_block(self, block_id: int):
        self._free.append(block_id)
        self.freed.append(block_id)


def test_snapshot_aligned_basic():
    seq = _make_seq(num_tokens=8, num_cached=8, num_scheduled=0)
    seq.block_table = [10, 11]
    handle = snapshot_sequence(seq, allow_unaligned=False)
    assert handle.seq_id == str(seq.seq_id)
    assert handle.block_table == [10, 11]
    assert handle.token_ids == list(range(8))
    assert handle.num_cached_tokens == 8
    assert handle.is_unaligned is False
    assert handle.inflight_token_ids == []
    assert handle.delta_block_ids == []
    assert handle.created_at_ms > 0


def test_snapshot_aligned_deep_copy():
    seq = _make_seq()
    handle = snapshot_sequence(seq)
    original = list(handle.block_table)
    seq.block_table.append(999)
    seq.token_ids.append(999)
    assert handle.block_table == original
    assert 999 not in handle.token_ids


def test_snapshot_aligned_requires_zero_scheduled():
    seq = _make_seq(num_tokens=8, num_cached=4, num_scheduled=4)
    with pytest.raises(AssertionError, match="num_scheduled_tokens==0"):
        snapshot_sequence(seq, allow_unaligned=False)


def test_incremental_requires_aligned():
    seq = _make_seq(num_tokens=8, num_cached=4, num_scheduled=4)
    with pytest.raises(AssertionError, match="aligned state"):
        incremental_snapshot(seq, base_blocks=[0])


def test_snapshot_unaligned_captures_inflight():
    seq = _make_seq(num_tokens=8, num_cached=4, num_scheduled=4)
    handle = snapshot_sequence(seq, allow_unaligned=True)
    assert handle.is_unaligned is True
    assert handle.inflight_token_ids == list(range(4, 8))
    assert handle.num_cached_tokens == 4


def test_snapshot_unaligned_no_inflight():
    seq = _make_seq(num_tokens=8, num_cached=8, num_scheduled=0)
    handle = snapshot_sequence(seq, allow_unaligned=True)
    assert handle.is_unaligned is False
    assert handle.inflight_token_ids == []


def test_incremental_snapshot_only_delta():
    seq = _make_seq(num_tokens=8, num_cached=8, num_scheduled=0)
    seq.block_table = [0, 1, 2, 3, 4, 5]
    handle = incremental_snapshot(seq, base_blocks=[0, 1, 2, 3])
    assert handle.delta_block_ids == [4, 5]
    assert handle.base_block_table == [0, 1, 2, 3]
    assert handle.block_table == [0, 1, 2, 3, 4, 5]


def test_incremental_snapshot_empty_delta():
    seq = _make_seq(num_tokens=8, num_cached=8, num_scheduled=0)
    seq.block_table = [0, 1, 2, 3]
    handle = incremental_snapshot(seq, base_blocks=[0, 1, 2, 3])
    assert handle.delta_block_ids == []


def test_apply_snapshot_aligned():
    sp = SamplingParams()
    handle = SnapHandle(
        seq_id="42",
        block_table=[0, 1],
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        num_cached_tokens=8,
        is_unaligned=False,
    )
    bm = _MockBlockManager(total_blocks=16)
    seq = apply_snapshot(handle, bm, sp)
    assert seq.status == SequenceStatus.KV_TRANSFERRING
    assert len(seq.block_table) == 2
    assert seq.num_cached_tokens == 8
    assert seq.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_apply_snapshot_unaligned():
    sp = SamplingParams()
    handle = SnapHandle(
        seq_id="99",
        block_table=[5],
        token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        num_cached_tokens=4,
        is_unaligned=True,
        inflight_token_ids=[5, 6, 7, 8],
    )
    bm = _MockBlockManager(total_blocks=16)
    seq = apply_snapshot(handle, bm, sp)
    assert seq.status == SequenceStatus.WAITING
    assert seq.token_ids == [1, 2, 3, 4]
    assert seq.num_tokens == 4
    assert seq.num_cached_tokens == 4


def test_apply_snapshot_reuses_preallocated_blocks():
    """commit_migration must reuse the blocks allocated in Step 2 (not re-allocate)."""
    sp = SamplingParams()
    handle = SnapHandle(
        seq_id="42",
        block_table=[10, 11],
        token_ids=list(range(8)),
        num_cached_tokens=8,
    )
    bm = BlockManager(num_blocks=4, block_size=4)
    dst_blocks = pre_alloc_blocks("42", 2, handle.token_ids, bm)
    assert dst_blocks is not None

    seq = apply_snapshot(handle, bm, sp, dst_blocks=dst_blocks)

    assert seq.block_table == dst_blocks
    assert len(bm.used_block_ids) == 2
    assert len(bm.free_block_ids) == 2


def test_apply_snapshot_rejects_wrong_block_count():
    """Wrong number of pre-allocated blocks must raise AssertionError."""
    handle = SnapHandle(
        seq_id="42",
        block_table=[10, 11],
        token_ids=list(range(8)),
        num_cached_tokens=8,
    )
    bm = _MockBlockManager(total_blocks=4)

    with pytest.raises(AssertionError, match="block count mismatch"):
        apply_snapshot(handle, bm, SamplingParams(), dst_blocks=[0])


def test_apply_snapshot_oom():
    sp = SamplingParams()
    handle = SnapHandle(
        seq_id="1",
        block_table=[0, 1, 2, 3, 4],
        token_ids=list(range(10)),
        num_cached_tokens=10,
    )
    bm = _MockBlockManager(total_blocks=2)
    with pytest.raises(AssertionError, match="OOM"):
        apply_snapshot(handle, bm, sp)


def test_pre_alloc_blocks_real_manager_oom_rollback():
    """Exhausting the real BlockManager must roll back without leaking blocks."""
    bm = BlockManager(num_blocks=2, block_size=4)

    result = pre_alloc_blocks("seq_oom", 3, list(range(8)), bm)

    assert result is None
    assert len(bm.free_block_ids) == 2
    assert not bm.used_block_ids


def test_free_pre_alloc_real_manager():
    """free_pre_alloc on a real BlockManager returns the block to the free pool."""
    bm = BlockManager(num_blocks=2, block_size=4)
    block_id = bm._allocate_block()

    free_pre_alloc([block_id], bm)

    assert len(bm.free_block_ids) == 2
    assert not bm.used_block_ids


def test_watchdog_timeout_releases_blocks():
    bm = _MockBlockManager(total_blocks=16)
    watchdog = MigrationWatchdog(bm)
    blocks = [0, 1, 2]
    watchdog.register("seq_1", blocks)
    # Backdate the timestamp to simulate timeout
    watchdog._pending["seq_1"] = (blocks, time.monotonic() - MIGRATION_IN_TIMEOUT_S - 1)
    now = time.monotonic()
    expired = [
        sid for sid, (_, ts) in watchdog._pending.items()
        if now - ts > MIGRATION_IN_TIMEOUT_S
    ]
    for sid in expired:
        blks, _ = watchdog._pending.pop(sid)
        free_pre_alloc(blks, bm)
    assert "seq_1" not in watchdog._pending
    assert set(blocks).issubset(set(bm.freed))


def test_watchdog_commit_removes_from_pending():
    bm = _MockBlockManager()
    watchdog = MigrationWatchdog(bm)
    watchdog.register("seq_2", [5, 6])
    watchdog.commit("seq_2")
    assert "seq_2" not in watchdog._pending


def test_watchdog_pending_blocks_returns_correct_ids():
    """pending_blocks() returns the registered block list before commit."""
    bm = _MockBlockManager()
    watchdog = MigrationWatchdog(bm)
    watchdog.register("seq_3", [10, 11, 12])
    assert watchdog.pending_blocks("seq_3") == [10, 11, 12]
    watchdog.commit("seq_3")
    assert watchdog.pending_blocks("seq_3") is None


def test_free_pre_alloc_releases_blocks():
    bm = _MockBlockManager(total_blocks=16)
    blocks = [3, 4, 5]
    free_pre_alloc(blocks, bm)
    assert set(blocks).issubset(set(bm.freed))


def test_pre_alloc_blocks_success():
    bm = _MockBlockManager(total_blocks=16)
    result = pre_alloc_blocks("seq_test", 3, list(range(8)), bm)
    assert result is not None
    assert len(result) == 3


def test_pre_alloc_blocks_oom():
    bm = _MockBlockManager(total_blocks=2)
    result = pre_alloc_blocks("seq_oom", 5, list(range(8)), bm)
    assert result is None
    assert len(bm._free) == 2


def test_commit_migration_aligned():
    sp = SamplingParams()
    handle = SnapHandle(
        seq_id="7",
        block_table=[0],
        token_ids=[1, 2, 3],
        num_cached_tokens=3,
        is_unaligned=False,
    )
    bm = _MockBlockManager(total_blocks=16)
    seq = apply_snapshot(handle, bm, sp)
    commit_migration("7", handle, seq)
    assert seq.status == SequenceStatus.RUNNING


def test_commit_migration_unaligned():
    sp = SamplingParams()
    handle = SnapHandle(
        seq_id="8",
        block_table=[0],
        token_ids=[1, 2, 3, 4],
        num_cached_tokens=2,
        is_unaligned=True,
        inflight_token_ids=[3, 4],
    )
    bm = _MockBlockManager(total_blocks=16)
    seq = apply_snapshot(handle, bm, sp)
    commit_migration("8", handle, seq)
    assert seq.status == SequenceStatus.WAITING


def test_reset_to_waiting():
    seq = _make_seq()
    seq.status = SequenceStatus.KV_TRANSFERRING
    reset_to_waiting(seq)
    assert seq.status == SequenceStatus.WAITING


def test_resume_after_abort():
    seq = _make_seq()
    seq.status = SequenceStatus.MIGRATING_OUT
    resume_after_abort(seq)
    assert seq.status == SequenceStatus.RUNNING
