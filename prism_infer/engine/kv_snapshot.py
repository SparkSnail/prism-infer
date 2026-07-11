from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from prism_infer.engine.sequence import Sequence
    from prism_infer.engine.block_manager import BlockManager
    from prism_infer.sampling_params import SamplingParams


@dataclass
class SnapHandle:
    """Opaque snapshot handle passed between serve and infer.

    Contains only metadata (block ids, token ids) -- no KV tensors.
    KV data stays on the src GPU and is transferred separately via
    NCCLTransport.send_kv() in Step 3 of the three-way handshake.
    """
    seq_id: str
    block_table: List[int]
    token_ids: List[int]
    num_cached_tokens: int
    is_unaligned: bool = False
    inflight_token_ids: List[int] = field(default_factory=list)
    # unaligned snapshot: tokens in-flight at snapshot time; dst must re-prefill these
    base_block_table: List[int] = field(default_factory=list)
    delta_block_ids: List[int] = field(default_factory=list)
    created_at_ms: int = 0


@dataclass
class SnapshotReq:
    """Snapshot instruction from serve to infer (see interface contract 03)."""
    op_id: str
    seq_id: str
    mode: str = "aligned"       # "aligned" | "unaligned"
    incremental: bool = False
    base_blocks: List[int] = field(default_factory=list)


class MigrationStatus:
    RUNNING     = "RUNNING"
    ABORTED_DST = "ABORTED_DST"   # dst OOM -- src keeps the sequence
    ABORTED_SRC = "ABORTED_SRC"   # src sequence finished/preempted -- dst frees pre-alloc
    FINISHED    = "FINISHED"


def snapshot_sequence(
    seq: "Sequence",
    allow_unaligned: bool = False,
) -> SnapHandle:
    """Capture a KV snapshot for migration (handshake Step 1).

    Must be called after postprocess() when allow_unaligned=False: only then has
    store_kvcache fully written the last block. Calling mid-step risks a half-written
    block on the destination.

    Args:
        seq: sequence to snapshot; must be RUNNING or MIGRATING_OUT
        allow_unaligned: False = wait for step boundary (safe);
                         True = snapshot immediately under back-pressure

    Returns:
        SnapHandle with metadata only (no KV tensors)
    """
    if not allow_unaligned:
        assert seq.num_scheduled_tokens == 0, (
            f"aligned snapshot requires num_scheduled_tokens==0 (call after postprocess); "
            f"got {seq.num_scheduled_tokens}. Use allow_unaligned=True during chunked prefill."
        )
        inflight = []
    else:
        start = seq.num_cached_tokens
        inflight = list(seq.token_ids[start:])

    return SnapHandle(
        seq_id=str(seq.seq_id),
        block_table=seq.block_table[:],
        token_ids=seq.token_ids[:],
        num_cached_tokens=seq.num_cached_tokens,
        is_unaligned=allow_unaligned and bool(inflight),
        inflight_token_ids=inflight,
        created_at_ms=int(time.time() * 1000),
    )


def incremental_snapshot(
    seq: "Sequence",
    base_blocks: List[int],
) -> SnapHandle:
    """Incremental snapshot: only the blocks added since base_blocks.

    Must be called in aligned state (num_scheduled_tokens == 0); delta over a
    half-written block corrupts the destination.

    Useful for periodic KV replication where full retransfer would waste bandwidth.
    """
    assert seq.num_scheduled_tokens == 0, (
        "incremental_snapshot requires aligned state (num_scheduled_tokens==0)"
    )
    base_set = set(base_blocks)
    delta = [b for b in seq.block_table if b not in base_set]
    return SnapHandle(
        seq_id=str(seq.seq_id),
        block_table=seq.block_table[:],
        token_ids=seq.token_ids[:],
        num_cached_tokens=seq.num_cached_tokens,
        base_block_table=list(base_blocks),
        delta_block_ids=delta,
        created_at_ms=int(time.time() * 1000),
    )


def _try_allocate_blocks(
    block_num: int,
    block_manager: "BlockManager",
) -> Optional[List[int]]:
    """Transactional pre-allocation: roll back all allocations on any failure."""
    allocated: List[int] = []
    try:
        for _ in range(block_num):
            block_id = block_manager._allocate_block()
            if block_id is None:
                raise IndexError("block pool exhausted")
            allocated.append(block_id)
    except (KeyError, IndexError):
        for block_id in allocated:
            block_manager.release_block(block_id)
        return None
    return allocated


def apply_snapshot(
    handle: SnapHandle,
    block_manager: "BlockManager",
    sampling_params: "SamplingParams",
    dst_blocks: Optional[List[int]] = None,
) -> "Sequence":
    """Reconstruct a sequence on the destination from a SnapHandle (handshake Step 2).

    Allocates block slots only -- no KV data is written here.
    KV data is written by NCCLTransport.recv_kv() in Step 3.

    Args:
        handle:          SnapHandle from the source instance
        block_manager:   dst BlockManager
        sampling_params: needed to reconstruct Sequence
        dst_blocks:      pre-allocated block ids from pre_alloc_blocks (Step 2);
                         if None, allocates fresh blocks (standalone recovery path)

    Returns:
        Sequence with status KV_TRANSFERRING (aligned) or WAITING (unaligned)
    """
    from prism_infer.engine.sequence import Sequence, SequenceStatus

    seq = Sequence(handle.token_ids, sampling_params)
    seq.num_cached_tokens = handle.num_cached_tokens

    if handle.is_unaligned and handle.inflight_token_ids:
        # Truncate to cached portion; scheduler will re-feed the inflight tokens
        seq.status = SequenceStatus.WAITING
        seq.token_ids = seq.token_ids[:handle.num_cached_tokens]
        seq.num_tokens = handle.num_cached_tokens
    else:
        seq.status = SequenceStatus.KV_TRANSFERRING

    num_blocks = len(handle.block_table)
    if dst_blocks is None:
        # Standalone recovery: allocate fresh blocks
        dst_blocks = _try_allocate_blocks(num_blocks, block_manager)
        assert dst_blocks is not None, (
            f"dst OOM: cannot allocate {num_blocks} blocks for seq {handle.seq_id}"
        )
    assert len(dst_blocks) == num_blocks, (
        f"dst block count mismatch: expected {num_blocks}, got {len(dst_blocks)}"
    )
    seq.block_table.extend(dst_blocks)

    return seq


def pre_alloc_blocks(
    seq_id: str,
    block_num: int,
    token_ids: List[int],
    block_manager: "BlockManager",
) -> Optional[List[int]]:
    """Pre-allocate dst block slots (handshake Step 2).

    Returns dst_block_ids on success, None if OOM.
    On None, serve triggers ABORTED_DST and src keeps the sequence.
    """
    return _try_allocate_blocks(block_num, block_manager)


def commit_migration(
    seq_id: str,
    handle: SnapHandle,
    seq: "Sequence",
) -> None:
    """Transition dst sequence to active after KV transfer completes (handshake Step 3b).

    dst seq_id differs from handle.seq_id -- dst assigns a new local id via the
    global counter in Sequence.__init__. No cross-instance id consistency is enforced.
    """
    from prism_infer.engine.sequence import SequenceStatus

    if handle.is_unaligned:
        seq.status = SequenceStatus.WAITING
    else:
        seq.status = SequenceStatus.RUNNING


def free_pre_alloc(
    block_ids: List[int],
    block_manager: "BlockManager",
) -> None:
    """Release dst pre-allocated blocks on ABORTED_DST or ABORTED_SRC."""
    for b in block_ids:
        block_manager.release_block(b)


def reset_to_waiting(seq: "Sequence") -> None:
    """Revert a KV_TRANSFERRING sequence to WAITING for recompute fallback.

    Called by serve when KV transfer times out and on_fail="recompute".
    The decode instance re-runs prefill locally rather than waiting for KV.
    """
    from prism_infer.engine.sequence import SequenceStatus
    seq.status = SequenceStatus.WAITING


def resume_after_abort(seq: "Sequence") -> None:
    """Revert src sequence from MIGRATING_OUT to RUNNING after ABORTED_DST.

    Migration failed at dst allocation; src continues inference normally.
    """
    from prism_infer.engine.sequence import SequenceStatus
    seq.status = SequenceStatus.RUNNING


# 30s matches a conservative prefill-to-transfer upper bound for long sequences;
# keeps dst from leaking pre-alloc blocks if src crashes mid-handshake.
MIGRATION_IN_TIMEOUT_S = 30.0


class MigrationWatchdog:
    """Background watchdog on dst: release pre-alloc blocks if handshake times out.

    Mirrors Llumnix _watch_pending_migrate_in_requests_loop.

    Usage:
        watchdog = MigrationWatchdog(block_manager)
        asyncio.create_task(watchdog.watch_loop())  # started in LLMEngine.__init__
        watchdog.register(seq_id, dst_blocks)       # after Step 2 pre_alloc
        watchdog.commit(seq_id)                     # after Step 3 commit
    """

    def __init__(self, block_manager: "BlockManager"):
        self.block_manager = block_manager
        self._pending: Dict[str, tuple] = {}  # seq_id -> (blocks, timestamp)

    def register(self, seq_id: str, blocks: List[int]) -> None:
        self._pending[seq_id] = (blocks, time.monotonic())

    def commit(self, seq_id: str) -> None:
        self._pending.pop(seq_id, None)

    def pending_blocks(self, seq_id: str) -> Optional[List[int]]:
        """Return the pre-allocated block ids registered for seq_id, or None."""
        entry = self._pending.get(seq_id)
        return list(entry[0]) if entry is not None else None

    async def watch_loop(self) -> None:
        while True:
            await asyncio.sleep(MIGRATION_IN_TIMEOUT_S / 2)
            now = time.monotonic()
            expired = [
                sid for sid, (_, ts) in self._pending.items()
                if now - ts > MIGRATION_IN_TIMEOUT_S
            ]
            for sid in expired:
                blocks, _ = self._pending.pop(sid)
                free_pre_alloc(blocks, self.block_manager)
