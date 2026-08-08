"""Production control-plane bindings for the engine owner thread.

The HTTP and NATS frontends submit immutable commands here.  CUDA,
BlockManager and scheduler mutations are executed by one owner thread.
"""

from concurrent.futures import Future
import asyncio
from dataclasses import asdict, dataclass, replace
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Callable
import time

from prism_infer.sampling_params import SamplingParams
from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    OperationSnapshot,
    OperationState,
    RequestOutputNotFound,
)


CONTROLLED_RESOURCE_KINDS = (
    "SOURCE_RETAIN",
    "SOURCE_PIN",
    "TARGET_PENDING",
    "TARGET_SEQUENCE",
    "SOURCE_BLOCKS",
    "TRANSFER_BYTES",
)


def endpoint_ref_key(ref: EndpointOperationRef) -> str:
    return f"{ref.owner_generation}:{ref.operation_seq}:{ref.payload_digest}"


def build_block_resource_snapshot(
    block_manager,
    *,
    pending_block_ids: set[int] | None = None,
    quarantined_block_ids: set[int] | None = None,
    pinned_block_ids: set[int] | None = None,
) -> dict[str, object]:
    """Partition allocator blocks into mutually exclusive ownership buckets.

    Pending and quarantined blocks come from operation owners. Referenced or
    pinned blocks that are not in those sets remain sequence-owned.
    """
    pending_claim = set(pending_block_ids or ())
    quarantined_claim = set(quarantined_block_ids or ())
    pinned_claim = set(pinned_block_ids or ())
    with block_manager._prefix_state_lock:
        total = len(block_manager.blocks)
        all_blocks = set(range(total))
        raw_free = set(block_manager.free_block_ids)
        raw_evictable = set(block_manager.evictable)
        raw_used = set(block_manager.used_block_ids)

        assert not raw_free & raw_evictable, "free and evictable blocks overlap"
        assert not raw_free & raw_used, "free and used blocks overlap"
        assert not raw_evictable & raw_used, "evictable and used blocks overlap"
        assert raw_free | raw_evictable | raw_used == all_blocks, (
            "BlockManager physical block partition is incomplete"
        )
        claimed = pending_claim | quarantined_claim | pinned_claim
        assert claimed <= raw_used | raw_evictable, (
            f"operation claims non-owned blocks: {sorted(claimed - raw_used - raw_evictable)}"
        )

        quarantined = quarantined_claim
        pending = pending_claim - quarantined
        active_pins = pinned_claim - quarantined
        assert not pending & active_pins, (
            "the same physical block cannot be pending and source-pinned"
        )
        evictable = raw_evictable - quarantined - active_pins
        free = raw_free
        sequence = all_blocks - free - pending - evictable - quarantined

        buckets = {
            "free": len(free),
            "pending": len(pending),
            "sequence": len(sequence),
            "evictable": len(evictable),
            "quarantined": len(quarantined),
        }
        assert sum(buckets.values()) == total, (
            f"GPU block conservation violated: {buckets!r}, total={total}"
        )
        return {
            "num_gpu_blocks": total,
            "free_blocks": len(free),
            "block_buckets": buckets,
            "block_conservation_valid": True,
        }


class _ImmediateWork:
    def is_completed(self) -> bool:
        return True


class _ImmediateEvent:
    def query(self) -> bool:
        return True


class _PendingLaunchWork:
    """Keep a scheduled receive UNKNOWN until NCCL returns real Work handles."""

    def is_completed(self) -> bool:
        return False


@dataclass(frozen=True)
class _WatchdogDeadline:
    endpoint_ref_key: str
    endpoint_ref: EndpointOperationRef
    pair_id: str
    started_at: float


class MappedNCCLEndpoint:
    """Bind independent endpoint refs to concrete NCCL Work/CUDA fences."""

    def __init__(
        self, pair_groups, kv_cache, *, watchdog_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        from prism_infer.engine.kv_transfer import EndpointTransferRegistry

        self.pair_groups = pair_groups
        self.kv_cache = kv_cache
        self.registry = EndpointTransferRegistry()
        self._pending_receive_launches: dict[
            str, tuple[Future, object, list[tuple[int, object]]]
        ] = {}
        self._receive_keys: set[str] = set()
        self._pending_receives: dict[str, tuple[list, list[tuple[int, object]]]] = {}
        self._receive_copy_enqueued: set[str] = set()
        # NCCL asynchronous sends borrow their input storage.  Keep every
        # contiguous staging tensor alive until both the Work handles and the
        # CUDA visibility event are terminal; SOURCE_RETAIN accounting alone
        # does not hold a Python reference to these tensors.
        self._pending_sends: dict[str, tuple[list, list[object]]] = {}
        self._discarded_receives: dict[
            str, tuple[list, list[tuple[int, object]], bool]
        ] = {}
        self._failed_receive_buffers: dict[str, list[tuple[int, object]]] = {}
        self._operation_keys: dict[str, str] = {}
        self._operation_refs: dict[str, EndpointOperationRef] = {}
        self._resource_quantities: dict[str, dict[str, int]] = {}
        self._transfer_metadata: dict[str, dict[str, object]] = {}
        if watchdog_timeout_s <= 0:
            raise ValueError("NCCL watchdog timeout must be positive")
        self.watchdog_timeout_s = watchdog_timeout_s
        self._clock = clock
        self._launched_at: dict[str, float] = {}
        # Deadline publication is independent from the engine-owner thread.
        # The watchdog reader never touches Work, CUDA, or the mutable transfer
        # registry, so it can still fail-stop the process if owner progress is
        # blocked inside a backend call.
        self._watchdog_lock = Lock()
        self._watchdog_deadlines: dict[str, _WatchdogDeadline] = {}
        self.termination_requested = False
        self.watchdog_reason = ""
        self.watchdog_evidence: dict[str, object] | None = None
        self._launch_errors: dict[str, str] = {}
        self._terminal_progress_failed: set[str] = set()

    def _register_watchdog_deadline(
        self,
        key: str,
        ref: EndpointOperationRef,
        pair_id: str,
        *,
        started_at: float | None = None,
    ) -> None:
        launched_at = self._clock() if started_at is None else started_at
        deadline = _WatchdogDeadline(key, ref, pair_id, launched_at)
        with self._watchdog_lock:
            existing = self._watchdog_deadlines.get(key)
            if existing is not None and existing != deadline:
                raise ValueError("NCCL watchdog deadline identity changed")
            self._watchdog_deadlines[key] = deadline
            self._launched_at[key] = launched_at

    def _clear_watchdog_deadline(self, key: str) -> None:
        with self._watchdog_lock:
            if self.termination_requested:
                # A process-wide fail-stop already linearized.  Keep every
                # in-flight record intact until os._exit even if a blocked
                # owner later wakes on a different endpoint.
                return
            self._watchdog_deadlines.pop(key, None)
            self._launched_at.pop(key, None)

    def _watchdog_timeout_latched(self, key: str) -> bool:
        with self._watchdog_lock:
            return bool(
                self.termination_requested
                and key in self._watchdog_deadlines
            )

    def _expired_deadline_locked(
        self, now: float,
    ) -> _WatchdogDeadline | None:
        expired = [
            deadline for deadline in self._watchdog_deadlines.values()
            if now - deadline.started_at >= self.watchdog_timeout_s
        ]
        if not expired:
            return None
        return min(
            expired,
            key=lambda value: (value.started_at, value.endpoint_ref_key),
        )

    def _latch_watchdog_timeout_locked(
        self, deadline: _WatchdogDeadline,
    ) -> None:
        key = deadline.endpoint_ref_key
        self.termination_requested = True
        self.watchdog_reason = f"NCCL operation watchdog expired: {key}"
        endpoint_error = self._launch_errors.get(key)
        if endpoint_error:
            self.watchdog_reason += f"; endpoint error: {endpoint_error}"
        self.watchdog_evidence = {
            "kind": "nccl_watchdog_timeout",
            "reason": self.watchdog_reason,
            "pair_id": deadline.pair_id,
            "endpoint_key": key,
            "endpoint_ref": asdict(deadline.endpoint_ref),
            "operation_id": deadline.endpoint_ref.operation_id,
            "watchdog_timeout_s": self.watchdog_timeout_s,
        }

    def _finalize_terminal_deadline(
        self, key: str, cleanup: Callable[[], None] | None = None,
    ) -> bool:
        """Atomically let terminal publication or the deadline win once."""

        now = self._clock()
        with self._watchdog_lock:
            already_terminal = (
                key not in self._watchdog_deadlines
                and key not in self._launched_at
            )
            if already_terminal:
                # This endpoint already won and published terminal state (or
                # never required a watchdog).  Replays stay terminal even if a
                # different in-flight endpoint later triggers process fail-stop.
                pass
            else:
                if self.termination_requested:
                    return False
                expired = self._expired_deadline_locked(now)
                if expired is not None:
                    self._latch_watchdog_timeout_locked(expired)
                    return False
                self._watchdog_deadlines.pop(key, None)
                self._launched_at.pop(key, None)
        # The lock protects only the terminal-vs-timeout claim.  Dropping the
        # final Python reference may synchronously destroy CUDA/NCCL objects,
        # so detach borrowed storage only after the watchdog lock is released.
        if cleanup is not None:
            cleanup()
        return True

    def _cleanup_terminal_storage(self, key: str) -> None:
        """Drop borrowed CUDA storage only at the completion linearization."""

        self._pending_receives.pop(key, None)
        self._receive_copy_enqueued.discard(key)
        self._pending_sends.pop(key, None)
        self._discarded_receives.pop(key, None)

    def _finalize_terminal_snapshot(self, key: str, snapshot) -> bool:
        if snapshot.status.value not in {"COMPLETED", "FENCED"} \
                or not snapshot.work_terminal \
                or not snapshot.cuda_visibility_terminal:
            return False
        return self._finalize_terminal_deadline(
            key, lambda: self._cleanup_terminal_storage(key)
        )

    def _record_launch_error(
        self, key: str, exc: BaseException, *, overwrite: bool = True,
    ) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        with self._watchdog_lock:
            if overwrite:
                self._launch_errors[key] = reason
            else:
                self._launch_errors.setdefault(key, reason)

    def _launch_error(self, key: str) -> str:
        with self._watchdog_lock:
            return self._launch_errors.get(key, "")

    def _record_terminal_progress_error(
        self, key: str, exc: BaseException,
    ) -> None:
        self._terminal_progress_failed.add(key)
        self._record_launch_error(key, exc, overwrite=False)

    def _unknown_status(self, key: str):
        from prism_infer.engine.kv_transfer import (
            EndpointFenceSnapshot,
            EndpointFenceStatus,
        )

        operation = self.registry._operations[key]
        return EndpointFenceSnapshot(
            endpoint_ref_key=key,
            status=EndpointFenceStatus.UNKNOWN,
            work_terminal=operation.work_terminal,
            cuda_visibility_terminal=operation.cuda_visibility_terminal,
            resources_held=bool(operation.held_resource_kinds),
            held_resource_kinds=operation.held_resource_kinds,
        )

    def _safe_status(self, key: str):
        # A terminal-progress exception makes the endpoint permanently
        # uncertain for this process incarnation.  Never query it again: a
        # transiently successful second query could otherwise drop the
        # watchdog deadline while retained send/receive storage still exists.
        if self._watchdog_timeout_latched(key) \
                or key in self._terminal_progress_failed:
            return self._unknown_status(key)
        try:
            snapshot = self.registry.status(key)
        except BaseException as exc:
            self._record_terminal_progress_error(key, exc)
            return self._unknown_status(key)
        # A Work/Event query may itself block past the independent deadline.
        # Timeout wins unless terminal publication already cleared the record.
        if self._watchdog_timeout_latched(key):
            return self._unknown_status(key)
        return snapshot

    def _published_status(self, key: str):
        """Publish terminal state only after winning the deadline race."""

        snapshot = self._safe_status(key)
        if snapshot.status.value in {"COMPLETED", "FENCED"} \
                and snapshot.work_terminal \
                and snapshot.cuda_visibility_terminal \
                and not self._finalize_terminal_snapshot(key, snapshot):
            # The raw CUDA query became terminal, but an expired deadline (or
            # another process-wide fail-stop) won before publication.  Never
            # expose a terminal state that can later flip back to UNKNOWN.
            return self._unknown_status(key)
        return snapshot

    def _group(self, payload: dict[str, object]):
        source = str(payload["source_instance"])
        target = str(payload["target_instance"])
        pair_id = "--".join(sorted((source, target), key=lambda value: (value[0] == "d", value)))
        if {source, target} == {"d0", "d1"}:
            pair_id = "d0--d1"
        return self.pair_groups.pair(pair_id), pair_id

    def _peer(self, pair_id: str) -> int:
        # torch.distributed P2POp peer is a global rank even when ``group`` is
        # supplied.  PairGroupRegistry keeps the group-local conversion only
        # for APIs that explicitly request it.
        return self.pair_groups.global_peer(pair_id)

    def _completed_bytes(
        self, payload: dict[str, object], *, local_block_key: str
    ) -> int:
        source_ids = payload.get("src_block_ids")
        target_ids = payload.get("dst_block_ids")
        if source_ids is not None and target_ids is not None and (
            not isinstance(source_ids, (list, tuple))
            or not isinstance(target_ids, (list, tuple))
            or len(source_ids) != len(target_ids)
            or any(type(block_id) is not int for block_id in source_ids)
            or any(type(block_id) is not int for block_id in target_ids)
            or len(set(source_ids)) != len(source_ids)
            or len(set(target_ids)) != len(target_ids)
        ):
            raise ValueError("mapped transfer requires a bijective block mapping")
        block_ids = payload.get(local_block_key)
        if not isinstance(block_ids, (list, tuple)) or not block_ids \
                or self.kv_cache.shape[2] <= 0 \
                or any(
                    type(block_id) is not int
                    or not 0 <= block_id < self.kv_cache.shape[2]
                    for block_id in block_ids
                ):
            raise ValueError("mapped transfer cannot attest completed bytes")
        block_count = len(block_ids)
        block_bytes = (
            self.kv_cache[:, :, 0].numel() * self.kv_cache.element_size()
        )
        observed = block_count * block_bytes
        explicit = payload.get("kv_size_bytes")
        if explicit is not None and (type(explicit) is not int or explicit != observed):
            raise ValueError("declared transfer bytes do not match mapped KV slices")
        return observed

    def prepare_receive(
        self, ref: EndpointOperationRef, payload: dict[str, object]
    ) -> OperationSnapshot:
        import torch
        import torch.distributed as dist

        key = endpoint_ref_key(ref)
        block_ids = [int(value) for value in payload["dst_block_ids"]]
        pair, pair_id = self._group(payload)
        completed_bytes = self._completed_bytes(
            payload, local_block_key="dst_block_ids"
        )
        self._operation_keys[ref.operation_id] = key
        self._operation_refs[key] = ref
        self._receive_keys.add(key)
        # Destination blocks remain owned by request/prefix prepare.  This
        # endpoint owns only the writer fence, not a second block allocation.
        self.registry.register(key, resource_kinds=())
        if key in self._pending_receive_launches \
                or key in self._pending_receives \
                or key in self._discarded_receives \
                or self.registry._operations[key].launched:
            return self.refresh(ref)
        self._transfer_metadata[key] = {
            "pair_id": pair_id,
            "completed_bytes": completed_bytes,
        }
        gpu_transfer = dist.is_initialized() and self.kv_cache.is_cuda
        if not gpu_transfer:
            works, event = [_ImmediateWork()], _ImmediateEvent()
            self.registry.mark_launched(key, works, event)
            self._register_watchdog_deadline(key, ref, pair_id)
            self.registry.mark_work_terminal(key)
            self.registry.mark_completion_event_recorded(key)
            self.registry.mark_data_complete(key)
            self._finalize_terminal_deadline(key)
            return self.refresh(ref)
        # Publish the exact deadline before the first allocation/event/P2P
        # action that may enter the CUDA backend.  The out-of-band watchdog can
        # therefore terminate a process even when preparation itself wedges.
        self._register_watchdog_deadline(key, ref, pair_id)
        buffers = []
        ops = []
        try:
            peer = self._peer(pair.pair_id)
            for block_id in block_ids:
                buffer = torch.empty_like(
                    self.kv_cache[:, :, block_id, :, :, :]
                )
                buffers.append((block_id, buffer))
                ops.append(dist.P2POp(
                    dist.irecv, buffer, peer=peer, group=pair.process_group
                ))
            event = torch.cuda.Event()
        except BaseException:
            # No grouped call has started, so this endpoint has no ambiguous
            # NCCL writer and its deadline can be withdrawn safely.
            self._clear_watchdog_deadline(key)
            raise
        if self._watchdog_timeout_latched(key):
            raise RuntimeError("NCCL watchdog expired during receive staging")
        launch: Future = Future()
        entered_launch = Event()

        def launch_receive() -> None:
            # NCCL irecv may block until the peer posts isend. Only the blocking
            # launch leaves the owner thread; Work adoption and KV copy stay on it.
            entered_launch.set()
            try:
                launch.set_result(dist.batch_isend_irecv(ops))
            except BaseException as exc:
                launch.set_exception(exc)

        # A scheduled launcher is already a possible writer.  Represent it by
        # a non-terminal Work placeholder so abort/prune cannot falsely report
        # FENCED while NCCL may still produce a live receive operation.
        self.registry.mark_launched(key, [_PendingLaunchWork()], event)
        self._pending_receive_launches[key] = (launch, event, buffers)
        Thread(
            target=launch_receive,
            name=f"prism-nccl-recv-{ref.operation_seq}",
            daemon=True,
        ).start()
        # Entry into the NCCL call is enough to ACK the target before source launch.
        entered_launch.wait()
        return self.refresh(ref)

    def _adopt_receive_launch(self, key: str) -> None:
        pending = self._pending_receive_launches.get(key)
        if pending is None or not pending[0].done():
            return
        launch, event, buffers = pending
        if self._watchdog_timeout_latched(key):
            # Timeout already owns the state/resource decision.  A completed
            # launcher may still contribute diagnostic error text, but it must
            # never replace the placeholder, publish completion, or release
            # any retained receive storage.
            try:
                if not launch.result():
                    raise RuntimeError("NCCL receive returned no Work handles")
            except BaseException as exc:
                self._failed_receive_buffers.setdefault(key, buffers)
                self._record_launch_error(key, exc)
            return
        try:
            works = launch.result()
            # PyTorch may coalesce a P2P batch into grouped Work handles. All
            # returned handles must terminate before copying any receive buffer.
            if not works:
                raise RuntimeError("NCCL receive returned no Work handles")
            if self._watchdog_timeout_latched(key):
                return
            self._pending_receive_launches.pop(key, None)
            operation = self.registry._operations[key]
            # These handles replace a launch that was accepted before an abort;
            # they are not new work.  Adoption therefore remains legal after
            # accepting_new_work became false.
            operation.work_handles = tuple(works)
            operation.completion_event = event
            if operation.accepting_new_work:
                self._pending_receives[key] = (works, buffers)
            else:
                self._discarded_receives[key] = (works, buffers, False)
        except BaseException as exc:
            if self._watchdog_timeout_latched(key):
                return
            self._pending_receive_launches.pop(key, None)
            # A throwing grouped launch does not prove that NCCL submitted no
            # partial work.  Retain every temporary buffer and the UNKNOWN
            # placeholder until the watchdog terminates this process.
            self._failed_receive_buffers[key] = buffers
            self._record_launch_error(key, exc)

    def _advance_discarded_receive(self, key: str) -> None:
        if self._watchdog_timeout_latched(key):
            return
        pending = self._discarded_receives.get(key)
        if pending is None:
            return
        works, buffers, event_recorded = pending
        if not event_recorded and all(work.is_completed() for work in works):
            for work in works:
                work.wait()
                if self._watchdog_timeout_latched(key):
                    return
            if self._watchdog_timeout_latched(key):
                return
            self.registry.mark_work_terminal(key)
            event = self.registry._operations[key].completion_event
            event.record()
            if self._watchdog_timeout_latched(key):
                return
            self.registry.mark_completion_event_recorded(key)
            event_recorded = True
            self._discarded_receives[key] = (works, buffers, True)
        snapshot = self._safe_status(key)
        # The owner may observe terminal here, but storage is released only by
        # ``_finalize_terminal_snapshot`` under the watchdog lock.
        del snapshot

    def _advance_terminal_fences_unchecked(self, key: str) -> None:
        """Advance accepted NCCL work without depending on a Gateway query.

        Gateway is a control-plane observer, not the owner of CUDA/NCCL
        progress.  In particular, killing the Gateway after both endpoint
        mutations were accepted must not leave a completed receive behind a
        false watchdog timeout merely because nobody called ``refresh``.
        This helper runs only on the engine-owner thread (HTTP query or the
        serialized watchdog tick), so buffer copy and CUDA event recording
        remain single-writer operations.
        """
        self._adopt_receive_launch(key)
        if self._watchdog_timeout_latched(key):
            return
        self._advance_discarded_receive(key)
        if self._watchdog_timeout_latched(key):
            return
        pending = self._pending_receives.get(key)
        if pending is not None \
                and key not in self._receive_copy_enqueued \
                and all(work.is_completed() for work in pending[0]):
            # ``is_completed`` is only a readiness check.  Wait every returned
            # Work before copying temporary receive buffers into authoritative
            # KV storage, then publish the CUDA visibility fence.
            for work in pending[0]:
                work.wait()
                if self._watchdog_timeout_latched(key):
                    return
            if self._watchdog_timeout_latched(key):
                return
            self.registry.mark_work_terminal(key)
            for block_id, buffer in pending[1]:
                self.kv_cache[:, :, block_id, :, :, :].copy_(buffer)
                if self._watchdog_timeout_latched(key):
                    return
            event = self.registry._operations[key].completion_event
            event.record()
            if self._watchdog_timeout_latched(key):
                return
            self.registry.mark_completion_event_recorded(key)
            self.registry.mark_data_complete(key)
            # ``event.record`` only enqueues the visibility fence.  Retain the
            # temporary buffers until ``event.query`` is terminal and the
            # completion/deadline race is decided atomically.
            self._receive_copy_enqueued.add(key)
        operation = self.registry._operations[key]
        if key not in self._receive_keys and operation.launched \
                and not operation.data_complete and operation.work_handles \
                and all(work.is_completed() for work in operation.work_handles):
            # Source sends keep their contiguous slices alive through the same
            # Work/CUDA fence even when the Gateway process disappears.
            for work in operation.work_handles:
                work.wait()
                if self._watchdog_timeout_latched(key):
                    return
            if self._watchdog_timeout_latched(key):
                return
            self.registry.mark_work_terminal(key)
            operation.completion_event.record()
            if self._watchdog_timeout_latched(key):
                return
            self.registry.mark_completion_event_recorded(key)
            self.registry.mark_data_complete(key)

    def _advance_terminal_fences(self, key: str) -> None:
        """Keep per-operation polling faults inside the watchdog boundary."""

        if self._watchdog_timeout_latched(key):
            self._adopt_receive_launch(key)
            return
        if key in self._terminal_progress_failed:
            return
        try:
            self._advance_terminal_fences_unchecked(key)
        except BaseException as exc:
            # Polling is evidence acquisition, not permission to discard CUDA
            # storage.  Any query/wait/copy/event error leaves the endpoint
            # UNKNOWN and all buffers held until watchdog fail-stop.
            self._record_terminal_progress_error(key, exc)

    def start_transfer(
        self, ref: EndpointOperationRef, payload: dict[str, object]
    ) -> OperationSnapshot:
        import torch
        import torch.distributed as dist

        key = endpoint_ref_key(ref)
        block_count = len(payload["src_block_ids"])
        pair, pair_id = self._group(payload)
        completed_bytes = self._completed_bytes(
            payload, local_block_key="src_block_ids"
        )
        gpu_transfer = dist.is_initialized() and self.kv_cache.is_cuda
        slices: list[object] = []
        ops: list[object] = []
        event: object | None = None
        if gpu_transfer:
            # Publish only the immutable fail-stop deadline before staging.
            # Operation/resource ownership keeps its old commit point after
            # synchronous staging succeeds, so a local preparation exception
            # leaves no registry or accounting residue.
            self._register_watchdog_deadline(key, ref, pair_id)
            try:
                peer = self._peer(pair.pair_id)
                slices = [
                    self.kv_cache[:, :, int(block_id), :, :, :].contiguous()
                    for block_id in payload["src_block_ids"]
                ]
                ops = [
                    dist.P2POp(
                        dist.isend, value, peer=peer, group=pair.process_group
                    )
                    for value in slices
                ]
                # Construct every local fence before submission.  Once the
                # grouped NCCL call begins, a raised exception cannot prove
                # that no send was launched, so all staging tensors must have
                # a durable in-process owner and watchdog path first.
                event = torch.cuda.Event()
            except BaseException:
                self._clear_watchdog_deadline(key)
                raise
            if self._watchdog_timeout_latched(key):
                raise RuntimeError("NCCL watchdog expired during source staging")
        try:
            self._operation_keys[ref.operation_id] = key
            self._operation_refs[key] = ref
            self.registry.register(
                key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
            )
            self._resource_quantities[key] = {
                "SOURCE_RETAIN": block_count,
                "TRANSFER_BYTES": completed_bytes,
            }
            self._transfer_metadata[key] = {
                "pair_id": pair_id,
                "completed_bytes": completed_bytes,
            }
            if not gpu_transfer:
                self.registry.mark_launched(
                    key, [_ImmediateWork()], _ImmediateEvent()
                )
                self._register_watchdog_deadline(key, ref, pair_id)
                self.registry.mark_work_terminal(key)
                self.registry.mark_completion_event_recorded(key)
                self.registry.mark_data_complete(key)
                self._finalize_terminal_deadline(key)
                return self.refresh(ref)
            assert event is not None
            placeholder = _PendingLaunchWork()
            self.registry.mark_launched(key, [placeholder], event)
            self._pending_sends[key] = ([placeholder], slices)
        except BaseException:
            # No grouped NCCL call has begun, so every local publication above
            # is still reversible.  Roll it back in reverse ownership order;
            # after ``batch_isend_irecv`` starts, the separate UNKNOWN path
            # intentionally retains all state until process fail-stop.
            self._pending_sends.pop(key, None)
            self._transfer_metadata.pop(key, None)
            self._resource_quantities.pop(key, None)
            self.registry._operations.pop(key, None)
            self._operation_refs.pop(key, None)
            if self._operation_keys.get(ref.operation_id) == key:
                self._operation_keys.pop(ref.operation_id, None)
            self._clear_watchdog_deadline(key)
            raise
        try:
            works = dist.batch_isend_irecv(ops)
            if not works:
                raise RuntimeError("NCCL send returned no Work handles")
            self.registry._operations[key].work_handles = tuple(works)
            self._pending_sends[key] = (works, slices)
        except BaseException as exc:
            # A grouped launch error is an UNKNOWN write state.  Retain every
            # borrowed tensor behind the non-terminal placeholder until the
            # watchdog fail-stops this process; never return the storage to the
            # CUDA allocator based on a local exception.
            self._record_launch_error(key, exc)
        return self.refresh(ref)

    def poll_terminal_progress(self) -> None:
        """Advance Work/CUDA fences only on the engine-owner thread."""

        with self._watchdog_lock:
            if self.termination_requested:
                return
            keys = tuple(self._launched_at)
        for key in keys:
            self._advance_terminal_fences(key)
            snapshot = self._safe_status(key)
            # Deadline removal is the atomic terminal claim; borrowed-buffer
            # detachment follows outside the watchdog lock.
            self._finalize_terminal_snapshot(key, snapshot)

    def _publish_missing_watchdog_deadlines(self) -> None:
        """Keep direct unit fixtures using ``_launched_at`` compatible."""

        with self._watchdog_lock:
            for key, started_at in self._launched_at.items():
                if key in self._watchdog_deadlines:
                    continue
                ref = self._operation_refs.get(key)
                metadata = self._transfer_metadata.get(key)
                if ref is None or metadata is None:
                    continue
                self._watchdog_deadlines[key] = _WatchdogDeadline(
                    key, ref, str(metadata["pair_id"]), started_at
                )

    def poll_watchdog_deadline(self) -> bool:
        """Latch timeout evidence without calling owner, Work, or CUDA APIs."""

        now = self._clock()
        with self._watchdog_lock:
            if self.termination_requested:
                return True
            deadline = self._expired_deadline_locked(now)
            if deadline is None:
                return False
            self._latch_watchdog_timeout_locked(deadline)
            return True

    def poll_watchdog(self) -> bool:
        """Owner-side progress plus the independently callable deadline poll."""

        with self._watchdog_lock:
            if self.termination_requested:
                return True
        self.poll_terminal_progress()
        self._publish_missing_watchdog_deadlines()
        return self.poll_watchdog_deadline()

    def refresh(self, ref: EndpointOperationRef) -> OperationSnapshot:
        key = endpoint_ref_key(ref)
        self._advance_terminal_fences(key)
        self.poll_watchdog()
        snapshot = self._published_status(key)
        state = (
            OperationState.COMPLETED
            if snapshot.status.value == "COMPLETED"
            else OperationState.FENCED
            if snapshot.status.value == "FENCED"
            else OperationState.UNKNOWN
        )
        metadata = self._transfer_metadata.get(key)
        if metadata is None:
            if state != OperationState.UNKNOWN:
                raise ValueError("mapped transfer terminal metadata is missing")
            result = {}
        else:
            result = {
                **metadata,
                "work_terminal": bool(snapshot.work_terminal),
                "cuda_terminal": bool(snapshot.cuda_visibility_terminal),
            }
        return OperationSnapshot(
            ref, state, resources_held=snapshot.resources_held,
            held_resource_kinds=snapshot.held_resource_kinds,
            reason=self._launch_error(key), result=result,
        )

    def abort(self, ref: EndpointOperationRef):
        key = endpoint_ref_key(ref)
        if key in self.registry._operations:
            self.registry._operations[key].accepting_new_work = False
            snapshot = self._published_status(key)
            pending = self._pending_receives.get(key)
            if pending is not None and key not in self._receive_copy_enqueued:
                self._pending_receives.pop(key, None)
                self._discarded_receives[key] = (*pending, False)
            return snapshot
        return None

    def operation_completed(
        self, operation_id: str, expected_ref: EndpointOperationRef | None = None
    ) -> bool:
        key = self._operation_keys.get(operation_id)
        if key is None:
            return False
        if expected_ref is not None and key != endpoint_ref_key(expected_ref):
            return False
        snapshot = self._published_status(key)
        with self._watchdog_lock:
            deadline_active = key in self._watchdog_deadlines
        return not deadline_active and snapshot.status.value == "COMPLETED"

    def release(self, operation_id: str, resource_kinds: tuple[str, ...]):
        self.validate_release(operation_id, resource_kinds)
        key = self._operation_keys[operation_id]
        with self._watchdog_lock:
            if key in self._watchdog_deadlines:
                raise ValueError("transfer operation is not terminal")
            operation = self.registry._operations[key]
            requested = set(resource_kinds)
            operation.held_resource_kinds = tuple(
                kind for kind in operation.held_resource_kinds
                if kind not in requested
            )
            quantities = self._resource_quantities.get(key, {})
            released = {
                kind: int(quantities.pop(kind, 0)) for kind in resource_kinds
            }
            if not quantities:
                self._resource_quantities.pop(key, None)
            return released

    def resource_quantities(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for key, quantities in self._resource_quantities.items():
            held = set(self.registry._operations[key].held_resource_kinds)
            for kind, quantity in quantities.items():
                if kind in held:
                    totals[kind] = totals.get(kind, 0) + quantity
        return totals

    def prune(self, refs: set[EndpointOperationRef]) -> None:
        valid_keys = {endpoint_ref_key(ref) for ref in refs}
        retained_keys: set[str] = set()
        for key in set(self.registry._operations) - valid_keys:
            # Pruning an outer ref is an abort, not proof that an endpoint
            # writer stopped.  Fence new work first, then retain all launch/
            # receive state until Work and CUDA visibility are terminal.
            self.registry._operations[key].accepting_new_work = False
            pending = self._pending_receives.get(key)
            if pending is not None and key not in self._receive_copy_enqueued:
                self._pending_receives.pop(key, None)
                self._discarded_receives[key] = (*pending, False)
            self._advance_terminal_fences(key)
            snapshot = self._safe_status(key)
            self._finalize_terminal_snapshot(key, snapshot)
            with self._watchdog_lock:
                deadline_active = key in self._watchdog_deadlines
            if not (
                snapshot.work_terminal
                and snapshot.cuda_visibility_terminal
                and not deadline_active
                and key not in self._pending_receive_launches
                and key not in self._pending_receives
                and key not in self._receive_copy_enqueued
                and key not in self._discarded_receives
                and key not in self._failed_receive_buffers
                and key not in self._pending_sends
            ):
                retained_keys.add(key)
                continue
            self.registry._operations.pop(key, None)
            self._pending_receive_launches.pop(key, None)
            self._pending_receives.pop(key, None)
            self._receive_copy_enqueued.discard(key)
            self._pending_sends.pop(key, None)
            self._discarded_receives.pop(key, None)
            self._failed_receive_buffers.pop(key, None)
            self._receive_keys.discard(key)
            self._operation_refs.pop(key, None)
            self._resource_quantities.pop(key, None)
            self._transfer_metadata.pop(key, None)
            self._clear_watchdog_deadline(key)
            with self._watchdog_lock:
                self._launch_errors.pop(key, None)
            self._terminal_progress_failed.discard(key)
        kept_keys = valid_keys | retained_keys
        self._operation_keys = {
            operation_id: key
            for operation_id, key in self._operation_keys.items()
            if key in kept_keys
        }

    def validate_release(
        self, operation_id: str, resource_kinds: tuple[str, ...]
    ) -> None:
        key = self._operation_keys.get(operation_id)
        if key is None:
            raise ValueError("transfer operation is not terminal")
        snapshot = self._published_status(key)
        with self._watchdog_lock:
            deadline_active = key in self._watchdog_deadlines
        if deadline_active:
            raise ValueError("transfer operation is not terminal")
        if snapshot.status.value not in {"COMPLETED", "FENCED"}:
            raise ValueError("transfer operation is not terminal")
        allowed = {"SOURCE_RETAIN", "TRANSFER_BYTES"}
        if not set(resource_kinds) <= allowed:
            raise ValueError("mapped transfer does not own requested resources")
        held = set(self.registry._operations[key].held_resource_kinds)
        if not set(resource_kinds) <= held:
            raise ValueError("mapped transfer resources already released")


class EngineOwnerCommandQueue:
    def __init__(
        self,
        handler: Callable[[str, EndpointOperationRef, dict], OperationSnapshot],
        idle: Callable[[], None] | None = None,
    ):
        self._handler = handler
        self._idle = idle
        self._queue: Queue[tuple[str, EndpointOperationRef, dict, Future] | None] = Queue()
        self._thread = Thread(target=self._run, name="prism-engine-owner", daemon=True)
        self._thread.start()

    def submit(
        self, operation: str, ref: EndpointOperationRef, payload: dict[str, object]
    ) -> OperationSnapshot:
        future: Future = Future()
        self._queue.put((operation, ref, dict(payload), future))
        return future.result()

    def submit_future(
        self, operation: str, ref: EndpointOperationRef | None,
        payload: dict[str, object],
    ) -> Future:
        future: Future = Future()
        self._queue.put((operation, ref, dict(payload), future))
        return future

    async def submit_async(
        self, operation: str, ref: EndpointOperationRef,
        payload: dict[str, object],
    ):
        return await asyncio.wrap_future(
            self.submit_future(operation, ref, payload)
        )

    def submit_local(self, operation: str, payload: dict[str, object]):
        future: Future = Future()
        self._queue.put((operation, None, dict(payload), future))
        return future.result()

    async def submit_local_async(self, operation: str, payload: dict[str, object]):
        return await asyncio.wrap_future(
            self.submit_future(operation, None, payload)
        )

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.002 if self._idle else None)
            except Empty:
                try:
                    self._idle()
                except Exception:
                    # The driver already fenced affected sequences. Keep the owner
                    # thread alive so abort and finalize remain available.
                    pass
                continue
            if item is None:
                return
            operation, ref, payload, future = item
            try:
                future.set_result(self._handler(operation, ref, payload))
            except BaseException as exc:
                future.set_exception(exc)


class EngineControlRouter:
    """Bind generic endpoint operations to the live LLMEngine services."""

    def __init__(
        self,
        engine,
        *,
        prepare_receive: Callable[[EndpointOperationRef, dict], OperationSnapshot] | None = None,
        start_transfer: Callable[[EndpointOperationRef, dict], OperationSnapshot] | None = None,
        transfer_terminal: Callable[
            [str, EndpointOperationRef | None], bool
        ] | None = None,
        transfer_release: Callable[[str, tuple[str, ...]], object] | None = None,
        request_committed: Callable[
            [str, str, object, dict[str, object]], None
        ] | None = None,
    ) -> None:
        self.engine = engine
        self.prefix_cache = engine.prefix_cache
        self.prepare_receive = prepare_receive
        self.start_transfer = start_transfer
        self.transfer_terminal = transfer_terminal
        self.transfer_release = transfer_release
        self.request_committed = request_committed

    @staticmethod
    def _sampling(payload: dict[str, object]) -> SamplingParams:
        value = payload.get("sampling_params", {})
        return value if isinstance(value, SamplingParams) else SamplingParams(**value)

    def __call__(
        self, operation: str, ref: EndpointOperationRef, payload: dict[str, object]
    ) -> OperationSnapshot:
        if operation == "prefix.resolve":
            expected = [
                (int(item["chain_hash"]), list(item["token_ids"]))
                for item in payload["expected_blocks"]
            ]
            blocks = self.prefix_cache.resolve_prefix(
                ref.operation_id,
                expected,
                namespace=str(payload["namespace"]),
                kv_compatibility_id=str(payload["kv_compatibility_id"]),
                request_context_digest=str(payload["request_context_digest"]),
            )
            return OperationSnapshot(
                ref, OperationState.COMPLETED,
                resources_held=blocks is not None,
                held_resource_kinds=("SOURCE_PIN",) if blocks is not None else (),
                result={"src_block_ids": list(blocks)} if blocks is not None else {"miss": True},
            )
        if operation in {"prefix.prepare", "request.prepare"}:
            prepared = self.prefix_cache.prepare(
                ref.operation_id,
                str(payload["req_id"]),
                mode=str(payload.get("mode", "remote_transfer")),
                block_count=int(payload["block_count"]),
                token_ids=list(payload["token_ids"]),
                sampling_params=self._sampling(payload),
            )
            if prepared is None:
                return OperationSnapshot(ref, OperationState.FENCED, reason="allocation failed")
            local_reuse = prepared.mode == "local_reuse"
            return OperationSnapshot(
                ref,
                OperationState.PREPARED,
                resources_held=not local_reuse,
                held_resource_kinds=("TARGET_PENDING",) if not local_reuse else (),
                result={"mode": prepared.mode, "dst_block_ids": list(prepared.dst_block_ids)},
            )
        if operation in {"prefix.commit", "request.commit"}:
            transfer_proven = False
            transfer_ref_value = payload.get("transfer_endpoint_ref")
            if transfer_ref_value is not None:
                if not isinstance(transfer_ref_value, dict):
                    raise ValueError("transfer_endpoint_ref must be an object")
                transfer_ref = EndpointOperationRef(**transfer_ref_value)
                transfer_operation_id = str(payload["transfer_operation_id"])
                if transfer_ref.operation_id != transfer_operation_id:
                    raise ValueError("transfer proof operation identity mismatch")
                transfer_proven = bool(
                    self.transfer_terminal is not None
                    and self.transfer_terminal(transfer_operation_id, transfer_ref)
                )
                if not transfer_proven:
                    raise ValueError("target transfer endpoint is not complete")
            elif operation == "request.commit" or payload.get("mode") == "remote_transfer":
                raise ValueError("remote commit requires exact target transfer proof")
            sequence = self.prefix_cache.commit(
                ref.operation_id,
                namespace=str(payload.get("namespace", "")),
                kv_compatibility_id=str(payload.get("kv_compatibility_id", "")),
                request_context_digest=str(payload.get("request_context_digest", "")),
                cached_prefix_tokens=int(payload.get("cached_prefix_tokens", 0)),
                transfer_proven=transfer_proven,
            )
            sequence.defer_deallocation = True
            if operation == "request.commit":
                first_token = payload.get("first_token")
                if isinstance(first_token, bool) or not isinstance(first_token, int):
                    raise ValueError("request commit requires sampled first_token")
                if sequence in self.engine.scheduler.waiting:
                    self.engine.scheduler.waiting.remove(sequence)
                sequence.num_cached_tokens = sequence.num_prompt_tokens
                sequence.append_token(first_token)
                sequence.is_prefill = False
                from prism_infer.engine.sequence import SequenceStatus
                first_token_terminal = (
                    (not sequence.ignore_eos and first_token == self.engine.scheduler.eos)
                    or sequence.num_completion_tokens >= sequence.max_tokens
                )
                sequence.status = (
                    SequenceStatus.FINISHED
                    if first_token_terminal else SequenceStatus.RUNNING
                )
                if not first_token_terminal \
                        and sequence not in self.engine.scheduler.running:
                    self.engine.scheduler.running.append(sequence)
            if operation == "request.commit" and self.request_committed is not None:
                self.request_committed(
                    str(payload["req_id"]), ref.operation_id, sequence, payload
                )
            return OperationSnapshot(
                ref, OperationState.COMPLETED, resources_held=True,
                held_resource_kinds=("TARGET_SEQUENCE",),
                result={"seq_id": sequence.seq_id, "dst_block_ids": list(sequence.block_table)},
            )
        if operation == "transfer.prepare_receive" and self.prepare_receive is not None:
            return self.prepare_receive(ref, payload)
        if operation == "transfer.start" and self.start_transfer is not None:
            return self.start_transfer(ref, payload)
        raise ValueError(f"unsupported production control operation: {operation}")

    def release(self, operation_id: str, resource_kinds: tuple[str, ...]):
        if set(resource_kinds) <= {"SOURCE_RETAIN", "TRANSFER_BYTES"}:
            if self.transfer_release is None:
                raise ValueError("transfer release handler is not installed")
            return self.transfer_release(operation_id, resource_kinds)
        return self.prefix_cache.finalize_release(operation_id, resource_kinds)


class PDExecutionDriver:
    """Drive one prefill/suffix command on the engine owner thread."""

    def __init__(
        self,
        engine,
        role: str = "prefill",
        *,
        model_profile: dict[str, object] | None = None,
    ) -> None:
        self.engine = engine
        self.role = role
        self.model_profile = (
            dict(model_profile) if model_profile is not None else None
        )
        self._requests: dict[str, object] = {}
        self._operations: dict[str, object] = {}
        self._event_metadata: dict[str, dict[str, object]] = {}
        self._output_cursors: dict[str, int] = {}
        self._terminal_emitted: set[str] = set()
        self.events: Queue[tuple[str, dict[str, object]]] = Queue()

    def __call__(self, kind: str, ref: EndpointOperationRef, payload: dict[str, object]):
        if kind == "prefill":
            self.engine.add_request(
                list(payload["token_ids"]),
                SamplingParams(**dict(payload.get("sampling_params", {}))),
            )
            sequence = self.engine.scheduler.waiting[-1]
            self._requests[str(payload["req_id"])] = sequence
            self._operations[ref.operation_id] = sequence
        elif kind == "suffix":
            operation = self.engine.prefix_cache._operations[ref.operation_id]
            sequence = operation.sequence
            if sequence is None:
                raise ValueError("suffix dispatch requires committed sequence")
            self._requests[str(payload["req_id"])] = sequence
            self._operations[ref.operation_id] = sequence
            if sequence not in self.engine.scheduler.waiting \
                    and sequence not in self.engine.scheduler.running:
                self.engine.scheduler.add(sequence)
            self._event_metadata[str(payload["req_id"])] = dict(payload)
        else:
            raise ValueError(f"unknown PD command kind: {kind}")

        if kind == "prefill":
            # Prefill produces the handoff token and then finishes. Retain its KV
            # until transfer and generic finalize complete.
            sequence.defer_deallocation = True
        completion_count_before = len(sequence.completion_token_ids)
        while sequence.num_cached_tokens < sequence.num_prompt_tokens:
            cached_before = sequence.num_cached_tokens
            try:
                self.engine.step()
            except Exception:
                # Fence this exact sequence but retain its blocks for finalize.
                self.abort_request(ref.operation_id)
                raise
            if sequence.num_cached_tokens <= cached_before:
                status = getattr(sequence.status, "name", str(sequence.status))
                # Prevent rescheduling without releasing ownership on this fail-fast path.
                self.abort_request(ref.operation_id)
                raise RuntimeError(
                    "prefill engine made no progress: "
                    f"cached={sequence.num_cached_tokens}, "
                    f"prompt={sequence.num_prompt_tokens}, status={status}"
                )
        kv_cache = getattr(self.engine.model_runner, "kv_cache", None)
        block_bytes = 0
        if kv_cache is not None and kv_cache.shape[2] > 0:
            block_bytes = kv_cache[:, :, 0].numel() * kv_cache.element_size()
        first_token = (
            sequence.completion_token_ids[0]
            if sequence.completion_token_ids else None
        )
        if kind == "suffix" \
                and len(sequence.completion_token_ids) > completion_count_before:
            self._publish_current(str(payload["req_id"]), sequence)
        if kind == "prefill":
            from prism_infer.engine.sequence import SequenceStatus
            try:
                self.engine.scheduler.running.remove(sequence)
            except ValueError:
                pass
            sequence.status = SequenceStatus.KV_TRANSFERRING
        owns_source_blocks = kind == "prefill"
        return OperationSnapshot(
            ref, OperationState.COMPLETED,
            resources_held=owns_source_blocks,
            held_resource_kinds=("SOURCE_BLOCKS",) if owns_source_blocks else (),
            result={
                "block_table": list(sequence.block_table),
                "kv_size_bytes": block_bytes * len(sequence.block_table),
                "first_token": first_token,
                "output_seq_no": len(sequence.completion_token_ids),
                "token_ids": list(sequence.completion_token_ids),
            },
        )

    def request_committed(
        self,
        req_id: str,
        operation_id: str,
        sequence,
        payload: dict[str, object],
    ) -> None:
        self._requests[req_id] = sequence
        # Side state follows the commit ref; the completed transfer id may be pruned.
        self._operations[operation_id] = sequence
        self._event_metadata[req_id] = dict(payload)
        self._publish_current(req_id, sequence)

    def _publish_current(self, req_id: str, sequence) -> None:
        metadata = self._event_metadata.get(req_id, {})
        tokens = list(sequence.completion_token_ids)
        cursor = self._output_cursors.get(req_id, 0)
        base = {
            "req_id": req_id,
            "instance_epoch": self.engine.prefix_cache.instance_epoch,
            "operation_id": str(metadata.get("operation_id", req_id)),
            "output_seq_no": len(tokens),
            "token_ids": tokens,
        }
        if len(tokens) > cursor:
            if cursor == 0 and metadata.get("first_token_subject"):
                self.events.put((str(metadata["first_token_subject"]), base))
            if metadata.get("decode_progress_subject"):
                self.events.put((str(metadata["decode_progress_subject"]), base))
            self._output_cursors[req_id] = len(tokens)
        if sequence.is_finished and req_id not in self._terminal_emitted:
            subject = metadata.get("decode_done_subject")
            if subject:
                self.events.put((str(subject), {**base, "terminal": True}))
            self._terminal_emitted.add(req_id)

    def abort_request(self, operation_id: str) -> bool:
        sequence = self._operations.get(operation_id)
        if sequence is None:
            return False
        from prism_infer.engine.sequence import SequenceStatus
        for collection in (
            self.engine.scheduler.waiting, self.engine.scheduler.running
        ):
            try:
                collection.remove(sequence)
            except ValueError:
                pass
        sequence.status = SequenceStatus.ABORTED
        for req_id, current in list(self._requests.items()):
            if current is not sequence:
                continue
            self._requests.pop(req_id, None)
            self._event_metadata.pop(req_id, None)
            self._output_cursors.pop(req_id, None)
            self._terminal_emitted.add(req_id)
        return True

    def validate_source_blocks(self, operation_id: str) -> None:
        sequence = self._operations.get(operation_id) or self._requests.get(operation_id)
        # A prefill command declares SOURCE_BLOCKS before entering the engine
        # owner thread.  If execution fails before Sequence registration or KV
        # allocation, the exact FENCED ref still needs a zero-count generic
        # finalize.  A present sequence must nevertheless expose the canonical
        # block table so a partially allocated failure cannot be hidden.
        if sequence is not None and not hasattr(sequence, "block_table"):
            raise ValueError("source request block ownership is unavailable")

    def release_source_blocks(self, operation_id: str) -> dict[str, int]:
        self.validate_source_blocks(operation_id)
        sequence = self._operations.get(operation_id) or self._requests.get(
            operation_id
        )
        if sequence is None:
            return {"SOURCE_BLOCKS": 0}
        count = len(sequence.block_table)
        if count:
            self.engine.scheduler.block_manager.deallocate(sequence)
        for collection in (
            self.engine.scheduler.waiting, self.engine.scheduler.running
        ):
            try:
                collection.remove(sequence)
            except ValueError:
                pass
        from prism_infer.engine.sequence import SequenceStatus

        sequence.status = SequenceStatus.ABORTED
        self._operations.pop(operation_id, None)
        for req_id, current in list(self._requests.items()):
            if current is not sequence:
                continue
            self._requests.pop(req_id, None)
            self._event_metadata.pop(req_id, None)
            self._output_cursors.pop(req_id, None)
            self._terminal_emitted.add(req_id)
        return {"SOURCE_BLOCKS": count}

    def idle_step(self) -> None:
        if self.role != "decode":
            return
        if not self.engine.scheduler.is_finished():
            try:
                self.engine.step()
            except Exception:
                from prism_infer.engine.sequence import SequenceStatus

                # A failed batch cannot be attributed to one sequence. Fence every
                # active decode writer and retain block tables for finalize.
                terminal = {SequenceStatus.FINISHED, SequenceStatus.ABORTED}
                for operation_id, sequence in tuple(self._operations.items()):
                    if sequence.status not in terminal:
                        self.abort_request(operation_id)
                raise
        for req_id, metadata in list(self._event_metadata.items()):
            sequence = self._requests[req_id]
            self._publish_current(req_id, sequence)

    def output(self, req_id: str, after_seq: int) -> dict[str, object]:
        sequence = self._requests.get(req_id)
        if sequence is None:
            raise RequestOutputNotFound(
                f"request output is not available: {req_id}"
            )
        tokens = list(sequence.completion_token_ids)
        metadata = self._event_metadata.get(req_id, {})
        return {
            "req_id": req_id,
            "instance_epoch": self.engine.prefix_cache.instance_epoch,
            "operation_id": str(metadata.get("operation_id", req_id)),
            "output_seq_no": len(tokens),
            "token_ids": tokens,
            "terminal": bool(sequence.is_finished),
        }

    def resource_details(
        self,
        endpoint_snapshots: list[OperationSnapshot] | None = None,
    ) -> dict[str, object]:
        block_manager = self.engine.scheduler.block_manager
        active = [
            req_id for req_id, sequence in self._requests.items()
            if not sequence.is_finished
        ]
        # Report zero for every ledger kind so absence is never interpreted as zero.
        quantities = {kind: 0 for kind in CONTROLLED_RESOURCE_KINDS}
        prefix = self.engine.prefix_cache
        unknown_operation_ids = {
            snapshot.endpoint_ref.operation_id
            for snapshot in endpoint_snapshots or ()
            if snapshot.resources_held and snapshot.state == OperationState.UNKNOWN
        }
        with block_manager._prefix_state_lock:
            transfer_pins = {
                operation_id: tuple(block_ids)
                for operation_id, block_ids in block_manager._transfer_pins.items()
            }
            quantities["SOURCE_PIN"] = sum(
                len(block_ids)
                for block_ids in transfer_pins.values()
            )
        target_pending = 0
        target_sequence = 0
        prefix_sequences = set()
        pending_block_ids: set[int] = set()
        quarantined_block_ids: set[int] = set()
        for operation in prefix._operations.values():
            if not operation.resources_held:
                continue
            owned_blocks = set(
                operation.sequence.block_table
                if operation.sequence is not None
                else operation.dst_block_ids
            )
            if operation.operation_id in unknown_operation_ids:
                quarantined_block_ids.update(owned_blocks)
            if operation.sequence is None:
                target_pending += len(operation.dst_block_ids)
                if operation.operation_id not in unknown_operation_ids:
                    pending_block_ids.update(operation.dst_block_ids)
            else:
                prefix_sequences.add(id(operation.sequence))
                target_sequence += len(operation.sequence.block_table)
        quantities["TARGET_PENDING"] = target_pending
        quantities["TARGET_SEQUENCE"] = target_sequence
        source_sequences = {
            id(sequence): sequence for sequence in self._operations.values()
            if id(sequence) not in prefix_sequences
        }
        quantities["SOURCE_BLOCKS"] = sum(
            len(sequence.block_table) for sequence in source_sequences.values()
        )
        for operation_id, sequence in self._operations.items():
            if operation_id in unknown_operation_ids:
                quarantined_block_ids.update(sequence.block_table)
        pinned_block_ids: set[int] = set()
        for operation_id, block_ids in transfer_pins.items():
            if operation_id in unknown_operation_ids:
                quarantined_block_ids.update(block_ids)
            else:
                pinned_block_ids.update(block_ids)
        block_snapshot = build_block_resource_snapshot(
            block_manager,
            pending_block_ids=pending_block_ids,
            quarantined_block_ids=quarantined_block_ids,
            pinned_block_ids=pinned_block_ids,
        )
        buckets = block_snapshot["block_buckets"]
        assert isinstance(buckets, dict)
        unavailable_blocks = sum(
            int(buckets[name]) for name in ("pending", "sequence", "quarantined")
        )
        reported_resource_ids = (
            set(block_manager.used_block_ids)
            | pinned_block_ids
            | quarantined_block_ids
        )
        value = {
            "max_slots": int(self.engine.scheduler.max_num_seqs),
            "active_request_ids": sorted(active),
            "active_transfer_operation_ids": [],
            "pending_dispatch_command_ids": [],
            "resource_ids": [
                f"block:{block_id}" for block_id in sorted(reported_resource_ids)
            ],
            "kv_usage": (
                unavailable_blocks / block_snapshot["num_gpu_blocks"]
                if block_snapshot["num_gpu_blocks"] else 1.0
            ),
            "resource_quantities": quantities,
            **block_snapshot,
        }
        if self.model_profile is not None:
            value["model_profile"] = dict(self.model_profile)
        return value

    def prune(self, operation_ids: set[str]) -> None:
        self._operations = {
            operation_id: sequence
            for operation_id, sequence in self._operations.items()
            if operation_id in operation_ids
        }
        live_sequences = {id(sequence) for sequence in self._operations.values()}
        for req_id, sequence in list(self._requests.items()):
            if id(sequence) in live_sequences:
                continue
            self._requests.pop(req_id, None)
            self._event_metadata.pop(req_id, None)
            self._output_cursors.pop(req_id, None)
            self._terminal_emitted.discard(req_id)

    def prefix_directory(self, action: str, value: dict[str, object]):
        from dataclasses import asdict

        manager = self.engine.scheduler.block_manager
        if action == "register":
            report = manager.full_report_and_register(
                str(value["consumer_id"]), str(value["generation"])
            )
            return {
                "instance_id": report.instance_id,
                "instance_epoch": report.instance_epoch,
                "snapshot_seq_no": report.snapshot_seq_no,
                "locations": [asdict(event) for event in report.locations],
            }
        if action == "peek":
            events = manager.peek_events(
                str(value["consumer_id"]), str(value["generation"]),
                int(value["after_seq"]), int(value["limit"]),
            )
            return [asdict(event) for event in events]
        if action == "ack":
            manager.ack_events(
                str(value["consumer_id"]), str(value["generation"]),
                int(value["up_to_seq"]),
            )
            return None
        raise ValueError(f"unknown prefix directory action: {action}")
