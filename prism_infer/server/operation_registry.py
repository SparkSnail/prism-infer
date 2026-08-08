"""Bounded, epoch-fenced endpoint operation registry.

EndpointOperationRef is the idempotency key; operation_id is correlation only.
Abort fences writers, while generic finalize owns resource release.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from collections.abc import Mapping
from typing import Callable


def canonical_payload_digest(payload: dict[str, object]) -> str:
    """Digest the immutable mutation payload using the wire canonical form."""
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RegistryError(RuntimeError):
    """Registry error that maps to an HTTP response status."""


class FencedWorkerEpoch(RegistryError):
    pass


class UnknownOwner(RegistryError):
    pass


class RetiredOwner(RegistryError):
    pass


class OperationConflict(RegistryError):
    pass


class StaleOperation(RegistryError):
    pass


class RegistryCapacity(RegistryError):
    pass


class PreconditionFailed(RegistryError):
    pass


class RequestOutputNotFound(RegistryError):
    """The current worker incarnation does not own the requested output."""

    pass


class OperationState(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FENCED = "FENCED"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        return self in {OperationState.COMPLETED, OperationState.FENCED}


@dataclass(frozen=True)
class EndpointOperationRef:
    topology_generation: str
    owner_generation: str
    operation_seq: int
    target_instance: str
    target_worker_epoch: str
    operation_id: str
    payload_digest: str

    def __post_init__(self):
        assert self.operation_seq > 0, "operation_seq must start at one"
        assert self.payload_digest, "endpoint operation requires payload digest"


def _same_endpoint_ref(left, right) -> bool:
    """Compare wire identity without requiring identical dataclass types."""
    fields = (
        "topology_generation",
        "owner_generation",
        "operation_seq",
        "target_instance",
        "target_worker_epoch",
        "operation_id",
        "payload_digest",
    )
    try:
        return tuple(getattr(left, field) for field in fields) == tuple(
            getattr(right, field) for field in fields
        )
    except AttributeError:
        return False


@dataclass(frozen=True)
class OperationSnapshot:
    endpoint_ref: EndpointOperationRef
    state: OperationState
    resources_held: bool = False
    held_resource_kinds: tuple[str, ...] = ()
    reason: str = ""
    result: dict[str, object] | None = None
    delivery_count: int = 0
    execution_count: int = 0

    @classmethod
    def running(
        cls,
        endpoint_ref: EndpointOperationRef,
        *,
        held_resource_kinds: tuple[str, ...] = (),
    ):
        return cls(
            endpoint_ref=endpoint_ref,
            state=OperationState.RUNNING,
            resources_held=bool(held_resource_kinds),
            held_resource_kinds=tuple(sorted(set(held_resource_kinds))),
        )


@dataclass(frozen=True)
class FinalizeReleaseRequest:
    cleanup_id: str
    operation_id: str
    lease_id: str
    endpoint_refs: tuple[EndpointOperationRef, ...]
    resource_kinds: tuple[str, ...]
    release_basis: str
    payload_digest: str

    @classmethod
    def build(
        cls,
        *,
        cleanup_id: str,
        operation_id: str,
        lease_id: str,
        endpoint_refs: tuple[EndpointOperationRef, ...],
        resource_kinds: tuple[str, ...],
    ):
        payload = {
            "cleanup_id": cleanup_id,
            "operation_id": operation_id,
            "lease_id": lease_id,
            "endpoint_refs": [ref.__dict__ for ref in endpoint_refs],
            "resource_kinds": sorted(set(resource_kinds)),
            "release_basis": "ENDPOINT_TERMINAL",
        }
        digest = "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            cleanup_id=cleanup_id,
            operation_id=operation_id,
            lease_id=lease_id,
            endpoint_refs=endpoint_refs,
            resource_kinds=tuple(payload["resource_kinds"]),
            release_basis="ENDPOINT_TERMINAL",
            payload_digest=digest,
        )

    def canonical_digest(self) -> str:
        return FinalizeReleaseRequest.build(
            cleanup_id=self.cleanup_id,
            operation_id=self.operation_id,
            lease_id=self.lease_id,
            endpoint_refs=self.endpoint_refs,
            resource_kinds=self.resource_kinds,
        ).payload_digest


@dataclass(frozen=True)
class ReleaseSnapshot:
    cleanup_id: str
    operation_id: str
    lease_id: str
    endpoint_epoch: str
    released_resource_kinds: tuple[str, ...]
    released_counts: tuple[tuple[str, int], ...]
    resources_held_after: bool
    payload_digest: str


@dataclass
class _FinalizeRecord:
    payload_digest: str
    snapshot: ReleaseSnapshot


class OperationRegistry:
    """Accept one active owner generation per worker.

    Active entries do not expire by time. Sequence numbers below the reorder
    window remain stale, and only resource-free terminal snapshots are evictable.
    """

    def __init__(
        self,
        *,
        instance_id: str,
        instance_epoch: str,
        topology_generation: str,
        active_operation_cap: int = 512,
        operation_reorder_window: int = 4096,
        terminal_snapshot_cap: int = 4096,
    ):
        assert active_operation_cap > 0
        assert operation_reorder_window > 0
        assert terminal_snapshot_cap > 0
        self.instance_id = instance_id
        self.instance_epoch = instance_epoch
        self.topology_generation = topology_generation
        self.active_operation_cap = active_operation_cap
        self.operation_reorder_window = operation_reorder_window
        self.terminal_snapshot_cap = terminal_snapshot_cap
        self.active_owner: str | None = None
        self._retired_owners: set[str] = set()
        self._high_watermark = 0
        self._seen: OrderedDict[int, str] = OrderedDict()
        self._active: dict[int, OperationSnapshot] = {}
        self._terminal: OrderedDict[int, OperationSnapshot] = OrderedDict()
        self._finalize_by_cleanup: OrderedDict[str, _FinalizeRecord] = OrderedDict()

    @property
    def floor(self) -> int:
        return max(1, self._high_watermark - self.operation_reorder_window + 1)

    def activate_owner(self, owner_generation: str) -> str:
        if owner_generation in self._retired_owners:
            raise RetiredOwner(owner_generation)
        if self.active_owner is None:
            self.active_owner = owner_generation
        elif self.active_owner != owner_generation:
            raise OperationConflict("another owner generation is active")
        return self.active_owner

    def retire_owner(self, owner_generation: str) -> str:
        """Retire the active owner only after every operation is terminal/free.

        Retirement is the fencing boundary used by a replacement Gateway.  It
        must never turn an in-flight operation or a held-terminal snapshot into
        an implicit release proof.
        """
        if owner_generation in self._retired_owners:
            return owner_generation
        if self.active_owner != owner_generation:
            raise UnknownOwner(owner_generation)
        owned = self.snapshots(owner_generation)
        if any(not snapshot.state.terminal for snapshot in owned):
            raise PreconditionFailed("owner still has active operations")
        if any(snapshot.resources_held for snapshot in owned):
            raise PreconditionFailed("owner still holds resources")
        self._retired_owners.add(owner_generation)
        self.active_owner = None
        # Sequence numbers and dedup/finalize records are scoped to one owner
        # generation.  Retirement is allowed only after every entry is
        # terminal/resource-free, so resetting these bounded maps cannot hide
        # live ownership and lets the replacement owner legitimately start at
        # endpoint sequence 1.
        self._high_watermark = 0
        self._seen.clear()
        self._active.clear()
        self._terminal.clear()
        self._finalize_by_cleanup.clear()
        return owner_generation

    def owner_status(self) -> dict[str, object]:
        return {
            "active_owner": self.active_owner,
            "retired_owners": tuple(sorted(self._retired_owners)),
        }

    def _validate_ref(self, ref: EndpointOperationRef) -> None:
        if ref.target_worker_epoch != self.instance_epoch:
            raise FencedWorkerEpoch(ref.target_worker_epoch)
        if ref.topology_generation != self.topology_generation:
            raise FencedWorkerEpoch(ref.topology_generation)
        if ref.target_instance != self.instance_id:
            raise OperationConflict("endpoint ref targets another instance")
        if ref.owner_generation in self._retired_owners:
            raise RetiredOwner(ref.owner_generation)
        if self.active_owner is None or ref.owner_generation != self.active_owner:
            raise UnknownOwner(ref.owner_generation)

    def _lookup(self, ref: EndpointOperationRef) -> OperationSnapshot | None:
        snapshot = self._active.get(ref.operation_seq)
        if snapshot is None:
            snapshot = self._terminal.get(ref.operation_seq)
        if snapshot is None:
            return None
        if snapshot.endpoint_ref.payload_digest != ref.payload_digest:
            raise OperationConflict("endpoint sequence reused with different digest")
        if not _same_endpoint_ref(snapshot.endpoint_ref, ref):
            raise OperationConflict("endpoint sequence reused with different ref")
        return snapshot

    def _record_seen(self, ref: EndpointOperationRef) -> None:
        if ref.operation_seq > self._high_watermark:
            self._high_watermark = ref.operation_seq
        while self._seen and next(iter(self._seen)) < self.floor:
            self._seen.popitem(last=False)
        prior_digest = self._seen.get(ref.operation_seq)
        if prior_digest is not None and prior_digest != ref.payload_digest:
            raise OperationConflict("seen endpoint sequence changed digest")
        self._seen[ref.operation_seq] = ref.payload_digest
        self._seen = OrderedDict(sorted(self._seen.items()))

    def _check_new_ref(self, ref: EndpointOperationRef) -> None:
        """Check sequence and capacity gates without advancing the window."""
        if ref.operation_seq < self.floor:
            raise StaleOperation(ref.operation_seq)
        seen_digest = self._seen.get(ref.operation_seq)
        if seen_digest is not None:
            if seen_digest != ref.payload_digest:
                raise OperationConflict("seen endpoint sequence changed digest")
            raise StaleOperation(ref.operation_seq)
        if len(self._active) >= self.active_operation_cap:
            raise RegistryCapacity("active operation registry is full")

    def _admit_new_ref(self, ref: EndpointOperationRef) -> None:
        """Check all gates before recording a new endpoint ref."""
        self._check_new_ref(ref)
        # Capacity rejection must not advance the sequence window.
        self._record_seen(ref)

    def accept(
        self,
        ref: EndpointOperationRef,
        execute: Callable[[], OperationSnapshot],
    ) -> OperationSnapshot:
        snapshot, _ = self.accept_or_replay(ref, execute)
        return snapshot

    def accept_or_replay(
        self,
        ref: EndpointOperationRef,
        execute: Callable[[], OperationSnapshot],
    ) -> tuple[OperationSnapshot, bool]:
        """Return the snapshot and whether this caller installed it."""
        self._validate_ref(ref)
        existing = self._lookup(ref)
        if existing is not None:
            return existing, False
        self._admit_new_ref(ref)
        snapshot = execute()
        if not _same_endpoint_ref(snapshot.endpoint_ref, ref):
            raise OperationConflict("executor returned another endpoint ref")
        self._store_snapshot(snapshot)
        return snapshot, True

    def abort(self, ref: EndpointOperationRef, *, reason: str) -> OperationSnapshot:
        existing = self.classify_abort(ref)
        if existing is not None:
            if existing.state.terminal:
                return existing
            snapshot = replace(existing, state=OperationState.FENCED, reason=reason)
            self._store_snapshot(snapshot)
            return snapshot
        self._admit_new_ref(ref)
        snapshot = OperationSnapshot(
            endpoint_ref=ref,
            state=OperationState.FENCED,
            resources_held=False,
            held_resource_kinds=(),
            reason=reason,
        )
        self._store_snapshot(snapshot)
        return snapshot

    def terminalize(
        self,
        ref: EndpointOperationRef,
        state: OperationState,
        *,
        reason: str = "",
    ) -> OperationSnapshot:
        assert state.terminal, f"terminalize requires terminal state: {state}"
        self._validate_ref(ref)
        existing = self._lookup(ref)
        if existing is None:
            raise PreconditionFailed("unknown endpoint operation")
        if existing.state.terminal:
            return existing
        snapshot = replace(existing, state=state, reason=reason)
        self._store_snapshot(snapshot)
        return snapshot

    def classify_abort(
        self, ref: EndpointOperationRef
    ) -> OperationSnapshot | None:
        """Classify an abort ref without mutating physical resources.

        Return an existing snapshot for an exact replay. Return None only for an
        unseen, admissible ref whose operation_id does not collide with another ref.
        """
        self._validate_ref(ref)
        existing = self._lookup(ref)
        if existing is not None:
            return existing
        self._check_new_ref(ref)
        if any(
            snapshot.endpoint_ref.operation_id == ref.operation_id
            for snapshot in (*self._active.values(), *self._terminal.values())
        ):
            raise OperationConflict(
                "abort requires the original endpoint ref for operation id"
            )
        return None

    def store_result(
        self, ref: EndpointOperationRef, snapshot: OperationSnapshot
    ) -> OperationSnapshot:
        """Store an owner-thread result without changing endpoint identity."""
        self._validate_ref(ref)
        existing = self._lookup(ref)
        if existing is None:
            raise PreconditionFailed("unknown endpoint operation")
        if not _same_endpoint_ref(snapshot.endpoint_ref, ref):
            raise OperationConflict("engine result changed endpoint ref")
        if existing.state.terminal:
            return existing
        snapshot = replace(
            snapshot,
            delivery_count=existing.delivery_count,
            execution_count=existing.execution_count,
        )
        self._store_snapshot(snapshot)
        return snapshot

    def record_delivery(self, ref: EndpointOperationRef) -> OperationSnapshot:
        """Count an actual validated NATS delivery for this exact endpoint ref."""
        self._validate_ref(ref)
        existing = self._lookup(ref)
        if existing is None:
            raise PreconditionFailed("unknown endpoint operation")
        snapshot = replace(existing, delivery_count=existing.delivery_count + 1)
        self._store_snapshot(snapshot)
        return snapshot

    def record_execution(self, ref: EndpointOperationRef) -> OperationSnapshot:
        """Count creation of the sole executor task for this exact endpoint ref."""
        self._validate_ref(ref)
        existing = self._lookup(ref)
        if existing is None:
            raise PreconditionFailed("unknown endpoint operation")
        snapshot = replace(existing, execution_count=existing.execution_count + 1)
        self._store_snapshot(snapshot)
        return snapshot

    def migrate_operation_resources(
        self, operation_id: str, keep_ref: EndpointOperationRef
    ) -> None:
        """Move logical ownership to commit and supersede its prepared ref."""
        self._validate_ref(keep_ref)
        for snapshot in self.snapshots(keep_ref.owner_generation):
            if snapshot.endpoint_ref.operation_id != operation_id:
                continue
            if snapshot.endpoint_ref == keep_ref:
                continue
            if not snapshot.resources_held \
                    and snapshot.state != OperationState.PREPARED:
                continue
            self._store_snapshot(replace(
                snapshot, state=OperationState.COMPLETED,
                resources_held=False, held_resource_kinds=()
            ))

    def snapshot(self, ref: EndpointOperationRef) -> OperationSnapshot:
        self._validate_ref(ref)
        snapshot = self._lookup(ref)
        if snapshot is None:
            raise PreconditionFailed("unknown endpoint operation")
        return snapshot

    def snapshots(self, owner_generation: str | None = None) -> tuple[OperationSnapshot, ...]:
        values = tuple(self._active.values()) + tuple(self._terminal.values())
        if owner_generation is None:
            return values
        return tuple(
            snapshot for snapshot in values
            if snapshot.endpoint_ref.owner_generation == owner_generation
        )

    def _store_snapshot(self, snapshot: OperationSnapshot) -> None:
        seq = snapshot.endpoint_ref.operation_seq
        if snapshot.state.terminal and not snapshot.resources_held:
            self._active.pop(seq, None)
            self._terminal[seq] = snapshot
            self._terminal.move_to_end(seq)
            while len(self._terminal) > self.terminal_snapshot_cap:
                evicted_seq = min(self._terminal)
                evicted = self._terminal.pop(evicted_seq)
                stale_cleanup_ids = [
                    cleanup_id
                    for cleanup_id, record in self._finalize_by_cleanup.items()
                    if record.snapshot.operation_id
                    == evicted.endpoint_ref.operation_id
                ]
                for cleanup_id in stale_cleanup_ids:
                    self._finalize_by_cleanup.pop(cleanup_id, None)
        else:
            self._terminal.pop(seq, None)
            self._active[seq] = snapshot

    def finalize_release(
        self,
        request: FinalizeReleaseRequest,
        release: Callable[[tuple[str, ...]], Mapping[str, int]],
    ) -> ReleaseSnapshot:
        replay = self.finalize_replay(request)
        if replay is not None:
            return replay
        requested = self.prepare_finalize_release(request)
        return self.commit_finalize_release(request, release(requested))

    def finalize_replay(
        self, request: FinalizeReleaseRequest
    ) -> ReleaseSnapshot | None:
        # A response-loss retry must replay the stored release before checking
        # current epoch or ownership, which changed during the first release.
        record = self._finalize_by_cleanup.get(request.cleanup_id)
        if record is not None:
            if record.payload_digest != request.payload_digest:
                raise OperationConflict("cleanup id reused with different digest")
            return record.snapshot
        return None

    def prepare_finalize_release(
        self, request: FinalizeReleaseRequest
    ) -> tuple[str, ...]:
        if request.release_basis != "ENDPOINT_TERMINAL":
            raise PreconditionFailed("worker finalize only accepts ENDPOINT_TERMINAL")
        if request.payload_digest != request.canonical_digest():
            raise PreconditionFailed("non-canonical finalize payload digest")
        if not request.endpoint_refs:
            raise PreconditionFailed("finalize requires local endpoint refs")
        snapshots: list[OperationSnapshot] = []
        held: set[str] = set()
        for ref in request.endpoint_refs:
            self._validate_ref(ref)
            if ref.operation_id != request.operation_id:
                raise PreconditionFailed("operation id does not match endpoint ref")
            snapshot = self._lookup(ref)
            if snapshot is None or not snapshot.state.terminal:
                raise PreconditionFailed("endpoint predicate is not terminal")
            if not snapshot.resources_held:
                raise PreconditionFailed("endpoint no longer holds resources")
            held.update(snapshot.held_resource_kinds)
            snapshots.append(snapshot)
        requested = set(request.resource_kinds)
        if not requested or requested != held:
            raise PreconditionFailed("resource kinds do not match held ownership")
        return tuple(sorted(requested))

    def commit_finalize_release(
        self,
        request: FinalizeReleaseRequest,
        released_counts: Mapping[str, int],
    ) -> ReleaseSnapshot:
        replay = self.finalize_replay(request)
        if replay is not None:
            return replay
        requested = set(request.resource_kinds)
        if set(released_counts) != requested:
            raise PreconditionFailed("release counts do not match requested resources")
        normalized_counts = {}
        for kind, count in released_counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise PreconditionFailed("release counts must be non-negative integers")
            normalized_counts[str(kind)] = count
        snapshots = [self.snapshot(ref) for ref in request.endpoint_refs]
        for snapshot in snapshots:
            released = replace(
                snapshot,
                resources_held=False,
                held_resource_kinds=(),
            )
            self._store_snapshot(released)
        result = ReleaseSnapshot(
            cleanup_id=request.cleanup_id,
            operation_id=request.operation_id,
            lease_id=request.lease_id,
            endpoint_epoch=self.instance_epoch,
            released_resource_kinds=tuple(sorted(requested)),
            released_counts=tuple(sorted(normalized_counts.items())),
            resources_held_after=False,
            payload_digest=request.payload_digest,
        )
        self._finalize_by_cleanup[request.cleanup_id] = _FinalizeRecord(
            payload_digest=request.payload_digest,
            snapshot=result,
        )
        self._finalize_by_cleanup.move_to_end(request.cleanup_id)
        return result
