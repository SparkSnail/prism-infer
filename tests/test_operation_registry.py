from dataclasses import replace

import pytest

from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    FinalizeReleaseRequest,
    OperationConflict,
    OperationRegistry,
    OperationSnapshot,
    OperationState,
    PreconditionFailed,
    RegistryCapacity,
    RetiredOwner,
    StaleOperation,
)


def _ref(
    seq: int, *, digest: str | None = None, epoch: str = "pod-a:boot-a",
    owner: str = "gateway-a:boot-a",
):
    return EndpointOperationRef(
        topology_generation="world-a",
        owner_generation=owner,
        operation_seq=seq,
        target_instance="d0",
        target_worker_epoch=epoch,
        operation_id=f"op-{seq}",
        payload_digest=digest or f"sha256:{seq}",
    )


def _registry(**kwargs):
    registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-a:boot-a",
        topology_generation="world-a",
        **kwargs,
    )
    registry.activate_owner("gateway-a:boot-a")
    return registry


def test_abort_before_dispatch_installs_cancelled_snapshot():
    registry = _registry()

    snapshot = registry.abort(_ref(1), reason="client cancelled")

    assert snapshot.state == OperationState.FENCED
    assert snapshot.resources_held is False


def test_late_dispatch_after_cancel_never_executes():
    registry = _registry()
    ref = _ref(1)
    registry.abort(ref, reason="publish outcome unknown")
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return OperationSnapshot.running(ref, held_resource_kinds=("TARGET_PENDING",))

    snapshot = registry.accept(ref, execute)

    assert snapshot.state == OperationState.FENCED
    assert executed is False


@pytest.mark.parametrize(
    ("first", "late"),
    [
        (OperationState.COMPLETED, OperationState.FENCED),
        (OperationState.FENCED, OperationState.COMPLETED),
    ],
)
def test_first_persisted_terminal_snapshot_wins(first, late):
    registry = _registry()
    ref = _ref(1)
    registry.accept(ref, lambda: OperationSnapshot.running(ref))

    persisted = registry.terminalize(ref, first, reason="first terminal")
    replay = registry.terminalize(ref, late, reason="late terminal")

    assert replay == persisted
    assert replay.state == first
    assert replay.reason == "first terminal"
    assert registry.snapshot(ref) == persisted


def test_abort_carries_original_endpoint_ref():
    registry = _registry()
    original = _ref(7)
    registry.accept(original, lambda: OperationSnapshot.running(original))

    snapshot = registry.abort(original, reason="client cancelled")

    assert snapshot.endpoint_ref == original
    assert snapshot.state == OperationState.FENCED


def test_abort_classification_rejects_new_ref_reusing_operation_id():
    registry = _registry()
    original = _ref(1)
    registry.accept(original, lambda: OperationSnapshot.running(original))
    collision = replace(_ref(2), operation_id=original.operation_id)

    with pytest.raises(OperationConflict):
        registry.classify_abort(collision)

    assert tuple(registry._seen) == (1,)
    assert registry.snapshot(original).state == OperationState.RUNNING


def test_cross_endpoint_refs_use_independent_target_sequences():
    owner = "gateway-a:boot-a"
    operation_id = "shared-operation"
    source_registry = OperationRegistry(
        instance_id="p0",
        instance_epoch="pod-p0:boot-a",
        topology_generation="world-a",
    )
    source_registry.activate_owner(owner)
    target_registry = _registry()
    source_ref = EndpointOperationRef(
        topology_generation="world-a",
        owner_generation=owner,
        operation_seq=41,
        target_instance="p0",
        target_worker_epoch="pod-p0:boot-a",
        operation_id=operation_id,
        payload_digest="sha256:source",
    )
    target_ref = EndpointOperationRef(
        topology_generation="world-a",
        owner_generation=owner,
        operation_seq=7,
        target_instance="d0",
        target_worker_epoch="pod-a:boot-a",
        operation_id=operation_id,
        payload_digest="sha256:target",
    )

    source = source_registry.accept(
        source_ref, lambda: OperationSnapshot.running(source_ref)
    )
    target = target_registry.accept(
        target_ref, lambda: OperationSnapshot.running(target_ref)
    )

    assert source.endpoint_ref.operation_seq == 41
    assert target.endpoint_ref.operation_seq == 7
    assert source.endpoint_ref.operation_id == target.endpoint_ref.operation_id


def test_same_sequence_different_digest_conflicts():
    registry = _registry()
    registry.accept(_ref(1), lambda: OperationSnapshot.running(_ref(1)))

    with pytest.raises(OperationConflict):
        registry.accept(
            _ref(1, digest="sha256:different"),
            lambda: OperationSnapshot.running(_ref(1)),
        )


def test_operation_seq_below_floor_is_stale_without_execution():
    registry = _registry(operation_reorder_window=2)
    registry.accept(_ref(3), lambda: OperationSnapshot.running(_ref(3)))
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return OperationSnapshot.running(_ref(1))

    with pytest.raises(StaleOperation):
        registry.accept(_ref(1), execute)
    assert executed is False


def test_operation_seq_large_jump_advances_high_watermark_without_gap_wait():
    registry = _registry(operation_reorder_window=4096)
    registry.accept(
        _ref(10), lambda: OperationSnapshot(_ref(10), OperationState.COMPLETED)
    )

    jumped = registry.accept(
        _ref(10000),
        lambda: OperationSnapshot(_ref(10000), OperationState.COMPLETED),
    )
    reordered = registry.accept(
        _ref(5905),
        lambda: OperationSnapshot(_ref(5905), OperationState.COMPLETED),
    )

    assert jumped.endpoint_ref.operation_seq == 10000
    assert registry.floor == 5905
    assert reordered.endpoint_ref.operation_seq == 5905


def test_operation_seq_accepts_in_window_reorder_once():
    registry = _registry(operation_reorder_window=4)
    registry.accept(
        _ref(10), lambda: OperationSnapshot(_ref(10), OperationState.COMPLETED)
    )
    ref = _ref(8)
    executions = 0

    def execute():
        nonlocal executions
        executions += 1
        return OperationSnapshot.running(ref)

    first = registry.accept(ref, execute)
    replay = registry.accept(ref, execute)

    assert first == replay
    assert executions == 1


def test_cached_snapshot_below_floor_replays_before_stale_check():
    registry = _registry(operation_reorder_window=2)
    old = _ref(1)
    registry.accept(old, lambda: OperationSnapshot(old, OperationState.COMPLETED))
    registry.accept(
        _ref(3), lambda: OperationSnapshot(_ref(3), OperationState.COMPLETED)
    )
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return OperationSnapshot.running(old)

    replay = registry.accept(old, execute)

    assert replay.state == OperationState.COMPLETED
    assert registry.floor == 2
    assert executed is False


def test_active_registry_cap_fails_closed():
    registry = _registry(active_operation_cap=1)
    registry.accept(_ref(1), lambda: OperationSnapshot.running(_ref(1)))

    with pytest.raises(RegistryCapacity):
        registry.accept(_ref(2), lambda: OperationSnapshot.running(_ref(2)))


def test_registry_capacity_rejection_does_not_advance_sequence_window():
    registry = _registry(active_operation_cap=1, operation_reorder_window=2)
    first = _ref(1)
    registry.accept(first, lambda: OperationSnapshot.running(first))
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return OperationSnapshot.running(_ref(10000))

    with pytest.raises(RegistryCapacity):
        registry.accept(_ref(10000), execute)
    with pytest.raises(RegistryCapacity):
        registry.abort(_ref(9999), reason="publish outcome unknown")

    assert executed is False
    assert registry.floor == 1
    assert tuple(registry._seen) == (1,)

    registry.terminalize(first, OperationState.FENCED)
    second = _ref(2)
    assert registry.accept(
        second, lambda: OperationSnapshot.running(second)
    ).endpoint_ref == second


def test_held_terminal_remains_active_and_consumes_capacity():
    registry = _registry(active_operation_cap=1, terminal_snapshot_cap=1)
    held = _ref(1)
    registry.accept(
        held,
        lambda: OperationSnapshot.running(
            held, held_resource_kinds=("TARGET_PENDING",)
        ),
    )

    registry.terminalize(held, OperationState.FENCED)

    assert registry.snapshot(held).resources_held is True
    assert tuple(registry._active) == (1,)
    assert not registry._terminal
    with pytest.raises(RegistryCapacity):
        registry.accept(_ref(2), lambda: OperationSnapshot.running(_ref(2)))


def test_finalize_response_loss_replays_before_preconditions_when_resources_held_false():
    registry = _registry()
    ref = _ref(1)
    registry.accept(
        ref,
        lambda: OperationSnapshot.running(ref, held_resource_kinds=("TARGET_PENDING",)),
    )
    registry.terminalize(ref, OperationState.FENCED)
    request = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-1",
        operation_id=ref.operation_id,
        lease_id="lease-1",
        endpoint_refs=(ref,),
        resource_kinds=("TARGET_PENDING",),
    )
    releases: list[tuple[str, ...]] = []

    def release(kinds):
        releases.append(kinds)
        return {"TARGET_PENDING": 3}

    first = registry.finalize_release(request, release)
    replay = registry.finalize_release(request, release)

    assert first == replay
    assert releases == [("TARGET_PENDING",)]
    assert replay.resources_held_after is False
    assert replay.released_counts == (("TARGET_PENDING", 3),)


def test_finalize_before_predicates_not_sent_or_rejected():
    registry = _registry()
    ref = _ref(1)
    registry.accept(
        ref,
        lambda: OperationSnapshot.running(ref, held_resource_kinds=("TARGET_PENDING",)),
    )
    request = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-1",
        operation_id=ref.operation_id,
        lease_id="lease-1",
        endpoint_refs=(ref,),
        resource_kinds=("TARGET_PENDING",),
    )

    with pytest.raises(PreconditionFailed):
        registry.finalize_release(request, lambda kinds: {kind: 1 for kind in kinds})
    assert registry.snapshot(ref).resources_held is True


def test_finalize_same_cleanup_id_different_digest_conflicts():
    registry = _registry()
    ref = _ref(1)
    registry.accept(
        ref,
        lambda: OperationSnapshot.running(ref, held_resource_kinds=("TARGET_PENDING",)),
    )
    registry.terminalize(ref, OperationState.FENCED)
    request = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-1",
        operation_id=ref.operation_id,
        lease_id="lease-1",
        endpoint_refs=(ref,),
        resource_kinds=("TARGET_PENDING",),
    )
    registry.finalize_release(request, lambda kinds: {kind: 1 for kind in kinds})
    conflicting = FinalizeReleaseRequest(
        cleanup_id=request.cleanup_id,
        operation_id=request.operation_id,
        lease_id=request.lease_id,
        endpoint_refs=request.endpoint_refs,
        resource_kinds=request.resource_kinds,
        release_basis=request.release_basis,
        payload_digest="sha256:different",
    )

    with pytest.raises(OperationConflict):
        registry.finalize_release(
            conflicting, lambda kinds: {kind: 1 for kind in kinds}
        )


def test_commit_migrates_pending_ownership_without_releasing_blocks():
    registry = _registry()
    prepare = _ref(1)
    commit = EndpointOperationRef(
        topology_generation=prepare.topology_generation,
        owner_generation=prepare.owner_generation,
        operation_seq=2,
        target_instance=prepare.target_instance,
        target_worker_epoch=prepare.target_worker_epoch,
        operation_id=prepare.operation_id,
        payload_digest="sha256:commit",
    )
    registry.accept(
        prepare,
        lambda: OperationSnapshot.running(
            prepare, held_resource_kinds=("TARGET_PENDING",)
        ),
    )
    registry.accept(
        commit,
        lambda: OperationSnapshot(
            commit, OperationState.COMPLETED, True, ("TARGET_SEQUENCE",)
        ),
    )

    registry.migrate_operation_resources(prepare.operation_id, commit)

    assert registry.snapshot(prepare).resources_held is False
    assert registry.snapshot(commit).resources_held is True
    assert registry.snapshot(commit).held_resource_kinds == ("TARGET_SEQUENCE",)


def test_commit_preserves_terminal_nonheld_predecessor_result():
    registry = _registry()
    failed = _ref(1)
    commit = EndpointOperationRef(
        topology_generation=failed.topology_generation,
        owner_generation=failed.owner_generation,
        operation_seq=2,
        target_instance=failed.target_instance,
        target_worker_epoch=failed.target_worker_epoch,
        operation_id=failed.operation_id,
        payload_digest="sha256:commit",
    )
    registry.accept(
        failed, lambda: OperationSnapshot(failed, OperationState.FENCED)
    )
    registry.accept(
        commit,
        lambda: OperationSnapshot(
            commit, OperationState.COMPLETED, True, ("TARGET_SEQUENCE",)
        ),
    )

    registry.migrate_operation_resources(failed.operation_id, commit)

    assert registry.snapshot(failed).state == OperationState.FENCED
    assert registry.snapshot(commit).resources_held is True


def test_retire_resets_owner_scoped_dedup_before_new_owner_sequence_one():
    registry = _registry()
    old = _ref(1)
    registry.accept(
        old, lambda: OperationSnapshot(old, OperationState.COMPLETED)
    )
    registry.retire_owner("gateway-a:boot-a")
    registry.activate_owner("gateway-b:boot-b")
    new = _ref(1, owner="gateway-b:boot-b")
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return OperationSnapshot(new, OperationState.COMPLETED)

    assert registry.accept(new, execute).endpoint_ref.owner_generation \
        == "gateway-b:boot-b"
    assert executed is True
    with pytest.raises(RetiredOwner):
        registry.accept(old, lambda: OperationSnapshot(old, OperationState.COMPLETED))


def test_finalize_records_are_bounded_with_terminal_snapshot_eviction():
    registry = _registry(terminal_snapshot_cap=1)
    for seq in (1, 2):
        ref = _ref(seq)
        registry.accept(
            ref,
            lambda ref=ref: OperationSnapshot.running(
                ref, held_resource_kinds=("TARGET_PENDING",)
            ),
        )
        registry.terminalize(ref, OperationState.FENCED)
        request = FinalizeReleaseRequest.build(
            cleanup_id=f"cleanup-{seq}", operation_id=ref.operation_id,
            lease_id=f"lease-{seq}", endpoint_refs=(ref,),
            resource_kinds=("TARGET_PENDING",),
        )
        registry.finalize_release(
            request, lambda kinds: {kind: 1 for kind in kinds}
        )
    assert len(registry._terminal) == 1
    assert tuple(registry._finalize_by_cleanup) == ("cleanup-2",)


def test_abort_after_terminal_eviction_is_stale():
    registry = _registry(terminal_snapshot_cap=1)
    evicted = _ref(1)
    registry.accept(
        evicted, lambda: OperationSnapshot(evicted, OperationState.COMPLETED)
    )
    retained = _ref(2)
    registry.accept(
        retained, lambda: OperationSnapshot(retained, OperationState.COMPLETED)
    )

    with pytest.raises(StaleOperation):
        registry.abort(evicted, reason="late cancellation")


def test_abort_after_terminal_eviction_with_different_digest_conflicts():
    registry = _registry(terminal_snapshot_cap=1)
    evicted = _ref(1)
    registry.accept(
        evicted, lambda: OperationSnapshot(evicted, OperationState.COMPLETED)
    )
    retained = _ref(2)
    registry.accept(
        retained, lambda: OperationSnapshot(retained, OperationState.COMPLETED)
    )

    with pytest.raises(OperationConflict):
        registry.abort(
            _ref(1, digest="sha256:different"), reason="late cancellation"
        )


def test_terminal_cache_evicts_minimum_seq_out_of_order():
    registry = _registry(terminal_snapshot_cap=1)
    high = _ref(3)
    registry.accept(high, lambda: OperationSnapshot(high, OperationState.COMPLETED))
    low = _ref(1)
    registry.accept(low, lambda: OperationSnapshot(low, OperationState.COMPLETED))

    assert tuple(registry._terminal) == (3,)
    assert registry.snapshot(high).state == OperationState.COMPLETED
    executed = False

    def execute():
        nonlocal executed
        executed = True
        return OperationSnapshot.running(low)

    with pytest.raises(StaleOperation):
        registry.accept(low, execute)
    assert executed is False
