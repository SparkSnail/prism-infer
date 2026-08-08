import asyncio
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from prism_infer.server.app import WorkerControlRuntime, create_app
from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    FinalizeReleaseRequest,
    OperationRegistry,
    canonical_payload_digest,
    OperationSnapshot,
    OperationState,
)
from prism_infer.server.runtime import (
    EngineControlRouter,
    EngineOwnerCommandQueue,
    MappedNCCLEndpoint,
    PDExecutionDriver,
    endpoint_ref_key,
)


def _runtime():
    registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
    )
    return WorkerControlRuntime(
        identity={
            "instance_id": "d0",
            "role": "decode",
            "topology_generation": "world-a",
            "pod_uid": "pod-d0",
            "process_generation": "boot-a",
            "instance_epoch": "pod-d0:boot-a",
            "rpc_endpoint": "http://d0:8001",
            "global_rank": 2,
            "topology_digest": "sha256:topology-a",
            "kv_compatibility_id": "kv-a",
        },
        capabilities={"pairs": [], "ready": True},
        registry=registry,
    )


def _ref(seq=1, payload=None):
    payload = {} if payload is None else payload
    return EndpointOperationRef(
        topology_generation="world-a",
        owner_generation="gateway-a:boot-a",
        operation_seq=seq,
        target_instance="d0",
        target_worker_epoch="pod-d0:boot-a",
        operation_id=f"op-{seq}",
        payload_digest=canonical_payload_digest(payload),
    )


def test_not_ready_worker_rejects_new_owner_and_mutation_without_registry_state():
    runtime = _runtime()
    runtime.capabilities.update({
        "ready": False,
        "failure_kind": "nccl_watchdog_timeout",
    })
    client = TestClient(create_app(runtime))
    payload = {"held_resource_kinds": ["TARGET_PENDING"]}
    ref = _ref(payload=payload)

    owner_response = client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    )
    mutation_response = client.post(
        "/v1/requests/prepare",
        json={
            "schema_version": 1,
            "endpoint_ref": asdict(ref),
            "payload": payload,
        },
    )

    assert owner_response.status_code == 503
    assert mutation_response.status_code == 503
    assert runtime.registry.owner_status()["active_owner"] is None
    assert runtime.registry.snapshots() == ()


def test_real_http_identity_activate_prepare_abort_and_finalize_replay():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    assert client.get("/v1/identity").json()["instance_epoch"] == "pod-d0:boot-a"
    assert client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    ).status_code == 200
    payload = {"held_resource_kinds": ["TARGET_PENDING"]}
    ref = _ref(payload=payload)
    envelope = {
        "schema_version": 1,
        "endpoint_ref": ref.__dict__,
        "payload": payload,
    }
    assert client.post("/v1/requests/prepare", json=envelope).status_code == 202
    aborted = client.post(
        f"/v1/requests/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "cancel"},
    )
    assert aborted.status_code == 200
    assert aborted.json()["resources_held"] is True
    finalize = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-1",
        operation_id=ref.operation_id,
        lease_id="lease-1",
        endpoint_refs=(ref,),
        resource_kinds=("TARGET_PENDING",),
    )
    body = {
        **finalize.__dict__,
        "endpoint_refs": [endpoint.__dict__ for endpoint in finalize.endpoint_refs],
    }
    first = client.post("/v1/cleanup/finalize", json=body)
    replay = client.post("/v1/cleanup/finalize", json=body)
    assert first.status_code == 200
    assert first.json() == replay.json()
    assert first.json()["released_counts"] == [["TARGET_PENDING", 1]]
    assert runtime.released_resource_kinds == [("TARGET_PENDING",)]


def test_local_reuse_commit_terminalizes_superseded_nonheld_prepare():
    """Commit supersedes a resource-free local-reuse prepare."""
    runtime = _runtime()

    def execute(operation, ref, _payload):
        if operation == "prefix.prepare":
            return OperationSnapshot(ref, OperationState.PREPARED)
        assert operation == "prefix.commit"
        return OperationSnapshot(
            ref,
            OperationState.COMPLETED,
            resources_held=True,
            held_resource_kinds=("TARGET_SEQUENCE",),
        )

    runtime.command_handler = execute
    client = TestClient(create_app(runtime))
    owner = "gateway-a:boot-a"
    assert client.post(
        "/v1/owners/activate", json={"owner_generation": owner}
    ).status_code == 200

    operation_id = "local-reuse-op"

    def ref(seq, payload):
        return EndpointOperationRef(
            "world-a", owner, seq, "d0", "pod-d0:boot-a",
            operation_id, canonical_payload_digest(payload),
        )

    prepare_payload = {"mode": "local_reuse"}
    prepare_ref = ref(1, prepare_payload)
    prepare_envelope = {
        "schema_version": 1,
        "endpoint_ref": prepare_ref.__dict__,
        "payload": prepare_payload,
    }
    prepared = client.post("/v1/prefix/prepare", json=prepare_envelope)
    assert prepared.status_code == 202
    assert prepared.json()["state"] == "PREPARED"
    assert prepared.json()["resources_held"] is False

    commit_payload = {"mode": "local_reuse"}
    commit_ref = ref(2, commit_payload)
    committed = client.post("/v1/prefix/commit", json={
        "schema_version": 1,
        "endpoint_ref": commit_ref.__dict__,
        "payload": commit_payload,
    })
    assert committed.status_code == 200
    assert committed.json()["held_resource_kinds"] == ["TARGET_SEQUENCE"]

    superseded = client.post(
        "/v1/operations/status", json=prepare_ref.__dict__
    ).json()
    assert superseded["state"] == "COMPLETED"
    assert superseded["resources_held"] is False
    replay = client.post("/v1/prefix/prepare", json=prepare_envelope)
    assert replay.status_code == 200
    assert replay.json()["state"] == "COMPLETED"
    assert runtime.registry.snapshot(commit_ref).resources_held is True


@pytest.mark.parametrize(
    ("path", "operation"),
    [
        ("/v1/transfers/prepare-receive", "transfer.prepare_receive"),
        ("/v1/transfers/start", "transfer.start"),
    ],
)
def test_nonterminal_http_mutation_replay_does_not_execute_twice(
    path: str, operation: str,
):
    calls = []
    runtime = _runtime()

    def execute(actual_operation, ref, _payload):
        calls.append(actual_operation)
        return OperationSnapshot(ref, OperationState.UNKNOWN)

    runtime.command_handler = execute
    client = TestClient(create_app(runtime))
    assert client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    ).status_code == 200
    payload = {"kind": operation}
    ref = _ref(payload=payload)
    envelope = {
        "schema_version": 1,
        "endpoint_ref": ref.__dict__,
        "payload": payload,
    }

    first = client.post(path, json=envelope)
    replay = client.post(path, json=envelope)

    assert first.status_code == replay.status_code == 202
    assert first.json() == replay.json()
    assert first.json()["execution_count"] == 1
    assert calls == [operation]


def test_nats_fault_target_http_prepare_execution_once_without_delivery():
    """HTTP prepare is target-local and must not count as NATS delivery."""
    calls = []
    runtime = _runtime()

    def execute(operation, ref, _payload):
        calls.append(operation)
        return OperationSnapshot(ref, OperationState.UNKNOWN)

    runtime.command_handler = execute
    client = TestClient(create_app(runtime))
    assert client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    ).status_code == 200
    payload = {"kind": "target-http-prepare"}
    ref = _ref(payload=payload)
    envelope = {
        "schema_version": 1,
        "endpoint_ref": ref.__dict__,
        "payload": payload,
    }

    first = client.post("/v1/transfers/prepare-receive", json=envelope)
    replay = client.post("/v1/transfers/prepare-receive", json=envelope)
    status = client.post("/v1/operations/status", json=ref.__dict__)

    assert first.status_code == replay.status_code == 202
    assert status.status_code == 200
    assert first.json() == replay.json() == status.json()
    assert (
        status.json()["delivery_count"],
        status.json()["execution_count"],
    ) == (0, 1)
    assert calls == ["transfer.prepare_receive"]


def test_abort_replays_completed_without_invoking_physical_abort_handler():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    assert client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    ).status_code == 200
    ref = _ref()
    runtime.registry.accept(ref, lambda: OperationSnapshot.running(ref))
    completed = runtime.registry.terminalize(
        ref, OperationState.COMPLETED, reason="engine completed"
    )
    calls = []

    def abort_handler(value):
        calls.append(value)
        return OperationSnapshot(value, OperationState.FENCED)

    runtime.abort_handler = abort_handler
    response = client.post(
        f"/v1/requests/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "timeout"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "COMPLETED"
    assert runtime.registry.snapshot(ref) == completed
    assert calls == []


def test_completion_while_abort_handler_runs_remains_first_terminal():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    assert client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    ).status_code == 200
    ref = _ref()
    runtime.registry.accept(ref, lambda: OperationSnapshot.running(ref))

    def abort_handler(value):
        runtime.registry.terminalize(
            value, OperationState.COMPLETED, reason="completion won"
        )
        return OperationSnapshot(
            value, OperationState.FENCED, reason="late abort"
        )

    runtime.abort_handler = abort_handler
    response = client.post(
        f"/v1/requests/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "timeout"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "COMPLETED"
    assert response.json()["reason"] == "completion won"
    assert runtime.registry.snapshot(ref).state == OperationState.COMPLETED


def test_executor_before_allocation_exact_ref_finalize_releases_zero_blocks():
    from collections import deque
    from types import SimpleNamespace

    runtime = _runtime()
    driver = PDExecutionDriver(
        SimpleNamespace(scheduler=SimpleNamespace(
            waiting=deque(), running=deque(),
            block_manager=SimpleNamespace(
                deallocate=lambda _sequence: (_ for _ in ()).throw(
                    AssertionError("no sequence was allocated")
                )
            ),
        )),
        role="prefill",
    )
    runtime.release_handler = (
        lambda operation_id, _kinds: driver.release_source_blocks(operation_id)
    )
    client = TestClient(create_app(runtime))
    owner = "gateway-a:boot-a"
    assert client.post(
        "/v1/owners/activate", json={"owner_generation": owner}
    ).status_code == 200
    payload = {"held_resource_kinds": ["SOURCE_BLOCKS"]}
    ref = _ref(payload=payload)
    runtime.registry.accept(
        ref,
        lambda: OperationSnapshot.running(
            ref, held_resource_kinds=("SOURCE_BLOCKS",)
        ),
    )
    runtime.registry.terminalize(
        ref, OperationState.FENCED, reason="executor failed before allocation"
    )
    finalize = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-zero", operation_id=ref.operation_id,
        lease_id="lease-zero", endpoint_refs=(ref,),
        resource_kinds=("SOURCE_BLOCKS",),
    )
    body = {
        **finalize.__dict__,
        "endpoint_refs": [value.__dict__ for value in finalize.endpoint_refs],
    }

    first = client.post("/v1/cleanup/finalize", json=body)
    replay = client.post("/v1/cleanup/finalize", json=body)

    assert first.status_code == 200
    assert replay.json() == first.json()
    assert first.json()["released_counts"] == [["SOURCE_BLOCKS", 0]]
    assert first.json()["resources_held_after"] is False
    assert runtime.registry.snapshot(ref).resources_held is False


def test_rejected_wrong_owner_refs_do_not_grow_operation_kind_side_state():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    assert client.post(
        "/v1/owners/activate",
        json={"owner_generation": "gateway-a:boot-a"},
    ).status_code == 200

    for seq in range(1, 600):
        ref = _ref(seq=seq)
        body = {
            "schema_version": 1,
            "endpoint_ref": {
                **ref.__dict__,
                "owner_generation": "stale-gateway:boot-old",
            },
            "payload": {},
        }
        assert client.post("/v1/requests/prepare", json=body).status_code == 409

    assert runtime.operation_kinds == {}
    assert runtime.registry.snapshots() == ()


def test_http_abort_before_dispatch_prevents_late_mutation():
    runtime = _runtime()
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    ref = _ref()
    abort = client.post(
        f"/v1/transfers/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "publish unknown"},
    )
    late = client.post(
        "/v1/transfers/start",
        json={"schema_version": 1, "endpoint_ref": ref.__dict__, "payload": {}},
    )

    assert abort.status_code == 200
    assert late.status_code == 200
    assert late.json()["state"] == "FENCED"
    assert physical_aborts == []


def test_http_abort_stops_exact_writer_before_persisting_fenced():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    ref = _ref()
    runtime.registry.accept(ref, lambda: OperationSnapshot.running(ref))
    observed_states = []

    def abort_handler(value):
        observed_states.append(runtime.registry.snapshot(value).state)
        return OperationSnapshot(value, OperationState.FENCED)

    runtime.abort_handler = abort_handler
    response = client.post(
        f"/v1/requests/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "timeout"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "FENCED"
    assert observed_states == [OperationState.RUNNING]


def test_http_abort_new_seq_collision_has_zero_physical_side_effect():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    original = _ref(1)
    runtime.registry.accept(
        original, lambda: OperationSnapshot.running(original)
    )
    collision = replace(_ref(2), operation_id=original.operation_id)
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append

    response = client.post(
        f"/v1/requests/{collision.operation_id}/abort",
        json={
            "target_operation_ref": collision.__dict__,
            "reason": "wrong new ref",
        },
    )

    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT"
    assert physical_aborts == []
    assert tuple(runtime.registry._seen) == (1,)
    assert runtime.registry.snapshot(original).state == OperationState.RUNNING


def test_http_abort_wrong_digest_has_zero_physical_side_effect():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    original = _ref(1)
    runtime.registry.accept(
        original, lambda: OperationSnapshot.running(original)
    )
    wrong_digest = replace(original, payload_digest="sha256:wrong")
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append

    response = client.post(
        f"/v1/requests/{wrong_digest.operation_id}/abort",
        json={
            "target_operation_ref": wrong_digest.__dict__,
            "reason": "wrong digest",
        },
    )

    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT"
    assert physical_aborts == []
    assert runtime.registry.snapshot(original).state == OperationState.RUNNING


def test_http_abort_same_seq_different_operation_id_has_zero_physical_side_effect():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    original = _ref(1)
    runtime.registry.accept(
        original, lambda: OperationSnapshot.running(original)
    )
    wrong_operation = replace(original, operation_id="another-operation")
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append

    response = client.post(
        f"/v1/requests/{wrong_operation.operation_id}/abort",
        json={
            "target_operation_ref": wrong_operation.__dict__,
            "reason": "wrong operation id",
        },
    )

    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT"
    assert physical_aborts == []
    assert runtime.registry.snapshot(original).state == OperationState.RUNNING


def test_http_abort_seen_evicted_ref_has_zero_physical_side_effect():
    runtime = _runtime()
    runtime.registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
        terminal_snapshot_cap=1,
    )
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    evicted = _ref(1)
    retained = _ref(2)
    runtime.registry.accept(
        evicted, lambda: OperationSnapshot(evicted, OperationState.COMPLETED)
    )
    runtime.registry.accept(
        retained, lambda: OperationSnapshot(retained, OperationState.COMPLETED)
    )
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append

    response = client.post(
        f"/v1/requests/{evicted.operation_id}/abort",
        json={"target_operation_ref": evicted.__dict__, "reason": "late"},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "STALE_OPERATION"
    assert physical_aborts == []


def test_http_abort_below_floor_has_zero_physical_side_effect():
    runtime = _runtime()
    runtime.registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
        operation_reorder_window=1,
        terminal_snapshot_cap=1,
    )
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    stale = _ref(1)
    current = _ref(2)
    runtime.registry.accept(
        stale, lambda: OperationSnapshot(stale, OperationState.COMPLETED)
    )
    runtime.registry.accept(
        current, lambda: OperationSnapshot(current, OperationState.COMPLETED)
    )
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append

    response = client.post(
        f"/v1/requests/{stale.operation_id}/abort",
        json={"target_operation_ref": stale.__dict__, "reason": "too old"},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "STALE_OPERATION"
    assert physical_aborts == []


def test_http_abort_capacity_rejection_has_zero_physical_side_effect():
    runtime = _runtime()
    runtime.registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
        active_operation_cap=1,
    )
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    active = _ref(1)
    runtime.registry.accept(active, lambda: OperationSnapshot.running(active))
    rejected = _ref(2)
    physical_aborts = []
    runtime.abort_handler = physical_aborts.append

    response = client.post(
        f"/v1/requests/{rejected.operation_id}/abort",
        json={
            "target_operation_ref": rejected.__dict__,
            "reason": "registry full",
        },
    )

    assert response.status_code == 503
    assert response.json()["code"] == "REGISTRY_CAPACITY"
    assert physical_aborts == []
    assert tuple(runtime.registry._seen) == (1,)


def test_http_finalize_before_terminal_is_rejected_without_release():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    payload = {"held_resource_kinds": ["TARGET_PENDING"]}
    ref = _ref(payload=payload)
    client.post(
        "/v1/requests/prepare",
        json={
            "schema_version": 1,
            "endpoint_ref": ref.__dict__,
            "payload": payload,
        },
    )
    finalize = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-1",
        operation_id=ref.operation_id,
        lease_id="lease-1",
        endpoint_refs=(ref,),
        resource_kinds=("TARGET_PENDING",),
    )
    body = {
        **finalize.__dict__,
        "endpoint_refs": [endpoint.__dict__ for endpoint in finalize.endpoint_refs],
    }

    response = client.post("/v1/cleanup/finalize", json=body)

    assert response.status_code == 409
    assert response.json()["code"] == "PRECONDITION_FAILED"
    assert runtime.released_resource_kinds == []


def test_owner_retire_requires_terminal_resource_free_operations():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    owner = "gateway-a:boot-a"
    client.post("/v1/owners/activate", json={"owner_generation": owner})
    ref = _ref()
    client.post(
        "/v1/requests/prepare",
        json={"schema_version": 1, "endpoint_ref": ref.__dict__, "payload": {}},
    )

    blocked = client.post(f"/v1/owners/{owner}/retire")
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "PRECONDITION_FAILED"

    client.post(
        f"/v1/requests/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "orphan sweep"},
    )
    retired = client.post(f"/v1/owners/{owner}/retire")
    assert retired.status_code == 200
    assert retired.json()["active_owner"] is None
    assert owner in retired.json()["retired_owners"]
    assert client.post(
        "/v1/owners/activate", json={"owner_generation": owner}
    ).json()["code"] == "RETIRED_OWNER"


def test_receive_launch_error_stays_unknown_and_blocks_owner_retirement():
    """Ambiguous NCCL launch failure must not become an outer FENCED proof."""
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch

    class Buffer:
        def numel(self):
            return 1

    class Cache:
        shape = (2, 1, 4, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, _key):
            return Buffer()

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 0

    class CudaEvent:
        def query(self):
            return False

    fail_launch = threading.Event()

    def batch_isend_irecv(_ops):
        fail_launch.wait(timeout=1.0)
        raise RuntimeError("synthetic launch failure")

    mapped = MappedNCCLEndpoint(PairGroups(), Cache())

    def route(operation, ref, payload):
        if operation == "transfer.prepare_receive":
            return mapped.prepare_receive(ref, payload)
        if operation == "status.local":
            return mapped.refresh(ref)
        if operation == "prune.local":
            mapped.prune(set(payload["refs"]))
            return None
        raise AssertionError(operation)

    owner_queue = EngineOwnerCommandQueue(route)
    runtime = _runtime()
    runtime.async_command_handler = owner_queue.submit_async
    runtime.async_status_handler = lambda ref: owner_queue.submit_async(
        "status.local", ref, {}
    )
    runtime.async_prune_handler = lambda refs, operation_ids: (
        owner_queue.submit_local_async(
            "prune.local", {"refs": refs, "operation_ids": operation_ids}
        )
    )
    owner = "gateway-a:boot-a"
    payload = {
        "source_instance": "p0", "target_instance": "d0",
        "src_block_ids": [1], "dst_block_ids": [2], "kv_size_bytes": 1,
    }
    ref = EndpointOperationRef(
        "world-a", owner, 1, "d0", "pod-d0:boot-a", "transfer-op",
        canonical_payload_digest(payload),
    )
    client = TestClient(create_app(runtime))
    try:
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch, "empty_like", side_effect=lambda value: value), \
             patch.object(torch.distributed, "P2POp", return_value=object()), \
             patch.object(
                 torch.distributed, "batch_isend_irecv",
                 side_effect=batch_isend_irecv,
             ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
            client.post(
                "/v1/owners/activate", json={"owner_generation": owner}
            )
            accepted = client.post("/v1/transfers/prepare-receive", json={
                "schema_version": 1, "endpoint_ref": ref.__dict__,
                "payload": payload,
            })
            assert accepted.status_code == 202
            assert accepted.json()["state"] == "UNKNOWN"

            key = endpoint_ref_key(ref)
            fail_launch.set()
            assert isinstance(
                mapped._pending_receive_launches[key][0].exception(timeout=1.0),
                RuntimeError,
            )
            refreshed = client.get(f"/v1/transfers/{ref.operation_id}")
            assert refreshed.status_code == 200
            assert refreshed.json()["state"] == "UNKNOWN"
            assert "synthetic launch failure" in refreshed.json()["reason"]

            listed = client.get("/v1/operations").json()["operations"]
            assert listed[0]["state"] == "UNKNOWN"
            retired = client.post(f"/v1/owners/{owner}/retire")
            assert retired.status_code == 409
            assert retired.json()["code"] == "PRECONDITION_FAILED"
            assert mapped.registry.status(key).status.value == "UNKNOWN"
            assert key in mapped._failed_receive_buffers
            assert key in mapped._launched_at
    finally:
        fail_launch.set()
        owner_queue.close()


def test_output_query_returns_epoch_fenced_cumulative_tokens():
    runtime = _runtime()
    runtime.output_handler = lambda req_id, after_seq: {
        "req_id": req_id,
        "instance_epoch": "pod-d0:boot-a",
        "output_seq_no": 3,
        "token_ids": [7, 8, 9],
        "terminal": False,
    }
    response = TestClient(create_app(runtime)).get(
        "/v1/requests/r1/output?after_seq=1"
    )
    assert response.status_code == 200
    assert response.json()["token_ids"] == [7, 8, 9]


def test_output_query_for_unknown_request_returns_not_found():
    """Unknown or cleaned request ids return protocol errors, not ASGI 500."""
    runtime = _runtime()
    driver = PDExecutionDriver(SimpleNamespace(
        prefix_cache=SimpleNamespace(instance_epoch=runtime.registry.instance_epoch)
    ), role="decode")
    runtime.output_handler = driver.output

    response = TestClient(
        create_app(runtime), raise_server_exceptions=False
    ).get("/v1/requests/missing/output?after_seq=0")

    assert response.status_code == 404
    assert response.json() == {
        "code": "REQUEST_OUTPUT_NOT_FOUND",
        "message": "request output is not available: missing",
    }


def test_http_payload_digest_mismatch_has_zero_mutation():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    ref = _ref(payload={})
    response = client.post(
        "/v1/requests/prepare",
        json={
            "schema_version": 1,
            "endpoint_ref": ref.__dict__,
            "payload": {"held_resource_kinds": ["TARGET_PENDING"]},
        },
    )
    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT"
    assert runtime.registry.snapshots() == ()


def test_http_abort_path_must_match_original_operation_ref():
    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    ref = _ref()
    response = client.post(
        "/v1/transfers/not-the-ref-operation/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "bad path"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "CONFLICT"
    assert runtime.registry.snapshots() == ()


def test_http_abort_incomplete_work_and_cuda_remains_unknown():
    class Work:
        def is_completed(self):
            return False

    class Event:
        def query(self):
            return False

    class PairGroups:
        def pair(self, pair_id):
            return type("Pair", (), {
                "pair_id": pair_id, "process_group": None,
            })()

        def global_peer(self, pair_id):
            return 2

    import torch

    runtime = _runtime()
    client = TestClient(create_app(runtime))
    client.post(
        "/v1/owners/activate", json={"owner_generation": "gateway-a:boot-a"}
    )
    payload = {
        "source_instance": "p0", "target_instance": "d0",
        "src_block_ids": [0], "dst_block_ids": [1],
        "held_resource_kinds": ["SOURCE_RETAIN", "TRANSFER_BYTES"],
    }
    ref = _ref(payload=payload)
    controller = MappedNCCLEndpoint(
        PairGroups(), torch.zeros(2, 1, 2, 1, 1, 1)
    )
    key = endpoint_ref_key(ref)
    controller._operation_keys[ref.operation_id] = key
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [Work()], Event())
    runtime.registry.accept(
        ref,
        lambda: OperationSnapshot.running(
            ref, held_resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
        ),
    )

    def abort_handler(value):
        controller.abort(value)
        return controller.refresh(value)

    runtime.abort_handler = abort_handler
    runtime.status_handler = controller.refresh
    response = client.post(
        f"/v1/transfers/{ref.operation_id}/abort",
        json={"target_operation_ref": ref.__dict__, "reason": "timeout"},
    )
    assert response.status_code == 200
    assert response.json()["state"] == "UNKNOWN"
    assert response.json()["resources_held"] is True
    assert runtime.registry.snapshot(ref).state == OperationState.UNKNOWN


def test_prefix_directory_http_register_peek_and_ack_bind_live_handler():
    runtime = _runtime()
    calls = []

    def handler(action, value):
        calls.append((action, value))
        if action == "register":
            return {
                "instance_id": "d0", "instance_epoch": "pod-d0:boot-a",
                "snapshot_seq_no": 0, "locations": [],
            }
        if action == "peek":
            return []

    runtime.prefix_directory_handler = handler
    client = TestClient(create_app(runtime))
    assert client.post(
        "/v1/prefix/reports/register",
        json={"consumer_id": "gateway", "generation": "g1"},
    ).status_code == 200
    assert client.get(
        "/v1/prefix/events?consumer_id=gateway&generation=g1&after_seq=0&limit=10"
    ).json() == {"events": []}
    assert client.post(
        "/v1/prefix/events/ack",
        json={"consumer_id": "gateway", "generation": "g1", "up_to_seq": 0},
    ).json() == {"acked": True}
    assert [item[0] for item in calls] == ["register", "peek", "ack"]


def test_production_prepare_transfer_commit_suffix_abort_finalize_and_retire(monkeypatch):
    """Real router/app lifecycle; repeated finalize models response loss replay."""
    from types import SimpleNamespace
    from prism_infer.engine.prefix_cache import PrefixCacheService
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import SequenceStatus
    from prism_infer.engine.sequence import Sequence

    monkeypatch.setattr(Sequence, "block_size", 4)

    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=8, eos=999,
        kvcache_block_size=4, num_kvcache_blocks=8,
    ))
    engine = SimpleNamespace(scheduler=scheduler)
    engine.prefix_cache = PrefixCacheService(scheduler.block_manager)
    engine.model_runner = SimpleNamespace(kv_cache=None)

    def step():
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [7] * len(seqs), is_prefill)

    engine.step = step
    driver = PDExecutionDriver(engine, role="decode")
    control = EngineControlRouter(
        engine,
        prepare_receive=lambda ref, payload: OperationSnapshot(
            ref, OperationState.COMPLETED
        ),
        transfer_terminal=lambda operation_id, ref=None: operation_id == "transfer-op",
        request_committed=driver.request_committed,
    )
    runtime = _runtime()

    def route(operation, ref, payload):
        if operation == "dispatch.suffix":
            return driver("suffix", ref, payload)
        return control(operation, ref, payload)

    def abort(ref):
        stopped = driver.abort_request(ref.operation_id)
        stopped = engine.prefix_cache.abort_sequence(ref.operation_id) or stopped
        if not stopped:
            return None
        current = runtime.registry.snapshot(ref)
        return OperationSnapshot(
            ref, OperationState.FENCED,
            resources_held=current.resources_held,
            held_resource_kinds=current.held_resource_kinds,
        )

    runtime.command_handler = route
    runtime.abort_handler = abort
    runtime.release_handler = control.release
    app = create_app(runtime)
    client = TestClient(app)
    owner = "gateway-a:boot-a"
    assert client.post(
        "/v1/owners/activate", json={"owner_generation": owner}
    ).status_code == 200

    def ref(seq, operation_id, payload):
        return EndpointOperationRef(
            "world-a", owner, seq, "d0", "pod-d0:boot-a",
            operation_id, canonical_payload_digest(payload),
        )

    prepare_payload = {
        "req_id": "request-1", "mode": "remote_transfer", "block_count": 1,
        "token_ids": [1, 2, 3, 4, 5], "sampling_params": {},
    }
    prepare_ref = ref(1, "op", prepare_payload)
    assert client.post("/v1/requests/prepare", json={
        "schema_version": 1, "endpoint_ref": prepare_ref.__dict__,
        "payload": prepare_payload,
    }).status_code == 202

    transfer_payload = {"dst_block_ids": [0]}
    transfer_ref = ref(2, "transfer-op", transfer_payload)
    assert client.post("/v1/transfers/prepare-receive", json={
        "schema_version": 1, "endpoint_ref": transfer_ref.__dict__,
        "payload": transfer_payload,
    }).status_code == 200

    commit_payload = {
        "req_id": "request-1", "transfer_operation_id": "transfer-op",
        "transfer_endpoint_ref": transfer_ref.__dict__, "first_token": 7,
        "namespace": "", "kv_compatibility_id": "",
        "request_context_digest": "", "cached_prefix_tokens": 5,
        "operation_id": "op",
    }
    commit_ref = ref(3, "op", commit_payload)
    assert client.post("/v1/requests/commit", json={
        "schema_version": 1, "endpoint_ref": commit_ref.__dict__,
        "payload": commit_payload,
    }).status_code == 200
    prepared_status = client.post(
        "/v1/operations/status", json=prepare_ref.__dict__
    ).json()
    assert prepared_status["state"] == "COMPLETED"
    assert prepared_status["resources_held"] is False
    committed_sequence = driver._operations["op"]
    assert committed_sequence.num_prompt_tokens == 5
    assert committed_sequence.num_cached_tokens == 5
    assert committed_sequence.completion_token_ids == [7]
    scheduled, is_prefill = scheduler.schedule()
    assert scheduled == [committed_sequence]
    assert is_prefill is False
    scheduler.postprocess(scheduled, [8], is_prefill)
    assert committed_sequence.completion_token_ids[:2] == [7, 8]

    suffix_payload = {
        "req_id": "request-1", "remaining_token_ids": [5],
        "first_token_subject": "first.owner", "decode_progress_subject": "progress.owner",
        "decode_done_subject": "done.owner",
    }
    suffix_ref = ref(4, "op", suffix_payload)
    runtime.operation_kinds[suffix_ref] = "dispatch.suffix"
    runtime.registry.accept(
        suffix_ref,
        lambda: OperationSnapshot.running(suffix_ref),
    )
    suffix = runtime.execute("dispatch.suffix", suffix_ref, suffix_payload)
    sequence = driver._operations["op"]
    assert sequence in scheduler.running
    assert suffix.resources_held is False

    aborted = client.post("/v1/requests/op/abort", json={
        "target_operation_ref": suffix_ref.__dict__, "reason": "suffix timeout",
    })
    assert aborted.json()["state"] == "FENCED"
    assert sequence.status == SequenceStatus.ABORTED
    assert scheduler.schedule() == ([], False)
    assert driver.events.empty()
    late_completion = runtime.registry.store_result(suffix_ref, suffix)
    assert late_completion.state == OperationState.FENCED

    finalize = FinalizeReleaseRequest.build(
        cleanup_id="cleanup-op", operation_id="op", lease_id="lease-op",
        endpoint_refs=(commit_ref,), resource_kinds=("TARGET_SEQUENCE",),
    )
    finalize_body = {
        **finalize.__dict__,
        "endpoint_refs": [value.__dict__ for value in finalize.endpoint_refs],
    }
    first = client.post("/v1/cleanup/finalize", json=finalize_body)
    replay = client.post("/v1/cleanup/finalize", json=finalize_body)
    assert first.status_code == 200
    assert replay.json() == first.json()
    assert client.post(f"/v1/owners/{owner}/retire").status_code == 200


def test_remote_prefix_commit_requires_exact_completed_target_transfer_ref(monkeypatch):
    from types import SimpleNamespace
    import torch

    from prism_infer.engine.prefix_cache import PrefixCacheService
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import Sequence

    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=8, eos=999,
        kvcache_block_size=4, num_kvcache_blocks=8,
    ))
    engine = SimpleNamespace(scheduler=scheduler)
    engine.prefix_cache = PrefixCacheService(scheduler.block_manager)

    class PairGroups:
        def pair(self, pair_id):
            return type("Pair", (), {"pair_id": pair_id, "process_group": None})()

    mapped = MappedNCCLEndpoint(
        PairGroups(), torch.zeros(2, 1, 8, 1, 1, 1)
    )
    control = EngineControlRouter(
        engine, prepare_receive=mapped.prepare_receive,
        transfer_terminal=mapped.operation_completed,
    )
    runtime = _runtime()
    runtime.command_handler = control
    client = TestClient(create_app(runtime))
    owner = "gateway-a:boot-a"
    client.post("/v1/owners/activate", json={"owner_generation": owner})

    def ref(seq, payload):
        return EndpointOperationRef(
            "world-a", owner, seq, "d0", "pod-d0:boot-a", "prefix-op",
            canonical_payload_digest(payload),
        )

    prepare_payload = {
        "req_id": "request-1", "mode": "remote_transfer", "block_count": 1,
        "token_ids": [1, 2, 3, 4, 5], "sampling_params": {},
    }
    prepare_ref = ref(1, prepare_payload)
    assert client.post("/v1/prefix/prepare", json={
        "schema_version": 1, "endpoint_ref": prepare_ref.__dict__,
        "payload": prepare_payload,
    }).status_code == 202
    dst_block = client.post("/v1/operations/status", json=prepare_ref.__dict__).json()[
        "result"
    ]["dst_block_ids"][0]

    transfer_payload = {
        "source_instance": "p0", "target_instance": "d0",
        "dst_block_ids": [dst_block],
    }
    transfer_ref = ref(2, transfer_payload)
    assert client.post("/v1/transfers/prepare-receive", json={
        "schema_version": 1, "endpoint_ref": transfer_ref.__dict__,
        "payload": transfer_payload,
    }).status_code == 200

    wrong_transfer_ref = EndpointOperationRef(
        "world-a", owner, 99, "d0", "pod-d0:boot-a", "prefix-op",
        "sha256:not-the-completed-transfer",
    )
    rejected_payload = {
        "mode": "remote_transfer", "transfer_operation_id": "prefix-op",
        "transfer_endpoint_ref": wrong_transfer_ref.__dict__, "namespace": "",
        "kv_compatibility_id": "", "request_context_digest": "",
        "cached_prefix_tokens": 4,
    }
    rejected_ref = ref(3, rejected_payload)
    rejected = client.post("/v1/prefix/commit", json={
        "schema_version": 1, "endpoint_ref": rejected_ref.__dict__,
        "payload": rejected_payload,
    })
    assert rejected.status_code == 422
    assert engine.prefix_cache._operations["prefix-op"].sequence is None

    commit_payload = {
        **rejected_payload, "transfer_endpoint_ref": transfer_ref.__dict__,
    }
    commit_ref = ref(4, commit_payload)
    response = client.post("/v1/prefix/commit", json={
        "schema_version": 1, "endpoint_ref": commit_ref.__dict__,
        "payload": commit_payload,
    })

    assert response.status_code == 200
    assert engine.prefix_cache._operations["prefix-op"].sequence is not None


@pytest.mark.asyncio
async def test_inflight_owner_mutation_keeps_http_loop_responsive_and_deduplicated():
    import threading
    import httpx

    entered = threading.Event()
    release = threading.Event()
    executions = 0

    def handler(operation, ref, payload):
        nonlocal executions
        if operation == "abort.local":
            current = runtime.registry.snapshot(ref)
            return OperationSnapshot(
                ref, OperationState.FENCED, current.resources_held,
                current.held_resource_kinds,
            )
        executions += 1
        entered.set()
        assert release.wait(2)
        return OperationSnapshot(
            ref, OperationState.COMPLETED, True, ("TARGET_PENDING",)
        )

    owner = EngineOwnerCommandQueue(handler)
    runtime = _runtime()
    runtime.async_command_handler = owner.submit_async
    runtime.async_abort_handler = lambda ref: owner.submit_async(
        "abort.local", ref, {}
    )
    app = create_app(runtime)
    payload = {"held_resource_kinds": ["TARGET_PENDING"]}
    ref = _ref(payload=payload)
    envelope = {
        "schema_version": 1, "endpoint_ref": ref.__dict__, "payload": payload,
    }
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://worker"
        ) as client:
            await client.post(
                "/v1/owners/activate",
                json={"owner_generation": "gateway-a:boot-a"},
            )
            first = asyncio.create_task(client.post("/v1/requests/prepare", json=envelope))
            assert await asyncio.to_thread(entered.wait, 1)
            duplicate = asyncio.create_task(
                client.post("/v1/requests/prepare", json=envelope)
            )

            identity, status = await asyncio.gather(
                client.get("/v1/identity"),
                client.post("/v1/operations/status", json=ref.__dict__),
            )
            assert identity.status_code == 200
            assert status.json()["state"] == "RUNNING"
            watchdog_ticks = 0

            async def watchdog():
                nonlocal watchdog_ticks
                await asyncio.sleep(0)
                watchdog_ticks += 1

            await watchdog()
            abort = asyncio.create_task(client.post(
                f"/v1/requests/{ref.operation_id}/abort",
                json={"target_operation_ref": ref.__dict__, "reason": "cancel"},
            ))
            await asyncio.sleep(0)
            assert watchdog_ticks == 1
            release.set()
            first_result, duplicate_result, abort_result = await asyncio.gather(
                first, duplicate, abort
            )
            assert executions == 1
            assert first_result.json()["state"] in {"COMPLETED", "FENCED"}
            assert duplicate_result.json()["state"] in {"COMPLETED", "FENCED"}
            assert abort_result.json()["state"] == "FENCED"
    finally:
        release.set()
        owner.close()


@pytest.mark.asyncio
async def test_finalize_status_resources_output_and_prefix_use_owner_queue():
    import threading
    import httpx

    calls = []
    runtime = _runtime()

    def handler(operation, ref, payload):
        calls.append((operation, threading.current_thread().name))
        if operation == "transfer.prepare_receive":
            return OperationSnapshot.running(ref)
        if operation == "status.local":
            return OperationSnapshot(ref, OperationState.COMPLETED)
        if operation == "request.prepare":
            return OperationSnapshot(
                ref, OperationState.COMPLETED, True, ("TARGET_PENDING",)
            )
        if operation == "finalize.local":
            return {kind: 1 for kind in payload["resource_kinds"]}
        if operation == "resources.local":
            return {"resource_quantities": {"SOURCE_RETAIN": 2}}
        if operation == "output.local":
            return {
                "req_id": payload["req_id"],
                "instance_epoch": runtime.registry.instance_epoch,
                "output_seq_no": 1,
                "token_ids": [7],
                "terminal": True,
            }
        if operation == "prefix.local":
            return [] if payload["action"] == "peek" else {"ok": True}
        if operation == "prune.local":
            return None
        raise AssertionError(operation)

    owner = EngineOwnerCommandQueue(handler)
    runtime.async_command_handler = owner.submit_async
    runtime.async_status_handler = lambda ref: owner.submit_async(
        "status.local", ref, {}
    )
    runtime.async_release_handler = lambda operation_id, kinds: owner.submit_local_async(
        "finalize.local", {"operation_id": operation_id, "resource_kinds": kinds}
    )
    runtime.async_resource_details_handler = lambda: owner.submit_local_async(
        "resources.local", {}
    )
    runtime.async_output_handler = lambda req_id, after_seq: owner.submit_local_async(
        "output.local", {"req_id": req_id, "after_seq": after_seq}
    )
    runtime.async_prefix_directory_handler = lambda action, body: owner.submit_local_async(
        "prefix.local", {"action": action, "body": body}
    )
    runtime.async_prune_handler = lambda refs, operation_ids: owner.submit_local_async(
        "prune.local", {"refs": refs, "operation_ids": operation_ids}
    )
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            await client.post(
                "/v1/owners/activate",
                json={"owner_generation": "gateway-a:boot-a"},
            )
            transfer_ref = _ref(1, {})
            await client.post("/v1/transfers/prepare-receive", json={
                "schema_version": 1, "endpoint_ref": transfer_ref.__dict__,
                "payload": {},
            })
            request_payload = {"held_resource_kinds": ["TARGET_PENDING"]}
            request_ref = _ref(2, request_payload)
            await client.post("/v1/requests/prepare", json={
                "schema_version": 1, "endpoint_ref": request_ref.__dict__,
                "payload": request_payload,
            })
            finalize = FinalizeReleaseRequest.build(
                cleanup_id="cleanup-owner", operation_id=request_ref.operation_id,
                lease_id="lease-owner", endpoint_refs=(request_ref,),
                resource_kinds=("TARGET_PENDING",),
            )
            finalize_body = {
                **finalize.__dict__,
                "endpoint_refs": [value.__dict__ for value in finalize.endpoint_refs],
            }
            results = await asyncio.gather(
                client.get(f"/v1/transfers/{transfer_ref.operation_id}"),
                client.get("/v1/resources"),
                client.get("/v1/requests/r/output"),
                client.post("/v1/prefix/reports/register", json={
                    "consumer_id": "c", "generation": "g",
                }),
                client.post("/v1/cleanup/finalize", json=finalize_body),
            )
            assert all(result.status_code == 200 for result in results)
            assert results[0].json()["state"] == "COMPLETED"
            assert results[1].json()["resources"]["SOURCE_RETAIN"] == 2
            assert set(results[1].json()["resources"]) == {
                "SOURCE_RETAIN",
                "SOURCE_PIN",
                "TARGET_PENDING",
                "TARGET_SEQUENCE",
                "SOURCE_BLOCKS",
                "TRANSFER_BYTES",
            }
            assert results[2].json()["token_ids"] == [7]
            assert results[4].json()["resources_held_after"] is False
            replay = await client.post("/v1/cleanup/finalize", json=finalize_body)
            assert replay.json() == results[4].json()
    finally:
        owner.close()

    owner_operations = {
        "status.local", "resources.local", "output.local",
        "prefix.local", "finalize.local", "prune.local",
    }
    assert owner_operations <= {operation for operation, _thread in calls}
    assert all(
        thread == "prism-engine-owner"
        for operation, thread in calls if operation in owner_operations
    )
    assert [operation for operation, _thread in calls].count("finalize.local") == 1
