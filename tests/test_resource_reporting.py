from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from prism_infer.engine.block_manager import BlockManager
from prism_infer.server.app import WorkerControlRuntime, create_app
from prism_infer.server.model_profile import FIXED_QWEN3_0_6B_PROFILE
from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    OperationRegistry,
    OperationSnapshot,
    OperationState,
)
from prism_infer.server.runtime import (
    PDExecutionDriver,
    build_block_resource_snapshot,
)


FORMAL_RESOURCE_QUANTITIES = {
    "SOURCE_RETAIN": 0,
    "SOURCE_PIN": 0,
    "TARGET_PENDING": 0,
    "TARGET_SEQUENCE": 0,
    "SOURCE_BLOCKS": 0,
    "TRANSFER_BYTES": 0,
}


def _allocate_evictable(manager: BlockManager) -> int:
    block_id = manager._allocate_block()
    manager.blocks[block_id].update(123, [1, 2, 3, 4])
    manager.release_block(block_id)
    return block_id


def test_block_resource_snapshot_partitions_five_disjoint_buckets():
    manager = BlockManager(5, 4)
    pending = manager._allocate_block()
    sequence = manager._allocate_block()
    evictable = _allocate_evictable(manager)
    quarantined = manager._allocate_block()

    report = build_block_resource_snapshot(
        manager,
        pending_block_ids={pending},
        quarantined_block_ids={quarantined},
    )

    assert report == {
        "num_gpu_blocks": 5,
        "free_blocks": 1,
        "block_buckets": {
            "free": 1,
            "pending": 1,
            "sequence": 1,
            "evictable": 1,
            "quarantined": 1,
        },
        "block_conservation_valid": True,
    }
    assert sequence not in {pending, evictable, quarantined}


def test_source_pin_is_not_reported_as_evictable_capacity():
    manager = BlockManager(2, 4)
    pinned = _allocate_evictable(manager)

    report = build_block_resource_snapshot(
        manager,
        pinned_block_ids={pinned},
    )

    assert report["block_buckets"] == {
        "free": 1,
        "pending": 0,
        "sequence": 1,
        "evictable": 0,
        "quarantined": 0,
    }


def test_block_resource_snapshot_fails_closed_on_free_block_claim():
    manager = BlockManager(1, 4)

    with pytest.raises(AssertionError, match="non-owned blocks"):
        build_block_resource_snapshot(manager, pending_block_ids={0})


def test_driver_resource_report_includes_profile_total_free_and_conservation():
    manager = BlockManager(4, 4)
    pending = manager._allocate_block()
    manager._allocate_block()
    _allocate_evictable(manager)
    operation = SimpleNamespace(
        operation_id="pending-op",
        resources_held=True,
        sequence=None,
        dst_block_ids=(pending,),
    )
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(block_manager=manager, max_num_seqs=8),
        prefix_cache=SimpleNamespace(_operations={"pending-op": operation}),
    )
    profile = FIXED_QWEN3_0_6B_PROFILE.as_resource_report()
    driver = PDExecutionDriver(engine, role="decode", model_profile=profile)

    report = driver.resource_details()

    assert report["model_profile"] == profile
    assert report["num_gpu_blocks"] == 4
    assert report["free_blocks"] == 1
    assert report["block_buckets"] == {
        "free": 1,
        "pending": 1,
        "sequence": 1,
        "evictable": 1,
        "quarantined": 0,
    }
    assert report["resource_quantities"] == {
        **FORMAL_RESOURCE_QUANTITIES,
        "TARGET_PENDING": 1,
    }
    assert sum(report["block_buckets"].values()) == report["num_gpu_blocks"]


def test_unknown_endpoint_moves_owned_pending_blocks_to_quarantine():
    manager = BlockManager(2, 4)
    pending = manager._allocate_block()
    operation = SimpleNamespace(
        operation_id="pending-op",
        resources_held=True,
        sequence=None,
        dst_block_ids=(pending,),
    )
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(block_manager=manager, max_num_seqs=8),
        prefix_cache=SimpleNamespace(_operations={"pending-op": operation}),
    )
    driver = PDExecutionDriver(engine, role="decode")
    ref = EndpointOperationRef(
        topology_generation="world-a",
        owner_generation="gateway-a:boot-a",
        operation_seq=1,
        target_instance="d0",
        target_worker_epoch="pod-a:boot-a",
        operation_id="pending-op",
        payload_digest="sha256:payload",
    )

    report = driver.resource_details([
        OperationSnapshot(
            ref,
            OperationState.UNKNOWN,
            resources_held=True,
            held_resource_kinds=("TARGET_PENDING",),
        )
    ])

    assert report["block_buckets"] == {
        "free": 1,
        "pending": 0,
        "sequence": 0,
        "evictable": 0,
        "quarantined": 1,
    }


def test_clean_driver_reports_exact_formal_resource_shape_with_zero_defaults():
    manager = BlockManager(2, 4)
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(block_manager=manager, max_num_seqs=8),
        prefix_cache=SimpleNamespace(_operations={}),
    )

    report = PDExecutionDriver(engine, role="decode").resource_details()

    assert report["resource_quantities"] == FORMAL_RESOURCE_QUANTITIES


def test_clean_worker_resources_endpoint_reports_exact_six_zero_keys():
    registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
    )
    runtime = WorkerControlRuntime(
        identity={"instance_id": "d0"},
        capabilities={"ready": True},
        registry=registry,
    )

    response = TestClient(create_app(runtime)).get("/v1/resources")

    assert response.status_code == 200
    assert response.json()["resources"] == FORMAL_RESOURCE_QUANTITIES


def test_worker_resources_endpoint_preserves_known_held_counts_and_zero_fills():
    registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
    )
    registry.activate_owner("gateway-a:boot-a")
    ref = EndpointOperationRef(
        topology_generation="world-a",
        owner_generation="gateway-a:boot-a",
        operation_seq=1,
        target_instance="d0",
        target_worker_epoch="pod-d0:boot-a",
        operation_id="pending-op",
        payload_digest="sha256:pending",
    )
    registry.accept(
        ref,
        lambda: OperationSnapshot.running(
            ref, held_resource_kinds=("TARGET_PENDING",)
        ),
    )
    runtime = WorkerControlRuntime(
        identity={"instance_id": "d0"},
        capabilities={"ready": True},
        registry=registry,
        resource_details_handler=lambda: {
            "resource_quantities": {
                "SOURCE_RETAIN": 2,
                "TRANSFER_BYTES": 64,
            }
        },
    )

    response = TestClient(create_app(runtime)).get("/v1/resources")

    assert response.status_code == 200
    assert response.json()["resources"] == {
        "SOURCE_RETAIN": 2,
        "SOURCE_PIN": 0,
        "TARGET_PENDING": 1,
        "TARGET_SEQUENCE": 0,
        "SOURCE_BLOCKS": 0,
        "TRANSFER_BYTES": 64,
    }


@pytest.mark.parametrize("unknown_source", ["registry", "details"])
def test_worker_resource_report_fails_closed_on_unknown_kind(unknown_source):
    registry = OperationRegistry(
        instance_id="d0",
        instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
    )
    registry.activate_owner("gateway-a:boot-a")
    details = None
    if unknown_source == "registry":
        ref = EndpointOperationRef(
            topology_generation="world-a",
            owner_generation="gateway-a:boot-a",
            operation_seq=1,
            target_instance="d0",
            target_worker_epoch="pod-d0:boot-a",
            operation_id="unknown-op",
            payload_digest="sha256:unknown",
        )
        registry.accept(
            ref,
            lambda: OperationSnapshot.running(
                ref, held_resource_kinds=("UNKNOWN_KIND",)
            ),
        )
    else:
        details = lambda: {"resource_quantities": {"UNKNOWN_KIND": 1}}
    runtime = WorkerControlRuntime(
        identity={"instance_id": "d0"},
        capabilities={"ready": True},
        registry=registry,
        resource_details_handler=details,
    )

    with pytest.raises(ValueError, match="unknown formal resource kind"):
        runtime.resources()
