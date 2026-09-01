import asyncio
import hashlib
import json
from pathlib import Path
import threading
import tomllib
from types import SimpleNamespace

import pytest

import prism_infer.server.startup_lifecycle as startup_lifecycle
import prism_infer.server.worker as worker_module
from prism_infer.server.runtime import EngineOwnerCommandQueue
from prism_infer.server.startup_lifecycle import (
    PodIncarnationLifecycle,
    StartupPermitWaitCancelled,
    validate_startup_permit,
    wait_for_startup_permit,
)
from prism_infer.server.worker import (
    NCCL_WATCHDOG_EXIT_CODE,
    _attested_pair_transport,
    _capability_ready,
    _capture_pair_probe,
    _observed_nccl_transport,
    _supervise_watchdog,
)


def _startup_permit(*, generation="world-a", pod_suffix="a"):
    permit = {
        "schema_version": "prism.week12.worker-startup-permit/v1",
        "issuance_mode": "INIT",
        "permit_id": f"permit-{generation}",
        "topology_generation": generation,
        "members": {
            name: f"pod-{name}-{pod_suffix}"
            for name in ("p0", "p1", "d0", "d1")
        },
    }
    permit["canonical_digest"] = "sha256:" + hashlib.sha256(
        json.dumps(
            permit, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return permit


def _lifecycle(tmp_path, *, process_generation="boot-d1"):
    return PodIncarnationLifecycle(
        tmp_path / "incarnation" / "record.json",
        instance_id="d1",
        pod_uid="pod-d1-a",
        topology_generation="world-a",
        process_generation=process_generation,
    )


def test_startup_permit_requires_exact_four_pod_uid_digest():
    permit = _startup_permit()

    accepted = validate_startup_permit(
        permit,
        instance_id="d1",
        pod_uid="pod-d1-a",
        topology_generation="world-a",
    )

    assert accepted == permit


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(alias="not-allowed"),
            "fields are not exact",
        ),
        (
            lambda value: value["members"].pop("p1"),
            "exact four members",
        ),
        (
            lambda value: value["members"].update(
                p1=value["members"]["p0"]
            ),
            "must be unique",
        ),
        (
            lambda value: value.update(topology_generation="world-old"),
            "topology_generation mismatch",
        ),
        (
            lambda value: value["members"].update(d1="replacement-pod"),
            "does not authorize this Pod UID",
        ),
        (
            lambda value: value.update(
                canonical_digest="sha256:" + "f" * 64
            ),
            "canonical_digest mismatch",
        ),
    ],
)
def test_startup_permit_rejects_partial_stale_or_tampered_snapshot(
    mutate, message,
):
    permit = _startup_permit()
    mutate(permit)

    with pytest.raises(ValueError, match=message):
        validate_startup_permit(
            permit,
            instance_id="d1",
            pod_uid="pod-d1-a",
            topology_generation="world-a",
        )


def test_startup_permit_wait_is_cancellable_without_runtime_init(tmp_path):
    path = tmp_path / "startup-permit.json"
    path.write_text("{}", encoding="utf-8")
    polls = []

    with pytest.raises(StartupPermitWaitCancelled):
        wait_for_startup_permit(
            path,
            instance_id="d1",
            pod_uid="pod-d1-a",
            topology_generation="world-a",
            poll_interval_s=0.25,
            cancelled=lambda: len(polls) == 1,
            sleep=lambda interval: polls.append(interval),
        )

    assert polls == [0.25]


def test_first_incarnation_create_once_blocks_same_pod_restart(tmp_path):
    permit = _startup_permit()
    first = _lifecycle(tmp_path)
    replacement = _lifecycle(
        tmp_path, process_generation="boot-replacement"
    )

    assert first.create_active(permit) is True
    assert replacement.create_active(permit) is False

    record = first.read()
    assert record["state"] == "ACTIVE"
    assert record["process_generation"] == "boot-d1"


def test_active_record_torn_fsync_never_grants_second_init(
    tmp_path, monkeypatch,
):
    permit = _startup_permit()
    first = _lifecycle(tmp_path)
    replacement = _lifecycle(
        tmp_path, process_generation="boot-replacement"
    )
    monkeypatch.setattr(
        startup_lifecycle,
        "_fsync_directory",
        lambda path: (_ for _ in ()).throw(OSError("directory fsync failed")),
    )

    with pytest.raises(OSError, match="directory fsync failed"):
        first.create_active(permit)

    assert replacement.create_active(permit) is False


def test_fail_stop_latch_is_exact_ref_idempotent_and_conflict_fenced(tmp_path):
    lifecycle = _lifecycle(tmp_path)
    assert lifecycle.create_active(_startup_permit()) is True
    evidence = {
        "pair_id": "p0--d1",
        "operation_id": "transfer-7",
        "endpoint_ref": {
            "topology_generation": "world-a",
            "owner_generation": "gateway-a:boot-a",
            "operation_seq": 7,
            "target_instance": "d1",
            "target_worker_epoch": "pod-d1-a:boot-d1",
            "operation_id": "transfer-7",
            "payload_digest": "sha256:payload",
        },
        "reason": "NCCL operation watchdog expired: exact-ref",
        "watchdog_timeout_s": 30.0,
    }
    invalid_zero_seq = {
        **evidence,
        "endpoint_ref": {
            **evidence["endpoint_ref"],
            "operation_seq": 0,
        },
    }

    with pytest.raises(ValueError, match="operation_seq is invalid"):
        lifecycle.latch_fail_stop(invalid_zero_seq)
    assert lifecycle.read()["state"] == "ACTIVE"

    first = lifecycle.latch_fail_stop(evidence)
    replay = lifecycle.latch_fail_stop(evidence)

    assert first == replay
    assert replay["state"] == "FAIL_STOP"
    assert replay["watchdog"]["endpoint_ref"] == evidence["endpoint_ref"]
    conflict = {
        **evidence,
        "operation_id": "transfer-other",
        "endpoint_ref": {
            **evidence["endpoint_ref"],
            "operation_id": "transfer-other",
        },
    }
    with pytest.raises(ValueError, match="conflicts"):
        lifecycle.latch_fail_stop(conflict)


def test_worker_runtime_and_image_pin_httpx_dependency():
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert any(value.startswith("httpx>=") for value in dependencies)
    dockerfile = (root / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert '"httpx==0.28.1"' in dockerfile
    assert "import flash_attn, httpx, torch, triton" in dockerfile


def test_gdr_requirement_does_not_upgrade_socket_only_evidence(tmp_path):
    evidence = tmp_path / "nccl.log"
    evidence.write_text(
        "NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1\n",
        encoding="utf-8",
    )

    transport, details = _observed_nccl_transport(str(evidence))
    pair = {"probe_passed": True, "transport": transport}

    assert transport == "NCCL_SOCKET"
    assert details["gdr_marker_observed"] is False
    assert _capability_ready([pair], require_gdr=False) is True
    assert _capability_ready([pair], require_gdr=True) is False


def test_positive_nccl_gdr_marker_is_required(tmp_path):
    evidence = tmp_path / "nccl.log"
    evidence.write_text(
        "NCCL INFO NET/IB/GDRDMA : GPU Direct RDMA Enabled for GPU 0\n",
        encoding="utf-8",
    )

    transport, details = _observed_nccl_transport(str(evidence))

    assert transport == "NCCL_GDR"
    assert details["gdr_marker_observed"] is True
    assert _capability_ready(
        [{"probe_passed": True, "transport": transport}], require_gdr=True
    ) is True


def test_missing_or_unrecognized_nccl_evidence_is_unknown_and_not_ready(tmp_path):
    missing_transport, _ = _observed_nccl_transport(None)
    unrecognized = tmp_path / "nccl.log"
    unrecognized.write_text("NCCL INFO initialized\n", encoding="utf-8")
    unknown_transport, _ = _observed_nccl_transport(str(unrecognized))

    assert missing_transport == unknown_transport == "UNKNOWN"
    assert _capability_ready(
        [{"probe_passed": True, "transport": "UNKNOWN"}], require_gdr=False
    ) is False


def test_negative_gdr_markers_are_never_upgraded_by_gdr_substring(tmp_path):
    for index, marker in enumerate((
        "NCCL INFO NET/IB : GDR Disabled",
        "NCCL INFO GPU Direct RDMA off",
        "NCCL INFO NET/IB/GDRDMA : not enabled",
        "NCCL INFO NET/IB : GDR=0",
    )):
        evidence = tmp_path / f"negative-{index}.log"
        evidence.write_text(marker + "\n", encoding="utf-8")
        transport, details = _observed_nccl_transport(str(evidence))
        assert transport == "UNKNOWN"
        assert details["gdr_marker_observed"] is False
        assert "negative_marker" in details


def test_conflicting_positive_and_negative_gdr_evidence_is_unknown(tmp_path):
    evidence = tmp_path / "disagreement.log"
    evidence.write_text(
        "NCCL INFO GPU Direct RDMA Enabled for GPU 0\n"
        "NCCL INFO NET/IB : GDR Disabled for GPU 1\n",
        encoding="utf-8",
    )

    transport, details = _observed_nccl_transport(str(evidence))

    assert transport == "UNKNOWN"
    assert "conflicting" in details["reason"]


def test_negative_gdr_with_explicit_socket_path_classifies_socket(tmp_path):
    evidence = tmp_path / "socket-fallback.log"
    evidence.write_text(
        "NCCL INFO NET/IB : GDR Disabled\n"
        "NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1\n",
        encoding="utf-8",
    )

    transport, details = _observed_nccl_transport(str(evidence))

    assert transport == "NCCL_SOCKET"
    assert details["gdr_marker_observed"] is False


def test_unreadable_nccl_evidence_is_unknown(tmp_path):
    transport, details = _observed_nccl_transport(str(tmp_path / "missing.log"))
    assert transport == "UNKNOWN"
    assert "unavailable" in details["reason"]


@pytest.mark.asyncio
async def test_owner_progress_is_serialized_behind_prune_on_owner_queue():
    prune_entered = threading.Event()
    release_prune = threading.Event()
    calls = []

    def handler(operation, ref, payload):
        calls.append((operation, threading.current_thread().name))
        if operation == "prune.local":
            prune_entered.set()
            assert release_prune.wait(2)
            return None
        if operation == "progress.local":
            return {"expired": False, "reason": ""}
        raise AssertionError(operation)

    owner = EngineOwnerCommandQueue(handler)
    try:
        prune = asyncio.create_task(owner.submit_local_async("prune.local", {}))
        assert await asyncio.to_thread(prune_entered.wait, 1)
        progress = asyncio.create_task(
            owner.submit_local_async("progress.local", {})
        )
        await asyncio.sleep(0.02)
        assert [item[0] for item in calls] == ["prune.local"]
        release_prune.set()
        await asyncio.gather(prune, progress)
    finally:
        release_prune.set()
        owner.close()
    assert [item[0] for item in calls] == ["prune.local", "progress.local"]
    assert all(item[1] == "prism-engine-owner" for item in calls)


@pytest.mark.asyncio
async def test_watchdog_writes_exact_kubernetes_termination_message_before_exit(
    tmp_path,
):
    endpoint_ref = {
        "topology_generation": "world-a",
        "owner_generation": "gateway-a:boot-a",
        "operation_seq": 7,
        "target_instance": "d1",
        "target_worker_epoch": "pod-d1-a:boot-d1",
        "operation_id": "transfer-7",
        "payload_digest": "sha256:payload",
    }

    watchdog = SimpleNamespace(
        poll_watchdog_deadline=lambda: True,
        watchdog_evidence={
            "kind": "nccl_watchdog_timeout",
            "reason": "NCCL operation watchdog expired: exact-ref",
            "pair_id": "p0--d1",
            "endpoint_ref": endpoint_ref,
            "operation_id": "transfer-7",
            "watchdog_timeout_s": 30.0,
        },
    )
    runtime = SimpleNamespace(
        identity={
            "instance_id": "d1",
            "pod_uid": "pod-d1-a",
            "process_generation": "boot-d1",
            "instance_epoch": "pod-d1-a:boot-d1",
            "topology_generation": "world-a",
        },
        capabilities={"ready": True},
    )
    path = tmp_path / "termination-log"
    lifecycle = _lifecycle(tmp_path)
    assert lifecycle.create_active(_startup_permit()) is True
    exit_codes = []

    class Server:
        should_exit = False

    server = Server()

    def exact_exit(code):
        assert lifecycle.read()["state"] == "FAIL_STOP"
        assert path.exists(), "termination message must precede process exit"
        exit_codes.append(code)

    await _supervise_watchdog(
        runtime,
        watchdog,
        server,
        lifecycle=lifecycle,
        poll_interval_s=0.0,
        termination_message_path=path,
        exit_process=exact_exit,
    )

    message = json.loads(path.read_text(encoding="utf-8"))
    assert message["kind"] == "nccl_watchdog_timeout"
    assert message["pair_id"] == "p0--d1"
    assert message["endpoint_ref"] == endpoint_ref
    assert message["operation_id"] == "transfer-7"
    assert message["reason"].startswith("NCCL operation watchdog expired")
    assert message["expected_exit_code"] == NCCL_WATCHDOG_EXIT_CODE
    assert runtime.capabilities["worker_exit_code"] == NCCL_WATCHDOG_EXIT_CODE
    assert runtime.capabilities["ready"] is False
    assert exit_codes == [NCCL_WATCHDOG_EXIT_CODE]
    assert server.should_exit is False


@pytest.mark.asyncio
async def test_watchdog_durable_write_failure_retries_before_exit(
    tmp_path, monkeypatch,
):
    endpoint_ref = {
        "topology_generation": "world-a",
        "owner_generation": "gateway-a:boot-a",
        "operation_seq": 7,
        "target_instance": "d1",
        "target_worker_epoch": "pod-d1-a:boot-d1",
        "operation_id": "transfer-7",
        "payload_digest": "sha256:payload",
    }

    watchdog = SimpleNamespace(
        poll_watchdog_deadline=lambda: True,
        watchdog_evidence={
            "kind": "nccl_watchdog_timeout",
            "reason": "NCCL operation watchdog expired: exact-ref",
            "pair_id": "p0--d1",
            "endpoint_ref": endpoint_ref,
            "operation_id": "transfer-7",
            "watchdog_timeout_s": 30.0,
        },
    )

    lifecycle = _lifecycle(tmp_path)
    assert lifecycle.create_active(_startup_permit()) is True
    runtime = SimpleNamespace(
        identity={
            "instance_id": "d1",
            "pod_uid": "pod-d1-a",
            "process_generation": "boot-d1",
            "instance_epoch": "pod-d1-a:boot-d1",
            "topology_generation": "world-a",
        },
        capabilities={"ready": True},
    )
    path = tmp_path / "termination-log"
    writes = []
    original = worker_module._write_termination_message

    def fail_once(message_path, message):
        writes.append(lifecycle.read()["state"])
        if len(writes) == 1:
            raise OSError("termination write failed")
        original(message_path, message)

    monkeypatch.setattr(
        worker_module, "_write_termination_message", fail_once
    )
    exit_codes = []
    await _supervise_watchdog(
        runtime,
        watchdog,
        SimpleNamespace(should_exit=False),
        lifecycle=lifecycle,
        poll_interval_s=0.0,
        termination_message_path=path,
        exit_process=exit_codes.append,
    )

    assert writes == ["FAIL_STOP", "FAIL_STOP"]
    assert exit_codes == [NCCL_WATCHDOG_EXIT_CODE]
    assert path.exists()


@pytest.mark.asyncio
async def test_watchdog_deadline_exception_stays_not_ready_without_false_exit():
    def poll_deadline():
        raise RuntimeError("poll failed")

    watchdog = SimpleNamespace(
        poll_watchdog_deadline=poll_deadline,
        watchdog_evidence=None,
    )
    runtime = SimpleNamespace(capabilities={"ready": True})
    server = SimpleNamespace(should_exit=False)
    exit_codes = []
    await _supervise_watchdog(
        runtime,
        watchdog,
        server,
        lifecycle=SimpleNamespace(),
        poll_interval_s=0.0,
        exit_process=exit_codes.append,
    )
    assert runtime.capabilities["ready"] is False
    assert "poll failed" in runtime.capabilities["failure_reason"]
    assert "worker_exit_code" not in runtime.capabilities
    assert exit_codes == []
    assert server.should_exit is False


def test_pair_scoped_transport_ignores_other_pair_markers(tmp_path):
    evidence = tmp_path / "mixed-pairs.log"
    evidence.write_text(
        "p0--d0 NCCL INFO GPU Direct RDMA Enabled for GPU 0\n"
        "p1--d1 NCCL INFO NET/Socket : Using network Socket\n",
        encoding="utf-8",
    )

    gdr, gdr_details = _observed_nccl_transport(
        str(evidence), pair_id="p0--d0"
    )
    socket, socket_details = _observed_nccl_transport(
        str(evidence), pair_id="p1--d1"
    )

    assert gdr == "NCCL_GDR"
    assert socket == "NCCL_SOCKET"
    assert gdr_details["slice_line_count"] == 1
    assert socket_details["slice_line_count"] == 1
    assert gdr_details["slice_sha256"] != socket_details["slice_sha256"]


def test_unattributed_positive_gdr_marker_is_unknown_for_pair(tmp_path):
    evidence = tmp_path / "unattributed.log"
    evidence.write_text(
        "NCCL INFO GPU Direct RDMA Enabled for GPU 0\n", encoding="utf-8"
    )

    transport, details = _observed_nccl_transport(
        str(evidence), pair_id="p0--d0"
    )

    assert transport == "UNKNOWN"
    assert "exact pair" in details["reason"]


def test_pair_scoped_gdr_and_socket_disagreement_is_unknown(tmp_path):
    evidence = tmp_path / "pair-disagreement.log"
    evidence.write_text(
        "p0--d0 NCCL INFO GPU Direct RDMA Enabled\n"
        "p0--d0 NCCL INFO NET/Socket : Using network Socket\n",
        encoding="utf-8",
    )

    transport, details = _observed_nccl_transport(
        str(evidence), pair_id="p0--d0"
    )

    assert transport == "UNKNOWN"
    assert "GDR and socket" in details["reason"]


def _capture_test_probe(tmp_path, appended):
    log = tmp_path / "nccl-debug.log"
    prior = "NCCL INFO NET/Socket : Using network Socket\n"
    log.write_text(prior, encoding="utf-8")

    def probe():
        with log.open("a", encoding="utf-8") as stream:
            stream.write(appended)

    attestation = _capture_pair_probe(
        pair_id="p0--d0",
        global_ranks=(0, 2),
        global_rank=0,
        instance_id="p0",
        instance_epoch="pod-p0:boot-a",
        topology_generation="world-a",
        evidence_path=str(log),
        probe=probe,
    )
    return prior, attestation


def test_sequential_probe_binds_unlabelled_real_nccl_log_to_pair_and_epochs(tmp_path):
    prior, attestation = _capture_test_probe(
        tmp_path,
        "host:1:1 NCCL INFO NET/IB/GDRDMA : GPU Direct RDMA Enabled for GPU 0\n",
    )

    transport, details = _attested_pair_transport(
        attestation,
        pair_id="p0--d0",
        attester_instance="p0",
        attester_instance_epoch="pod-p0:boot-a",
        source_epoch="pod-p0:boot-a",
        target_epoch="pod-d0:boot-b",
        topology_generation="world-a",
    )

    assert attestation["log_offset_start"] == len(prior.encode())
    assert transport == "NCCL_GDR"
    assert details["slice_line_count"] == 1
    assert details["binding"] == {
        "pair_id": "p0--d0",
        "attester_instance": "p0",
        "attester_instance_epoch": "pod-p0:boot-a",
        "source_epoch": "pod-p0:boot-a",
        "target_epoch": "pod-d0:boot-b",
        "topology_generation": "world-a",
    }
    assert details["attestation_sha256"].startswith("sha256:")
    assert _capability_ready(
        [{"probe_passed": True, "transport": transport}], require_gdr=True
    ) is True


def test_sequential_probe_mixed_transport_slice_fails_unknown(tmp_path):
    _, attestation = _capture_test_probe(
        tmp_path,
        "NCCL INFO NET/IB/GDRDMA : GPU Direct RDMA Enabled for GPU 0\n"
        "NCCL INFO NET/Socket : Using network Socket\n",
    )

    transport, details = _attested_pair_transport(
        attestation,
        pair_id="p0--d0",
        attester_instance="p0",
        attester_instance_epoch="pod-p0:boot-a",
        source_epoch="pod-p0:boot-a",
        target_epoch="pod-d0:boot-b",
        topology_generation="world-a",
    )

    assert transport == "UNKNOWN"
    assert "GDR and socket" in details["reason"]


def test_unattributed_marker_outside_probe_slice_cannot_make_pair_ready(tmp_path):
    _, attestation = _capture_test_probe(
        tmp_path, "NCCL INFO Channel 00 initialized\n"
    )

    transport, details = _attested_pair_transport(
        attestation,
        pair_id="p0--d0",
        attester_instance="p0",
        attester_instance_epoch="pod-p0:boot-a",
        source_epoch="pod-p0:boot-a",
        target_epoch="pod-d0:boot-b",
        topology_generation="world-a",
    )

    assert transport == "UNKNOWN"
    assert "no positive NCCL transport marker" in details["reason"]
    assert _capability_ready(
        [{"probe_passed": True, "transport": transport}], require_gdr=False
    ) is False
