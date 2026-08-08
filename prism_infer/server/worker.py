"""Experimental 2P2D worker process entrypoint."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


NCCL_WATCHDOG_EXIT_CODE = 70
KUBERNETES_TERMINATION_MESSAGE_MAX_BYTES = 4096


def _write_termination_message(
    path: str | Path, evidence: dict[str, object]
) -> None:
    """Persist exact failure evidence for kubelet before process exit."""
    data = json.dumps(
        evidence, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(data) > KUBERNETES_TERMINATION_MESSAGE_MAX_BYTES:
        raise ValueError("Kubernetes termination message exceeds 4096 bytes")
    with Path(path).open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required worker environment: {name}")
    return value


def _observed_nccl_transport(
    evidence_path: str | None, *, pair_id: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Classify only transport actually reported by NCCL debug evidence.

    A deployment requirement is never evidence.  Missing, unreadable, or
    Positive socket logs classify as NCCL_SOCKET.  Missing, unreadable, or
    unrecognized evidence remains UNKNOWN and cannot make readiness true.
    """
    evidence = {
        "source": evidence_path or "",
        "gdr_marker_observed": False,
        "pair_id": pair_id or "",
    }
    if not evidence_path:
        evidence["reason"] = "NCCL debug evidence path is not configured"
        return "UNKNOWN", evidence
    try:
        text = Path(evidence_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        evidence["reason"] = f"NCCL debug evidence unavailable: {exc}"
        return "UNKNOWN", evidence
    lines = text.splitlines()
    if pair_id is not None:
        try:
            source, target = pair_id.split("--", 1)
        except ValueError:
            evidence["reason"] = "invalid pair id"
            return "UNKNOWN", evidence
        pair_pattern = re.compile(rf"(?<![\w-]){re.escape(pair_id)}(?![\w-])")
        source_pattern = re.compile(rf"(?<![\w-]){re.escape(source)}(?![\w-])")
        target_pattern = re.compile(rf"(?<![\w-]){re.escape(target)}(?![\w-])")
        lines = [
            line for line in lines
            if pair_pattern.search(line)
            or (source_pattern.search(line) and target_pattern.search(line))
        ]
        if not lines:
            evidence["reason"] = "no NCCL marker attributed to exact pair"
            return "UNKNOWN", evidence
    slice_text = "\n".join(lines)
    evidence["slice_sha256"] = "sha256:" + hashlib.sha256(
        slice_text.encode()
    ).hexdigest()
    evidence["slice_line_count"] = len(lines)

    negative_pattern = re.compile(
        r"(?:\b(?:GDR|GDRDMA|GPU\s+Direct\s+RDMA)\b[^\n]*"
        r"(?:disabled?|\boff\b|not\s+enabled|(?:=|:)\s*0\b)"
        r"|(?:disabled?|\boff\b|not\s+enabled)[^\n]*"
        r"\b(?:GDR|GDRDMA|GPU\s+Direct\s+RDMA)\b)",
        flags=re.IGNORECASE,
    )
    positive_pattern = re.compile(
        r"(?:GPU\s+Direct\s+RDMA[^\n]*\b(?:Enabled|Active)\b"
        r"|\bGDRDMA\b[^\n]*\b(?:Enabled|Active)\b"
        r"|\b(?:Using|Selected|via)\b[^\n]*\bGDRDMA\b"
        r"|NET/IB[^\n]*\bGDR\b\s*(?:=|:)\s*1\b)",
        flags=re.IGNORECASE,
    )
    negative_lines = []
    positive_lines = []
    socket_lines = []
    for line in lines:
        negative = negative_pattern.search(line)
        if negative is not None:
            negative_lines.append(negative.group(0))
            continue
        positive = positive_pattern.search(line)
        if positive is not None:
            positive_lines.append(positive.group(0))
        socket = re.search(
            r"(?:NET/Socket|Using network Socket)", line,
            flags=re.IGNORECASE,
        )
        if socket is not None:
            socket_lines.append(socket.group(0))
    if negative_lines and positive_lines:
        evidence["negative_marker"] = negative_lines[0]
        evidence["positive_marker"] = positive_lines[0]
        evidence["reason"] = "conflicting positive and negative GDR markers"
        return "UNKNOWN", evidence
    if positive_lines and socket_lines:
        evidence["positive_marker"] = positive_lines[0]
        evidence["socket_marker"] = socket_lines[0]
        evidence["reason"] = "conflicting GDR and socket markers"
        return "UNKNOWN", evidence
    if not positive_lines:
        if socket_lines:
            evidence["marker"] = socket_lines[0]
            if negative_lines:
                evidence["negative_marker"] = negative_lines[0]
            return "NCCL_SOCKET", evidence
        if negative_lines:
            evidence["negative_marker"] = negative_lines[0]
            evidence["reason"] = "explicit negative GDR marker observed"
            return "UNKNOWN", evidence
        evidence["reason"] = "no positive NCCL transport marker observed"
        return "UNKNOWN", evidence
    evidence["gdr_marker_observed"] = True
    evidence["marker"] = positive_lines[0]
    return "NCCL_GDR", evidence


def _capture_pair_probe(
    *,
    pair_id: str,
    global_ranks: tuple[int, int],
    global_rank: int,
    instance_id: str,
    instance_epoch: str,
    topology_generation: str,
    evidence_path: str | None,
    probe,
) -> dict[str, object]:
    """Bind one sequential pair warmup to the bytes appended during its probe."""
    if global_rank not in global_ranks:
        raise ValueError(f"global rank {global_rank} is not a member of {pair_id}")
    started_ns = time.time_ns()
    path = Path(evidence_path) if evidence_path else None
    try:
        offset_start = path.stat().st_size if path is not None else 0
    except OSError:
        offset_start = 0
    probe()
    finished_ns = time.time_ns()

    raw_slice = b""
    offset_end = offset_start
    capture_error = ""
    if path is None:
        capture_error = "NCCL debug evidence path is not configured"
    else:
        try:
            raw = path.read_bytes()
            offset_end = len(raw)
            if offset_end < offset_start:
                capture_error = "NCCL debug log truncated during pair probe"
            else:
                raw_slice = raw[offset_start:offset_end]
        except OSError as exc:
            capture_error = f"NCCL debug evidence unavailable: {exc}"
    slice_path = ""
    if path is not None:
        slice_file = path.with_name(
            f"{path.name}.{pair_id}.rank{global_rank}.probe.log"
        )
        try:
            slice_file.write_bytes(raw_slice)
            slice_path = str(slice_file)
        except OSError as exc:
            capture_error = f"NCCL probe slice write failed: {exc}"
    attestation = {
        "pair_id": pair_id,
        "global_ranks": list(global_ranks),
        "attester_global_rank": global_rank,
        "attester_instance": instance_id,
        "attester_instance_epoch": instance_epoch,
        "topology_generation": topology_generation,
        "probe_started_unix_ns": started_ns,
        "probe_finished_unix_ns": finished_ns,
        "log_path": str(path) if path is not None else "",
        "log_offset_start": offset_start,
        "log_offset_end": offset_end,
        "log_slice_path": slice_path,
        "log_slice_sha256": "sha256:" + hashlib.sha256(raw_slice).hexdigest(),
        "probe_completed": True,
    }
    if capture_error:
        attestation["capture_error"] = capture_error
    return attestation


def _attested_pair_transport(
    attestation: dict[str, object] | None,
    *,
    pair_id: str,
    attester_instance: str,
    attester_instance_epoch: str,
    source_epoch: str,
    target_epoch: str,
    topology_generation: str,
) -> tuple[str, dict[str, object]]:
    """Verify the pair/epoch/generation binding before classifying raw NCCL bytes."""
    binding = {
        "pair_id": pair_id,
        "attester_instance": attester_instance,
        "attester_instance_epoch": attester_instance_epoch,
        "source_epoch": source_epoch,
        "target_epoch": target_epoch,
        "topology_generation": topology_generation,
    }
    evidence: dict[str, object] = {"binding": binding}
    if attestation is None:
        evidence["reason"] = "pair probe attestation is missing"
        return "UNKNOWN", evidence
    expected = {
        "pair_id": pair_id,
        "attester_instance": attester_instance,
        "attester_instance_epoch": attester_instance_epoch,
        "topology_generation": topology_generation,
    }
    for field, value in expected.items():
        if attestation.get(field) != value:
            evidence["reason"] = f"pair probe attestation {field} mismatch"
            return "UNKNOWN", evidence
    if attestation.get("probe_completed") is not True:
        evidence["reason"] = "pair probe did not complete"
        return "UNKNOWN", evidence
    if attestation.get("capture_error"):
        evidence["reason"] = str(attestation["capture_error"])
        return "UNKNOWN", evidence
    try:
        offset_start = int(attestation["log_offset_start"])
        offset_end = int(attestation["log_offset_end"])
        slice_path = str(attestation["log_slice_path"])
        raw_slice = Path(slice_path).read_bytes()
    except (KeyError, TypeError, ValueError, OSError) as exc:
        evidence["reason"] = f"pair probe slice unavailable: {exc}"
        return "UNKNOWN", evidence
    if offset_start < 0 or offset_end <= offset_start:
        evidence["reason"] = "pair probe produced no attributable NCCL log bytes"
        return "UNKNOWN", evidence
    if len(raw_slice) != offset_end - offset_start:
        evidence["reason"] = "pair probe slice length does not match log offsets"
        return "UNKNOWN", evidence
    digest = "sha256:" + hashlib.sha256(raw_slice).hexdigest()
    if digest != attestation.get("log_slice_sha256"):
        evidence["reason"] = "pair probe slice digest mismatch"
        return "UNKNOWN", evidence
    transport, classified = _observed_nccl_transport(slice_path)
    evidence.update(classified)
    evidence["log_offset_start"] = offset_start
    evidence["log_offset_end"] = offset_end
    evidence["log_slice_sha256"] = digest
    evidence["probe_started_unix_ns"] = attestation.get("probe_started_unix_ns")
    evidence["probe_finished_unix_ns"] = attestation.get("probe_finished_unix_ns")
    evidence["attestation_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            {"binding": binding, "attestation": attestation},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return transport, evidence


def _capability_ready(
    pairs: list[dict[str, object]], *, require_gdr: bool
) -> bool:
    return bool(pairs) and all(
        pair.get("probe_passed") is True
        and pair.get("transport") in {"NCCL_GDR", "NCCL_SOCKET", "CUDA_IPC"}
        and (not require_gdr or pair.get("transport") == "NCCL_GDR")
        for pair in pairs
    )


async def _supervise_watchdog(
    runtime,
    watchdog,
    server,
    *,
    lifecycle,
    poll_interval_s: float = 0.05,
    termination_message_path: str | Path | None = None,
    exit_process=None,
) -> None:
    """Latch Pod fail-stop evidence before the watchdog performs exact exit 70."""
    if exit_process is None:
        exit_process = os._exit
    try:
        while True:
            # Deadline detection must not queue behind the operation it
            # monitors.  This call reads only immutable deadline records and
            # never invokes Work, CUDA, or engine-owner state transitions.
            if watchdog.poll_watchdog_deadline():
                runtime.capabilities["ready"] = False
                endpoint = watchdog.watchdog_evidence
                if not isinstance(endpoint, dict):
                    raise RuntimeError("NCCL watchdog endpoint evidence is missing")
                ref = endpoint.get("endpoint_ref")
                if not isinstance(ref, dict):
                    raise RuntimeError("NCCL watchdog endpoint ref is missing")
                identity = dict(runtime.identity)
                if (
                    endpoint.get("kind") != "nccl_watchdog_timeout"
                    or ref.get("target_instance") != identity.get("instance_id")
                    or ref.get("target_worker_epoch") != identity.get("instance_epoch")
                    or ref.get("topology_generation")
                    != identity.get("topology_generation")
                ):
                    raise RuntimeError("NCCL watchdog evidence identity mismatch")
                message = {
                    "schema_version": 1,
                    "kind": "nccl_watchdog_timeout",
                    "instance_id": identity["instance_id"],
                    "pod_uid": identity["pod_uid"],
                    "process_generation": identity["process_generation"],
                    "instance_epoch": identity["instance_epoch"],
                    "topology_generation": identity["topology_generation"],
                    "pair_id": endpoint["pair_id"],
                    "operation_id": endpoint["operation_id"],
                    "endpoint_ref": ref,
                    "reason": endpoint["reason"],
                    "watchdog_timeout_s": endpoint["watchdog_timeout_s"],
                    "expected_exit_code": NCCL_WATCHDOG_EXIT_CODE,
                }
                runtime.capabilities["failure_kind"] = "nccl_watchdog_timeout"
                runtime.capabilities["failure_reason"] = str(message["reason"])
                runtime.capabilities["worker_exit_code"] = NCCL_WATCHDOG_EXIT_CODE
                while True:
                    try:
                        lifecycle.latch_fail_stop(message)
                        _write_termination_message(
                            termination_message_path
                            or os.environ.get(
                                "PRISM_TERMINATION_LOG_PATH",
                                "/dev/termination-log",
                            ),
                            message,
                        )
                        break
                    except asyncio.CancelledError:
                        raise
                    except (OSError, ValueError) as exc:
                        runtime.capabilities["failure_reason"] = (
                            "NCCL watchdog durable evidence retry: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        await asyncio.sleep(poll_interval_s)
                # os._exit skips teardown; only a whole-world restart can remove
                # the stale communicator. Tests may inject a returning exit hook.
                exit_process(NCCL_WATCHDOG_EXIT_CODE)
                return
            await asyncio.sleep(poll_interval_s)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        runtime.capabilities["ready"] = False
        runtime.capabilities["failure_reason"] = (
            f"NCCL watchdog supervisor failed: {type(exc).__name__}: {exc}"
        )
        runtime.capabilities["failure_kind"] = "watchdog_supervisor_failed"
        # Without an exact watchdog ref, remain NotReady instead of fabricating
        # fail-stop evidence.


def _init_pair_groups(
    global_rank: int,
    *,
    instance_id: str,
    instance_epoch: str,
    topology_generation: str,
):
    import torch
    import torch.distributed as dist
    from prism_infer.utils.distributed import PairGroupRegistry, PAIR_GROUP_RANKS

    # Each StatefulSet pod owns exactly one GPU.  ``global_rank`` identifies the
    # worker in the four-process control/transfer world; it is not a pod-local
    # CUDA ordinal.  Pin the process before NCCL creates its communicators so a
    # logical rank such as d1/rank 3 cannot accidentally select cuda:3.
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "2P2D worker requires exactly one visible GPU per pod; "
            f"found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{_required('PRISM_MASTER_ADDR')}:{_required('PRISM_MASTER_PORT')}",
        rank=global_rank,
        world_size=4,
    )
    registry = PairGroupRegistry(global_rank=global_rank)
    registry.create_all()
    evidence_path = os.environ.get("PRISM_NCCL_DEBUG_LOG")
    for pair_id, ranks in PAIR_GROUP_RANKS:
        # The leading world barrier separates this probe from the preceding
        # pair.  The trailing barrier starts only after members captured their
        # local append-only log slice, so unrelated world traffic is excluded.
        dist.barrier()
        if global_rank in ranks:
            def probe():
                value = torch.tensor([global_rank + 1.0], device="cuda")
                dist.all_reduce(value, group=registry.pair(pair_id).process_group)
                torch.cuda.synchronize()

            registry.record_probe_attestation(
                pair_id,
                _capture_pair_probe(
                    pair_id=pair_id,
                    global_ranks=ranks,
                    global_rank=global_rank,
                    instance_id=instance_id,
                    instance_epoch=instance_epoch,
                    topology_generation=topology_generation,
                    evidence_path=evidence_path,
                    probe=probe,
                ),
            )
        dist.barrier()
        registry.mark_warmed_up(pair_id)
    return registry


async def _discover_capabilities(runtime, endpoints, pair_groups, evidence_dir):
    import httpx
    from prism_infer.utils.distributed import PAIR_GROUP_RANKS

    async with httpx.AsyncClient(timeout=5) as client:
        identities = {}
        deadline = asyncio.get_running_loop().time() + 120
        while set(identities) != set(endpoints):
            for name, endpoint in endpoints.items():
                try:
                    response = await client.get(f"{endpoint.rstrip('/')}/v1/identity")
                    response.raise_for_status()
                    identities[name] = response.json()
                except Exception:
                    pass
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("timed out discovering four worker identities")
            await asyncio.sleep(0.2)
    pairs = []
    transport_evidence = {}
    Path(evidence_dir).mkdir(parents=True, exist_ok=True)
    attester = str(runtime.identity["instance_id"])
    attester_epoch = str(runtime.identity["instance_epoch"])
    generation = str(runtime.registry.topology_generation)
    for pair_id, ranks in PAIR_GROUP_RANKS:
        source, target = pair_id.split("--")
        if attester not in {source, target}:
            continue
        source_epoch = str(identities[source]["instance_epoch"])
        target_epoch = str(identities[target]["instance_epoch"])
        transport, pair_transport_evidence = _attested_pair_transport(
            pair_groups.probe_attestation(pair_id),
            pair_id=pair_id,
            attester_instance=attester,
            attester_instance_epoch=attester_epoch,
            source_epoch=source_epoch,
            target_epoch=target_epoch,
            topology_generation=generation,
        )
        transport_evidence[pair_id] = pair_transport_evidence
        evidence_path = str(Path(evidence_dir) / f"{pair_id}.json")
        Path(evidence_path).write_text(json.dumps({
            "pair_id": pair_id,
            "attester_instance": attester,
            "probe_passed": pair_groups.ready(pair_id) and transport != "UNKNOWN",
            "source_epoch": source_epoch,
            "target_epoch": target_epoch,
            "transport": transport,
            "transport_evidence": pair_transport_evidence,
        }, sort_keys=True), encoding="utf-8")
        pairs.append({
            "pair_id": pair_id,
            "source_epoch": source_epoch,
            "target_epoch": target_epoch,
            "transport": transport,
            "probe_generation": generation,
            "probe_passed": pair_groups.ready(pair_id) and transport != "UNKNOWN",
            "evidence_path": evidence_path,
        })
    runtime.capabilities.clear()
    runtime.capabilities.update({
        "ready": _capability_ready(
            pairs, require_gdr=os.environ.get("PRISM_REQUIRE_GDR") == "1"
        ),
        "pairs": pairs,
        "transport_evidence": transport_evidence,
    })


async def _serve(
    engine,
    pair_groups,
    identity,
    endpoints,
    *,
    lifecycle,
    model_profile=None,
) -> bool:
    import nats
    import uvicorn
    from prism_infer.engine.pd_worker import PDWorkerConsumer
    from prism_infer.server.app import WorkerControlRuntime, create_app
    from prism_infer.server.operation_registry import OperationRegistry
    from prism_infer.server.runtime import (
        EngineControlRouter,
        EngineOwnerCommandQueue,
        MappedNCCLEndpoint,
        PDExecutionDriver,
    )

    registry = OperationRegistry(
        instance_id=identity["instance_id"],
        instance_epoch=identity["instance_epoch"],
        topology_generation=identity["topology_generation"],
    )
    mapped = MappedNCCLEndpoint(
        pair_groups, engine.model_runner.kv_cache,
        watchdog_timeout_s=float(
            os.environ.get("PRISM_NCCL_WATCHDOG_TIMEOUT_S", "30")
        ),
    )
    pd_driver = PDExecutionDriver(
        engine,
        role=str(identity["role"]),
        model_profile=model_profile,
    )
    control = EngineControlRouter(
        engine, prepare_receive=mapped.prepare_receive,
        start_transfer=mapped.start_transfer,
        transfer_terminal=mapped.operation_completed,
        transfer_release=mapped.release,
        request_committed=pd_driver.request_committed,
    )

    def finalize_local(operation_id: str, kinds: tuple[str, ...]):
        requested = set(kinds)
        transfer_kinds = tuple(sorted(
            requested & {"SOURCE_RETAIN", "TRANSFER_BYTES"}
        ))
        if transfer_kinds:
            mapped.validate_release(operation_id, transfer_kinds)
        if "SOURCE_BLOCKS" in requested:
            if requested - {"SOURCE_BLOCKS", *transfer_kinds}:
                raise ValueError("source request finalize mixed incompatible resources")
            pd_driver.validate_source_blocks(operation_id)
            result = pd_driver.release_source_blocks(operation_id)
        else:
            local_kinds = tuple(sorted(requested - set(transfer_kinds)))
            result = control.release(operation_id, local_kinds) if local_kinds else {}
        if transfer_kinds:
            result.update(mapped.release(operation_id, transfer_kinds))
        return result

    def route(operation, ref, payload):
        if operation == "finalize.local":
            return finalize_local(
                str(payload["operation_id"]),
                tuple(str(value) for value in payload["resource_kinds"]),
            )
        if operation == "status.local":
            if ref is None:
                raise ValueError("status refresh requires endpoint ref")
            return mapped.refresh(ref)
        if operation == "resources.local":
            return resource_details()
        if operation == "output.local":
            return pd_driver.output(
                str(payload["req_id"]), int(payload["after_seq"])
            )
        if operation == "prefix.local":
            return pd_driver.prefix_directory(
                str(payload["action"]), dict(payload["body"])
            )
        if operation == "prune.local":
            prune_side_state(
                set(payload["refs"]), set(payload["operation_ids"])
            )
            return None
        if operation == "watchdog.local":
            return {
                "expired": mapped.poll_watchdog(),
                "reason": mapped.watchdog_reason,
                "evidence": mapped.watchdog_evidence,
            }
        if operation.startswith("dispatch."):
            return pd_driver(operation.split(".", 1)[1], ref, payload)
        if operation == "abort.local":
            mapped_snapshot = mapped.abort(ref)
            if mapped_snapshot is not None:
                return mapped.refresh(ref)
            request_stopped = pd_driver.abort_request(ref.operation_id)
            prefix_stopped = engine.prefix_cache.abort_sequence(ref.operation_id)
            if not prefix_stopped:
                prefix_stopped = (
                    engine.prefix_cache.abort(ref.operation_id).value != "UNKNOWN"
                )
            from prism_infer.server.operation_registry import OperationSnapshot, OperationState
            if request_stopped or prefix_stopped:
                existing = registry.snapshot(ref)
                return OperationSnapshot(
                    ref, OperationState.FENCED,
                    resources_held=existing.resources_held,
                    held_resource_kinds=existing.held_resource_kinds,
                )
            return None
        return control(operation, ref, payload)

    def owner_idle() -> None:
        try:
            pd_driver.idle_step()
        finally:
            # Work/CUDA progress remains serialized on the original engine
            # owner.  A block here is covered by the independent deadline
            # supervisor above.
            mapped.poll_terminal_progress()

    owner = EngineOwnerCommandQueue(route, idle=owner_idle)
    def resource_details():
        details = pd_driver.resource_details(registry.snapshots())
        quantities = dict(details.get("resource_quantities", {}))
        for kind, count in mapped.resource_quantities().items():
            quantities[kind] = quantities.get(kind, 0) + count
        details["resource_quantities"] = quantities
        return details

    def prune_side_state(refs, operation_ids):
        mapped.prune(refs)
        engine.prefix_cache.prune_operations(operation_ids)
        pd_driver.prune(operation_ids)

    runtime = WorkerControlRuntime(
        identity=identity, capabilities={"ready": False, "pairs": []},
        registry=registry, command_handler=owner.submit,
        async_command_handler=owner.submit_async,
        async_release_handler=lambda operation_id, kinds: owner.submit_local_async(
            "finalize.local",
            {"operation_id": operation_id, "resource_kinds": list(kinds)},
        ),
        async_output_handler=lambda req_id, after_seq: owner.submit_local_async(
            "output.local", {"req_id": req_id, "after_seq": after_seq},
        ),
        async_status_handler=lambda ref: owner.submit_async("status.local", ref, {}),
        abort_handler=lambda ref: owner.submit("abort.local", ref, {}),
        async_abort_handler=lambda ref: owner.submit_async("abort.local", ref, {}),
        async_resource_details_handler=lambda: owner.submit_local_async(
            "resources.local", {}
        ),
        async_prefix_directory_handler=lambda action, body: owner.submit_local_async(
            "prefix.local", {"action": action, "body": body}
        ),
        async_prune_handler=lambda refs, operation_ids: owner.submit_local_async(
            "prune.local", {"refs": refs, "operation_ids": operation_ids}
        ),
    )
    nc = await nats.connect(_required("PRISM_NATS_URL"))

    async def execute(kind, ref, payload):
        return await runtime.execute_async(f"dispatch.{kind}", ref, payload)

    consumer = PDWorkerConsumer(
        identity["instance_id"], registry, nc, execute,
        prune=runtime.prune_side_state_async,
    )
    await consumer.connect()

    async def publish_decode_events():
        from queue import Empty
        while True:
            try:
                subject, event = pd_driver.events.get_nowait()
            except Empty:
                await asyncio.sleep(0.002)
                continue
            await nc.publish(subject, json.dumps(event).encode())

    event_task = asyncio.create_task(publish_decode_events())
    capability_task = asyncio.create_task(_discover_capabilities(
        runtime, endpoints, pair_groups,
        os.environ.get("PRISM_PAIR_EVIDENCE_DIR", "/var/run/prism/pair-probes"),
    ))
    server = uvicorn.Server(uvicorn.Config(
        create_app(runtime), host="0.0.0.0", port=int(_required("PRISM_RPC_PORT")),
        log_level="info",
    ))
    watchdog_task = asyncio.create_task(
        _supervise_watchdog(
            runtime,
            mapped,
            server,
            lifecycle=lifecycle,
        )
    )
    try:
        await server.serve()
    finally:
        capability_task.cancel()
        event_task.cancel()
        watchdog_task.cancel()
        await asyncio.gather(
            capability_task, event_task, watchdog_task, return_exceptions=True
        )
        await nc.drain()
        owner.close()
    return runtime.capabilities.get("worker_exit_code") == NCCL_WATCHDOG_EXIT_CODE


def _hold_worker_bootstrap() -> None:
    """Hold a restarted or uncertain pod without initializing GPU or NCCL."""

    interval_s = float(
        os.environ.get("PRISM_STARTUP_HOLD_INTERVAL_S", "1")
    )
    if interval_s <= 0:
        raise ValueError("startup hold interval must be positive")
    while True:
        time.sleep(interval_s)


def main() -> None:
    from prism_infer.server.startup_lifecycle import (
        PodIncarnationLifecycle,
        wait_for_startup_permit,
    )

    instance_id = _required("PRISM_INSTANCE_ID")
    role = _required("PRISM_ENGINE_ROLE")
    rank = int(_required("PRISM_GLOBAL_RANK"))
    pod_uid = _required("PRISM_POD_UID")
    topology_generation = _required("PRISM_TOPOLOGY_GENERATION")
    # Permit and create-once state precede model, CUDA, and NCCL initialization.
    # A later container in the same pod holds instead of re-entering old barriers.
    permit = wait_for_startup_permit(
        os.environ.get(
            "PRISM_STARTUP_PERMIT_PATH",
            "/etc/prism/topology/startup-permit.json",
        ),
        instance_id=instance_id,
        pod_uid=pod_uid,
        topology_generation=topology_generation,
    )
    process_generation = uuid.uuid4().hex
    instance_epoch = f"{pod_uid}:{process_generation}"
    lifecycle = PodIncarnationLifecycle(
        os.environ.get(
            "PRISM_INCARNATION_RECORD_PATH",
            "/var/run/prism/incarnation/record.json",
        ),
        instance_id=instance_id,
        pod_uid=pod_uid,
        topology_generation=topology_generation,
        process_generation=process_generation,
    )
    try:
        first_incarnation = lifecycle.create_active(permit)
    except (OSError, ValueError):
        _hold_worker_bootstrap()
        return
    if not first_incarnation:
        _hold_worker_bootstrap()
        return

    from prism_infer.config import Config
    from prism_infer.engine.llm_engine import LLMEngine
    from prism_infer.server.model_profile import preflight_model_profile

    # Model provenance fails closed before NCCL world or CUDA allocation.
    model_profile = preflight_model_profile()
    process_identity_path = os.environ.get("PRISM_PROCESS_IDENTITY_PATH", "")
    if process_identity_path:
        from prism_infer.server.process_identity import publish_process_identity

        publish_process_identity(
            process_identity_path,
            component="worker",
            instance_id=instance_id,
            pod_uid=pod_uid,
            process_generation=process_generation,
        )
    pair_groups = _init_pair_groups(
        rank,
        instance_id=instance_id,
        instance_epoch=instance_epoch,
        topology_generation=topology_generation,
    )
    config = Config(
        model=_required("PRISM_MODEL"), engine_mode="unified",
        instance_id=instance_id,
        tensor_parallel_size=(
            model_profile.tensor_parallel_size if model_profile is not None else 1
        ),
        kvcache_block_size=(
            model_profile.tokens_per_block if model_profile is not None else 256
        ),
    )
    engine = LLMEngine(config)
    engine.scheduler.block_manager.instance_epoch = instance_epoch
    endpoints = json.loads(_required("PRISM_WORKER_ENDPOINTS_JSON"))
    identity = {
        "instance_id": instance_id, "role": role,
        "topology_generation": topology_generation,
        "pod_uid": pod_uid, "process_generation": process_generation,
        "instance_epoch": instance_epoch,
        "rpc_endpoint": endpoints[instance_id], "global_rank": rank,
        "topology_digest": _required("PRISM_TOPOLOGY_DIGEST"),
        "kv_compatibility_id": _required("PRISM_KV_COMPATIBILITY_ID"),
    }
    watchdog_exit = asyncio.run(_serve(
        engine,
        pair_groups,
        identity,
        endpoints,
        lifecycle=lifecycle,
        model_profile=(
            model_profile.as_resource_report()
            if model_profile is not None else None
        ),
    ))
    if watchdog_exit:
        raise SystemExit(NCCL_WATCHDOG_EXIT_CODE)


if __name__ == "__main__":
    main()
