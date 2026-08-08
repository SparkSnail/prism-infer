"""Versioned HTTP control plane for worker operations.

CUDA and allocator mutations are delegated to the engine owner thread.
"""

from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    FencedWorkerEpoch,
    FinalizeReleaseRequest,
    OperationConflict,
    OperationRegistry,
    OperationSnapshot,
    OperationState,
    PreconditionFailed,
    RegistryCapacity,
    RegistryError,
    RequestOutputNotFound,
    RetiredOwner,
    StaleOperation,
    UnknownOwner,
    canonical_payload_digest,
)
from prism_infer.server.runtime import CONTROLLED_RESOURCE_KINDS


def _snapshot_json(snapshot: OperationSnapshot) -> dict[str, object]:
    value = asdict(snapshot)
    value["state"] = snapshot.state.value
    return value


def _ref(value: dict[str, object]) -> EndpointOperationRef:
    return EndpointOperationRef(**value)


@dataclass
class WorkerControlRuntime:
    identity: dict[str, object]
    capabilities: dict[str, object]
    registry: OperationRegistry
    command_handler: Callable[[str, EndpointOperationRef, dict[str, object]], OperationSnapshot] | None = None
    async_command_handler: Callable[[str, EndpointOperationRef, dict[str, object]], Awaitable[OperationSnapshot]] | None = None
    release_handler: Callable[[str, tuple[str, ...]], object] | None = None
    async_release_handler: Callable[[str, tuple[str, ...]], Awaitable[object]] | None = None
    output_handler: Callable[[str, int], dict[str, object]] | None = None
    async_output_handler: Callable[[str, int], Awaitable[dict[str, object]]] | None = None
    status_handler: Callable[[EndpointOperationRef], OperationSnapshot | None] | None = None
    async_status_handler: Callable[[EndpointOperationRef], Awaitable[OperationSnapshot | None]] | None = None
    abort_handler: Callable[[EndpointOperationRef], object] | None = None
    async_abort_handler: Callable[[EndpointOperationRef], Awaitable[object]] | None = None
    resource_details_handler: Callable[[], dict[str, object]] | None = None
    async_resource_details_handler: Callable[[], Awaitable[dict[str, object]]] | None = None
    prefix_directory_handler: Callable[[str, dict[str, object]], object] | None = None
    async_prefix_directory_handler: Callable[[str, dict[str, object]], Awaitable[object]] | None = None
    prune_handler: Callable[
        [set[EndpointOperationRef], set[str]], None
    ] | None = None
    async_prune_handler: Callable[
        [set[EndpointOperationRef], set[str]], Awaitable[None]
    ] | None = None
    released_resource_kinds: list[tuple[str, ...]] = field(default_factory=list)
    operation_kinds: dict[EndpointOperationRef, str] = field(default_factory=dict)
    _executions: dict[EndpointOperationRef, asyncio.Task] = field(default_factory=dict)
    _finalize_executions: dict[str, tuple[str, asyncio.Task]] = field(default_factory=dict)
    _finalize_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def execute(
        self,
        operation: str,
        ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> OperationSnapshot:
        if self.command_handler is not None:
            return self.command_handler(operation, ref, payload)
        # The default handler supports protocol tests but performs no GPU I/O.
        kinds = tuple(str(kind) for kind in payload.get("held_resource_kinds", ()))
        return OperationSnapshot.running(ref, held_resource_kinds=kinds)

    async def execute_async(
        self, operation: str, ref: EndpointOperationRef,
        payload: dict[str, object],
    ) -> OperationSnapshot:
        task = self._executions.get(ref)
        if task is None:
            async def invoke():
                if self.async_command_handler is not None:
                    return await self.async_command_handler(operation, ref, payload)
                if self.command_handler is not None:
                    return self.command_handler(operation, ref, payload)
                return self.execute(operation, ref, payload)

            task = asyncio.create_task(invoke())
            self._executions[ref] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._executions.pop(ref, None)

    def execution_inflight(self, ref: EndpointOperationRef) -> bool:
        return ref in self._executions

    async def abort_async(self, ref: EndpointOperationRef):
        if self.async_abort_handler is not None:
            return await self.async_abort_handler(ref)
        if self.abort_handler is not None:
            return self.abort_handler(ref)
        return None

    def release(self, operation_id: str, kinds: tuple[str, ...]) -> dict[str, int]:
        # Production assembly replaces this with the owner thread's atomic release.
        counts = {kind: 1 for kind in kinds}
        if self.release_handler is not None:
            result = self.release_handler(operation_id, kinds)
            if not isinstance(result, dict):
                raise ValueError("release handler must return released resource counts")
            counts = {str(kind): int(count) for kind, count in result.items()}
        self.released_resource_kinds.append(kinds)
        return counts

    async def release_async(
        self, operation_id: str, kinds: tuple[str, ...]
    ) -> dict[str, int]:
        if self.async_release_handler is not None:
            result = await self.async_release_handler(operation_id, kinds)
            if not isinstance(result, dict):
                raise ValueError("release handler must return released resource counts")
            counts = {str(kind): int(count) for kind, count in result.items()}
            self.released_resource_kinds.append(kinds)
            return counts
        return self.release(operation_id, kinds)

    def resources(self) -> dict[str, object]:
        details = self.resource_details_handler() \
            if self.resource_details_handler is not None else None
        return self._resources_with_details(details)

    async def resources_async(self) -> dict[str, object]:
        if self.async_resource_details_handler is not None:
            details = await self.async_resource_details_handler()
        else:
            details = self.resource_details_handler() \
                if self.resource_details_handler is not None else None
        return self._resources_with_details(details)

    def _resources_with_details(
        self, details_value: dict[str, object] | None
    ) -> dict[str, object]:
        snapshots = self.registry.snapshots()
        counts = {kind: 0 for kind in CONTROLLED_RESOURCE_KINDS}
        held_operation_ids = []
        for snapshot in snapshots:
            if not snapshot.resources_held:
                continue
            held_operation_ids.append(snapshot.endpoint_ref.operation_id)
            for kind in snapshot.held_resource_kinds:
                if kind not in counts:
                    raise ValueError(
                        f"unknown formal resource kind: {kind}"
                    )
                counts[kind] = counts.get(kind, 0) + 1
        value = {
            "instance_epoch": self.registry.instance_epoch,
            "complete": True,
            "resources": counts,
            "operation_ids": sorted(set(held_operation_ids)),
        }
        if details_value is not None:
            details = dict(details_value)
            quantities = details.pop("resource_quantities", None)
            if quantities is not None:
                if not isinstance(quantities, dict):
                    raise ValueError("resource_quantities must be an object")
                normalized_quantities = {
                    str(kind): int(quantity)
                    for kind, quantity in quantities.items()
                }
                unknown_kinds = (
                    set(normalized_quantities) - set(CONTROLLED_RESOURCE_KINDS)
                )
                if unknown_kinds:
                    raise ValueError(
                        "unknown formal resource kind: "
                        + ", ".join(sorted(unknown_kinds))
                    )
                counts.update(normalized_quantities)
            value.update(details)
        return value

    def prune_side_state(self) -> None:
        refs, operation_ids = self._prune_runtime_state()
        if self.prune_handler is not None:
            self.prune_handler(refs, operation_ids)

    async def prune_side_state_async(self) -> None:
        refs, operation_ids = self._prune_runtime_state()
        if self.async_prune_handler is not None:
            await self.async_prune_handler(refs, operation_ids)
        elif self.prune_handler is not None:
            self.prune_handler(refs, operation_ids)

    def _prune_runtime_state(
        self,
    ) -> tuple[set[EndpointOperationRef], set[str]]:
        snapshots = self.registry.snapshots()
        refs = {snapshot.endpoint_ref for snapshot in snapshots}
        operation_ids = {ref.operation_id for ref in refs}
        self.operation_kinds = {
            ref: kind for ref, kind in self.operation_kinds.items()
            if ref in refs
        }
        self._executions = {
            ref: task for ref, task in self._executions.items()
            if ref in refs and not task.done()
        }
        return refs, operation_ids

    def output(self, req_id: str, after_seq: int) -> dict[str, object]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        if self.output_handler is None:
            return {
                "req_id": req_id,
                "instance_epoch": self.registry.instance_epoch,
                "output_seq_no": 0,
                "token_ids": [],
                "terminal": False,
            }
        value = dict(self.output_handler(req_id, after_seq))
        return self._validate_output(req_id, value)

    async def output_async(self, req_id: str, after_seq: int) -> dict[str, object]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        if self.async_output_handler is not None:
            value = dict(await self.async_output_handler(req_id, after_seq))
            return self._validate_output(req_id, value)
        return self.output(req_id, after_seq)

    def _validate_output(
        self, req_id: str, value: dict[str, object]
    ) -> dict[str, object]:
        if value.get("req_id") != req_id:
            raise ValueError("output handler returned another request")
        if value.get("instance_epoch") != self.registry.instance_epoch:
            raise ValueError("output handler returned another worker epoch")
        token_ids = value.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) for token in token_ids
        ):
            raise ValueError("output token_ids must be an integer list")
        if value.get("output_seq_no") != len(token_ids):
            raise ValueError("output_seq_no must equal cumulative token count")
        return value

    async def refresh_status_async(
        self, ref: EndpointOperationRef
    ) -> OperationSnapshot | None:
        if self.async_status_handler is not None:
            return await self.async_status_handler(ref)
        if self.status_handler is not None:
            return self.status_handler(ref)
        return None

    async def prefix_directory_async(
        self, action: str, payload: dict[str, object]
    ) -> object:
        if self.async_prefix_directory_handler is not None:
            return await self.async_prefix_directory_handler(action, payload)
        if self.prefix_directory_handler is not None:
            return self.prefix_directory_handler(action, payload)
        raise ValueError("prefix directory handler is not installed")

    async def finalize_async(
        self, request: FinalizeReleaseRequest
    ) -> object:
        existing = self._finalize_executions.get(request.cleanup_id)
        if existing is not None:
            digest, task = existing
            if digest != request.payload_digest:
                raise OperationConflict("cleanup id reused with different digest")
            return await asyncio.shield(task)

        async def invoke():
            async with self._finalize_lock:
                replay = self.registry.finalize_replay(request)
                if replay is not None:
                    return replay
                kinds = self.registry.prepare_finalize_release(request)
                counts = await self.release_async(request.operation_id, kinds)
                return self.registry.commit_finalize_release(request, counts)

        task = asyncio.create_task(invoke())
        self._finalize_executions[request.cleanup_id] = (
            request.payload_digest, task
        )

        def forget(done: asyncio.Task) -> None:
            current = self._finalize_executions.get(request.cleanup_id)
            if current is not None and current[1] is done:
                self._finalize_executions.pop(request.cleanup_id, None)

        task.add_done_callback(forget)
        return await asyncio.shield(task)


def _error_response(exc: RegistryError) -> JSONResponse:
    if isinstance(exc, RequestOutputNotFound):
        code, status = "REQUEST_OUTPUT_NOT_FOUND", 404
    elif isinstance(exc, FencedWorkerEpoch):
        code, status = "FENCED_WORKER_EPOCH", 409
    elif isinstance(exc, UnknownOwner):
        code, status = "UNKNOWN_OWNER", 409
    elif isinstance(exc, RetiredOwner):
        code, status = "RETIRED_OWNER", 409
    elif isinstance(exc, StaleOperation):
        code, status = "STALE_OPERATION", 409
    elif isinstance(exc, OperationConflict):
        code, status = "CONFLICT", 409
    elif isinstance(exc, PreconditionFailed):
        code, status = "PRECONDITION_FAILED", 409
    elif isinstance(exc, RegistryCapacity):
        code, status = "REGISTRY_CAPACITY", 503
    else:
        code, status = "REGISTRY_ERROR", 409
    return JSONResponse({"code": code, "message": str(exc)}, status_code=status)


def create_app(runtime: WorkerControlRuntime) -> FastAPI:
    app = FastAPI(title="prism-infer worker control API")

    @app.get("/v1/identity")
    async def identity() -> dict[str, object]:
        return dict(runtime.identity)

    @app.get("/v1/capabilities")
    async def capabilities():
        value = dict(runtime.capabilities)
        if value.get("ready") is not True:
            return JSONResponse(value, status_code=503)
        return value

    @app.post("/v1/owners/activate")
    async def activate_owner(request: Request):
        if runtime.capabilities.get("ready") is not True:
            return JSONResponse(
                {
                    "code": "WORKER_NOT_READY",
                    "message": "worker is not ready for a new owner",
                },
                status_code=503,
            )
        body = await request.json()
        try:
            owner = runtime.registry.activate_owner(str(body["owner_generation"]))
            return {"active_owner": owner}
        except RegistryError as exc:
            return _error_response(exc)

    @app.get("/v1/owners/status")
    async def owner_status() -> dict[str, object]:
        return runtime.registry.owner_status()

    @app.post("/v1/owners/{owner_generation}/retire")
    async def retire_owner(owner_generation: str):
        try:
            retired = runtime.registry.retire_owner(owner_generation)
            await runtime.prune_side_state_async()
            return {**runtime.registry.owner_status(), "retired_owner": retired}
        except RegistryError as exc:
            return _error_response(exc)

    async def mutate(request: Request, operation: str):
        if runtime.capabilities.get("ready") is not True:
            return JSONResponse(
                {
                    "code": "WORKER_NOT_READY",
                    "message": "worker is not ready for new mutations",
                },
                status_code=503,
            )
        body = await request.json()
        if body.get("schema_version") != 1:
            return JSONResponse(
                {"code": "INVALID_SCHEMA", "message": "schema_version must be 1"},
                status_code=422,
            )
        try:
            ref = _ref(body["endpoint_ref"])
            payload = body.get("payload") or {}
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            if ref.payload_digest != canonical_payload_digest(payload):
                raise OperationConflict("endpoint ref payload digest mismatch")
            prior_kind = runtime.operation_kinds.get(ref)
            if prior_kind is not None and prior_kind != operation:
                raise OperationConflict("endpoint ref operation kind changed")
            kinds = tuple(
                str(value) for value in payload.get("held_resource_kinds", ())
            )
            snapshot, installed = runtime.registry.accept_or_replay(
                ref, lambda: OperationSnapshot.running(
                    ref, held_resource_kinds=kinds
                ),
            )
            # Only accepted/replayed registry refs may own side state.  Rejected
            # stale/wrong-owner/capacity refs must not grow this dictionary.
            runtime.operation_kinds[ref] = operation
            if installed or runtime.execution_inflight(ref):
                if installed:
                    runtime.registry.record_execution(ref)
                try:
                    result = await runtime.execute_async(operation, ref, payload)
                except Exception as exc:
                    runtime.registry.terminalize(
                        ref, OperationState.FENCED, reason=str(exc)
                    )
                    await runtime.prune_side_state_async()
                    raise
                snapshot = runtime.registry.store_result(ref, result)
            if installed and operation in {"request.commit", "prefix.commit"}:
                runtime.registry.migrate_operation_resources(
                    ref.operation_id, ref
                )
            await runtime.prune_side_state_async()
            status = 200 if snapshot.state.terminal else 202
            return JSONResponse(_snapshot_json(snapshot), status_code=status)
        except RegistryError as exc:
            return _error_response(exc)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(
                {"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422
            )

    @app.post("/v1/requests/prepare")
    async def prepare_request(request: Request):
        return await mutate(request, "request.prepare")

    @app.post("/v1/requests/commit")
    async def commit_request(request: Request):
        return await mutate(request, "request.commit")

    @app.post("/v1/transfers/prepare-receive")
    async def prepare_receive(request: Request):
        return await mutate(request, "transfer.prepare_receive")

    @app.post("/v1/transfers/start")
    async def start_transfer(request: Request):
        return await mutate(request, "transfer.start")

    @app.post("/v1/prefix/{action}")
    async def prefix_mutation(action: str, request: Request):
        if action == "abort":
            # This parameterized route is registered before the static abort
            # route below, so dispatch the action here rather than shadowing it
            # with a 404.
            return await abort(request)
        if action not in {"resolve", "prepare", "commit"}:
            return JSONResponse({"code": "NOT_FOUND"}, status_code=404)
        return await mutate(request, f"prefix.{action}")

    async def abort(request: Request, expected_operation_id: str | None = None):
        body = await request.json()
        try:
            ref = _ref(body["target_operation_ref"])
            if (
                expected_operation_id is not None
                and ref.operation_id != expected_operation_id
            ):
                raise OperationConflict("abort path operation id mismatch")
            existing = runtime.registry.classify_abort(ref)
            if existing is not None and existing.state.terminal:
                await runtime.prune_side_state_async()
                return _snapshot_json(existing)
            if existing is None:
                # An unseen ref has no writer to stop. Install a resource-free
                # terminal state so a late dispatch replays it without touching a
                # different writer that shares the operation_id.
                snapshot = runtime.registry.abort(
                    ref, reason=str(body.get("reason", "abort"))
                )
                await runtime.prune_side_state_async()
                return _snapshot_json(snapshot)
            if runtime.abort_handler is not None or runtime.async_abort_handler is not None:
                result = await runtime.abort_async(ref)
                if isinstance(result, OperationSnapshot):
                    snapshot = (
                        runtime.registry.terminalize(
                            ref, result.state, reason=result.reason
                        )
                        if result.state.terminal
                        else runtime.registry.store_result(ref, result)
                    )
                    await runtime.prune_side_state_async()
                    return _snapshot_json(snapshot)
            snapshot = runtime.registry.abort(
                ref, reason=str(body.get("reason", "abort"))
            )
            await runtime.prune_side_state_async()
            return _snapshot_json(snapshot)
        except RegistryError as exc:
            return _error_response(exc)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(
                {"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422
            )

    @app.post("/v1/requests/{operation_id}/abort")
    async def abort_request(operation_id: str, request: Request):
        return await abort(request, operation_id)

    @app.post("/v1/transfers/{operation_id}/abort")
    async def abort_transfer(operation_id: str, request: Request):
        return await abort(request, operation_id)

    @app.post("/v1/prefix/abort")
    async def abort_prefix(request: Request):
        return await abort(request)

    async def _refresh_transfer(snapshot: OperationSnapshot) -> OperationSnapshot:
        kind = runtime.operation_kinds.get(snapshot.endpoint_ref, "")
        if (
            (runtime.status_handler is not None or runtime.async_status_handler is not None)
            and kind.startswith("transfer.")
            and not snapshot.state.terminal
        ):
            refreshed = await runtime.refresh_status_async(snapshot.endpoint_ref)
            if refreshed is not None:
                result = runtime.registry.store_result(
                    snapshot.endpoint_ref, refreshed
                )
                await runtime.prune_side_state_async()
                return result
        return snapshot

    async def status_for(
        operation_id: str,
        *,
        allowed_kinds: tuple[str, ...],
        owner_generation: str | None = None,
    ):
        snapshots = runtime.registry.snapshots(owner_generation)
        matches = [
            snapshot for snapshot in snapshots
            if snapshot.endpoint_ref.operation_id == operation_id
            and runtime.operation_kinds.get(snapshot.endpoint_ref, "").startswith(
                allowed_kinds
            )
        ]
        if not matches:
            return JSONResponse({"code": "NOT_FOUND"}, status_code=404)
        snapshot = max(
            matches, key=lambda item: item.endpoint_ref.operation_seq
        )
        snapshot = await _refresh_transfer(snapshot)
        return _snapshot_json(snapshot)

    @app.get("/v1/requests/{operation_id}")
    async def request_status(operation_id: str):
        return await status_for(operation_id, allowed_kinds=("request.", "dispatch."))

    @app.get("/v1/requests/{req_id}/output")
    async def request_output(req_id: str, after_seq: int = 0):
        try:
            return await runtime.output_async(req_id, after_seq)
        except RequestOutputNotFound as exc:
            return _error_response(exc)
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                {"code": "INVALID_OUTPUT", "message": str(exc)}, status_code=409
            )

    @app.get("/v1/transfers/{operation_id}")
    async def transfer_status(operation_id: str):
        return await status_for(operation_id, allowed_kinds=("transfer.",))

    @app.get("/v1/prefix/status/{operation_id}")
    async def prefix_status(operation_id: str):
        return await status_for(operation_id, allowed_kinds=("prefix.",))

    @app.post("/v1/prefix/reports/register")
    async def prefix_report_register(request: Request):
        if runtime.prefix_directory_handler is None \
                and runtime.async_prefix_directory_handler is None:
            return JSONResponse({"code": "UNAVAILABLE"}, status_code=503)
        try:
            return await runtime.prefix_directory_async("register", await request.json())
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422)

    @app.get("/v1/prefix/events")
    async def prefix_events(
        consumer_id: str, generation: str, after_seq: int, limit: int = 256
    ):
        if runtime.prefix_directory_handler is None \
                and runtime.async_prefix_directory_handler is None:
            return JSONResponse({"code": "UNAVAILABLE"}, status_code=503)
        try:
            return {"events": await runtime.prefix_directory_async("peek", {
                "consumer_id": consumer_id, "generation": generation,
                "after_seq": after_seq, "limit": limit,
            })}
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422)

    @app.post("/v1/prefix/events/ack")
    async def prefix_events_ack(request: Request):
        if runtime.prefix_directory_handler is None \
                and runtime.async_prefix_directory_handler is None:
            return JSONResponse({"code": "UNAVAILABLE"}, status_code=503)
        try:
            await runtime.prefix_directory_async("ack", await request.json())
            return {"acked": True}
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422)

    @app.get("/v1/resources")
    async def resources() -> dict[str, object]:
        return await runtime.resources_async()

    @app.get("/v1/operations")
    async def operations(owner_generation: str | None = None) -> dict[str, object]:
        return {
            "instance_epoch": runtime.registry.instance_epoch,
            "complete": True,
            "operations": [
                _snapshot_json(snapshot)
                for snapshot in runtime.registry.snapshots(owner_generation)
            ],
        }

    @app.post("/v1/operations/status")
    async def operation_by_ref(request: Request):
        try:
            ref = _ref(await request.json())
            snapshot = runtime.registry.snapshot(ref)
            snapshot = await _refresh_transfer(snapshot)
            return _snapshot_json(snapshot)
        except RegistryError as exc:
            return _error_response(exc)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(
                {"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422
            )

    @app.post("/v1/cleanup/finalize")
    async def finalize(request: Request):
        body = await request.json()
        try:
            finalize_request = FinalizeReleaseRequest(
                cleanup_id=str(body["cleanup_id"]),
                operation_id=str(body["operation_id"]),
                lease_id=str(body["lease_id"]),
                endpoint_refs=tuple(_ref(value) for value in body["endpoint_refs"]),
                resource_kinds=tuple(str(kind) for kind in body["resource_kinds"]),
                release_basis=str(body["release_basis"]),
                payload_digest=str(body["payload_digest"]),
            )
            snapshot = await runtime.finalize_async(finalize_request)
            await runtime.prune_side_state_async()
            return asdict(snapshot)
        except RegistryError as exc:
            return _error_response(exc)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(
                {"code": "INVALID_SCHEMA", "message": str(exc)}, status_code=422
            )

    return app
