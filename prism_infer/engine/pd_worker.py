"""NATS command consumer and owner-scoped event publisher for P/D workers."""

from dataclasses import asdict
import json
from typing import Awaitable, Callable
import asyncio
import inspect

from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    OperationRegistry,
    OperationSnapshot,
    OperationState,
    RegistryError,
    canonical_payload_digest,
)


class PDWorkerConsumer:
    """Use NATS for wakeups while the local registry remains authoritative."""

    def __init__(
        self,
        instance_id: str,
        registry: OperationRegistry,
        nats_client,
        execute: Callable[
            [str, EndpointOperationRef, dict[str, object]],
            Awaitable[OperationSnapshot],
        ],
        prune: Callable[[], None] | None = None,
    ):
        self.instance_id = instance_id
        self.registry = registry
        self.nats = nats_client
        self.execute = execute
        self.prune = prune
        self._executions: dict[EndpointOperationRef, asyncio.Task] = {}

    async def connect(self) -> None:
        await self.nats.subscribe(
            f"dispatch_prefill.{self.instance_id}",
            cb=self._handler("prefill", "prefill_done"),
        )
        await self.nats.subscribe(
            f"dispatch_suffix.{self.instance_id}",
            cb=self._handler("suffix", "suffix_prefill_done"),
        )

    def _handler(self, kind: str, expected_event: str):
        async def handler(message) -> None:
            try:
                body = json.loads(message.data.decode())
                if body.get("schema_version") != 1:
                    return
                ref = EndpointOperationRef(**body["endpoint_ref"])
                payload = body.get("payload") or {}
                if not isinstance(payload, dict):
                    return
                if ref.payload_digest != canonical_payload_digest(payload):
                    return
                reply_subject = str(payload["reply_subject"])
                self._validate_reply_subject(reply_subject, expected_event)
                kinds = tuple(
                    str(value) for value in payload.get("held_resource_kinds", ())
                )
                snapshot, installed = self.registry.accept_or_replay(
                    ref,
                    lambda: OperationSnapshot.running(
                        ref, held_resource_kinds=kinds
                    ),
                )
                snapshot = self.registry.record_delivery(ref)
                if not snapshot.state.terminal:
                    task = self._executions.get(ref)
                    if task is None:
                        if not installed:
                            # A RUNNING replay can only be completed by the
                            # first in-process executor.  Yield once so that
                            # caller can observe its shared task publication.
                            await asyncio.sleep(0)
                            task = self._executions.get(ref)
                        if task is None:
                            self.registry.record_execution(ref)
                            task = asyncio.create_task(
                                self.execute(kind, ref, payload)
                            )
                            self._executions[ref] = task
                    try:
                        result = await asyncio.shield(task)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # An executor failure is an authoritative writer
                        # terminal, not permission to run the same endpoint ref
                        # again.  Preserve any held resources on the FENCED
                        # snapshot; the generic finalize evaluator remains the
                        # only release authority.
                        snapshot = self.registry.terminalize(
                            ref,
                            OperationState.FENCED,
                            reason=f"executor failed: {type(exc).__name__}",
                        )
                    else:
                        snapshot = self.registry.store_result(ref, result)
                    finally:
                        if task.done():
                            self._executions.pop(ref, None)
                if self.prune is not None:
                    result = self.prune()
                    if inspect.isawaitable(result):
                        await result
                event = asdict(snapshot)
                event["state"] = snapshot.state.value
                if snapshot.result:
                    event.update(snapshot.result)
                event["req_id"] = payload.get("req_id")
                event["operation_id"] = ref.operation_id
                event["instance_epoch"] = self.registry.instance_epoch
                await self.nats.publish(reply_subject, json.dumps(event).encode())
            except (KeyError, TypeError, ValueError, RegistryError, json.JSONDecodeError):
                # Malformed or stale messages cannot terminate the consumer loop.
                return
            except Exception:
                # Publishing failure must not discard the stored terminal result.
                return

        return handler

    @staticmethod
    def _validate_reply_subject(subject: str, expected_event: str) -> None:
        assert subject.startswith(f"{expected_event}.")
        assert "*" not in subject and ">" not in subject
        assert len(subject.split(".")) == 2
