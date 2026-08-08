import json
import asyncio

import pytest

from prism_infer.engine.pd_worker import PDWorkerConsumer
from prism_infer.server.operation_registry import (
    EndpointOperationRef,
    OperationRegistry,
    OperationSnapshot,
    OperationState,
    canonical_payload_digest,
)


class FakeNATS:
    def __init__(self):
        self.subscriptions = {}
        self.published = []

    async def subscribe(self, subject, cb):
        self.subscriptions[subject] = cb

    async def publish(self, subject, payload):
        self.published.append((subject, json.loads(payload.decode())))


class Message:
    def __init__(self, value):
        self.data = json.dumps(value).encode()


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


def _registry():
    registry = OperationRegistry(
        instance_id="d0", instance_epoch="pod-d0:boot-a",
        topology_generation="world-a",
    )
    registry.activate_owner("gateway-a:boot-a")
    return registry


@pytest.mark.asyncio
async def test_suffix_worker_consumes_target_subject_and_publishes_owner_event():
    nats = FakeNATS()
    registry = _registry()
    executed = []

    async def execute(kind, ref, payload):
        executed.append((kind, payload["remaining_token_ids"]))
        return OperationSnapshot(ref, OperationState.COMPLETED)

    worker = PDWorkerConsumer("d0", registry, nats, execute)
    await worker.connect()
    payload = {
        "remaining_token_ids": [5, 6],
        "reply_subject": "suffix_prefill_done.gateway-a--boot-a",
    }
    ref = _ref(payload=payload)
    await nats.subscriptions["dispatch_suffix.d0"](Message({
        "schema_version": 1,
        "endpoint_ref": ref.__dict__,
        "payload": payload,
    }))

    assert executed == [("suffix", [5, 6])]
    assert nats.published[0][0] == "suffix_prefill_done.gateway-a--boot-a"
    assert nats.published[0][1]["endpoint_ref"]["operation_seq"] == 1


@pytest.mark.asyncio
async def test_cancel_before_nats_arrival_never_executes_worker_callback():
    nats = FakeNATS()
    registry = _registry()
    executed = False

    async def execute(kind, ref, payload):
        nonlocal executed
        executed = True
        return OperationSnapshot(ref, OperationState.COMPLETED)

    worker = PDWorkerConsumer("d0", registry, nats, execute)
    await worker.connect()
    payload = {"reply_subject": "suffix_prefill_done.gateway-a--boot-a"}
    ref = _ref(payload=payload)
    registry.abort(ref, reason="publish unknown")
    await nats.subscriptions["dispatch_suffix.d0"](Message({
        "schema_version": 1,
        "endpoint_ref": ref.__dict__,
        "payload": payload,
    }))

    assert executed is False
    assert nats.published[0][1]["state"] == "FENCED"


@pytest.mark.asyncio
async def test_nats_payload_digest_mismatch_has_zero_execution():
    nats = FakeNATS()
    registry = _registry()
    executed = False

    async def execute(kind, ref, payload):
        nonlocal executed
        executed = True
        return OperationSnapshot(ref, OperationState.COMPLETED)

    worker = PDWorkerConsumer("d0", registry, nats, execute)
    await worker.connect()
    ref = _ref(payload={})
    await nats.subscriptions["dispatch_suffix.d0"](Message({
        "schema_version": 1,
        "endpoint_ref": ref.__dict__,
        "payload": {"reply_subject": "suffix_prefill_done.gateway-a--boot-a"},
    }))
    assert executed is False
    assert registry.snapshots() == ()
    assert nats.published == []


@pytest.mark.asyncio
async def test_concurrent_duplicate_nats_shares_one_executor_and_two_replays():
    nats = FakeNATS()
    registry = _registry()
    entered = asyncio.Event()
    release = asyncio.Event()
    executions = 0

    async def execute(kind, ref, payload):
        nonlocal executions
        executions += 1
        entered.set()
        await release.wait()
        return OperationSnapshot(ref, OperationState.COMPLETED)

    worker = PDWorkerConsumer("d0", registry, nats, execute)
    await worker.connect()
    payload = {"reply_subject": "suffix_prefill_done.gateway-a--boot-a"}
    ref = _ref(payload=payload)
    message = Message({
        "schema_version": 1, "endpoint_ref": ref.__dict__, "payload": payload,
    })
    handler = nats.subscriptions["dispatch_suffix.d0"]
    first = asyncio.create_task(handler(message))
    await entered.wait()
    second = asyncio.create_task(handler(message))
    await asyncio.sleep(0)
    assert executions == 1
    release.set()
    await asyncio.gather(first, second)

    assert executions == 1
    assert len(nats.published) == 2
    assert {event[1]["state"] for event in nats.published} == {"COMPLETED"}
    snapshot = registry.snapshot(ref)
    assert snapshot.delivery_count == 2
    assert snapshot.execution_count == 1


@pytest.mark.asyncio
async def test_executor_failure_fences_ref_and_redelivery_never_executes_again():
    nats = FakeNATS()
    registry = _registry()
    executions = 0

    async def execute(kind, ref, payload):
        nonlocal executions
        executions += 1
        raise RuntimeError("engine failed after accepting the command")

    worker = PDWorkerConsumer("d0", registry, nats, execute)
    await worker.connect()
    payload = {
        "reply_subject": "suffix_prefill_done.gateway-a--boot-a",
        "held_resource_kinds": ["TARGET_SEQUENCE"],
    }
    ref = _ref(payload=payload)
    message = Message({
        "schema_version": 1, "endpoint_ref": ref.__dict__, "payload": payload,
    })
    handler = nats.subscriptions["dispatch_suffix.d0"]

    await handler(message)
    await handler(message)

    snapshot = registry.snapshot(ref)
    assert executions == 1
    assert snapshot.state == OperationState.FENCED
    assert snapshot.resources_held is True
    assert snapshot.held_resource_kinds == ("TARGET_SEQUENCE",)
    assert snapshot.reason == "executor failed: RuntimeError"
    assert len(nats.published) == 2
    assert {event[1]["state"] for event in nats.published} == {"FENCED"}
    assert worker._executions == {}


@pytest.mark.asyncio
async def test_concurrent_failed_duplicates_share_one_executor_and_terminal():
    nats = FakeNATS()
    registry = _registry()
    entered = asyncio.Event()
    release = asyncio.Event()
    executions = 0

    async def execute(kind, ref, payload):
        nonlocal executions
        executions += 1
        entered.set()
        await release.wait()
        raise RuntimeError("shared failure")

    worker = PDWorkerConsumer("d0", registry, nats, execute)
    await worker.connect()
    payload = {"reply_subject": "suffix_prefill_done.gateway-a--boot-a"}
    ref = _ref(payload=payload)
    message = Message({
        "schema_version": 1, "endpoint_ref": ref.__dict__, "payload": payload,
    })
    handler = nats.subscriptions["dispatch_suffix.d0"]
    first = asyncio.create_task(handler(message))
    await entered.wait()
    second = asyncio.create_task(handler(message))
    await asyncio.sleep(0)
    assert executions == 1
    release.set()
    await asyncio.gather(first, second)

    assert executions == 1
    assert registry.snapshot(ref).state == OperationState.FENCED
    assert len(nats.published) == 2
    assert worker._executions == {}
