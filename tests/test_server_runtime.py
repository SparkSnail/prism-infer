from prism_infer.server.operation_registry import EndpointOperationRef, OperationSnapshot, OperationState
from prism_infer.server.runtime import (
    EngineControlRouter, EngineOwnerCommandQueue, MappedNCCLEndpoint, PDExecutionDriver,
    endpoint_ref_key,
)
from types import SimpleNamespace
from collections import deque
from unittest.mock import patch
import pytest


def test_engine_owner_command_queue_executes_on_single_owner_thread():
    calls = []

    def handler(operation, ref, payload):
        import threading
        calls.append((operation, threading.current_thread().name, payload["value"]))
        return OperationSnapshot(ref, OperationState.COMPLETED)

    ref = EndpointOperationRef("world", "owner", 1, "d0", "epoch", "op", "sha256:x")
    queue = EngineOwnerCommandQueue(handler)
    try:
        result = queue.submit("request.prepare", ref, {"value": 7})
    finally:
        queue.close()
    assert result.state == OperationState.COMPLETED
    assert calls == [("request.prepare", "prism-engine-owner", 7)]


def test_engine_owner_queue_survives_idle_callback_exception():
    import threading

    idle_failed = threading.Event()
    idle_calls = 0

    def idle():
        nonlocal idle_calls
        idle_calls += 1
        if idle_calls == 1:
            idle_failed.set()
            raise RuntimeError("idle step failed")

    def handler(_operation, _ref, payload):
        return payload["value"]

    queue = EngineOwnerCommandQueue(handler, idle=idle)
    try:
        assert idle_failed.wait(timeout=1.0)
        future = queue.submit_future("probe", None, {"value": 7})
        assert future.result(timeout=1.0) == 7
    finally:
        queue.close()


def test_mapped_nccl_endpoint_binds_ref_to_terminal_work_and_cuda_fence_cpu():
    import torch

    class PairGroups:
        def pair(self, pair_id):
            return type("Pair", (), {"pair_id": pair_id, "process_group": None})()

        def group_peer(self, pair_id):
            return 1

    controller = MappedNCCLEndpoint(PairGroups(), torch.zeros(2, 1, 4, 1, 1, 1))
    ref = EndpointOperationRef("world", "owner", 1, "d0", "epoch", "op", "sha256:x")
    result = controller.prepare_receive(ref, {
        "source_instance": "p0", "target_instance": "d0", "dst_block_ids": [1]
    })
    assert result.state == OperationState.COMPLETED
    assert result.resources_held is False
    assert result.held_resource_kinds == ()


def test_mapped_nccl_p2p_peer_uses_global_rank_not_group_local_rank():
    import torch

    class PairGroups:
        def global_peer(self, pair_id):
            assert pair_id == "p0--d1"
            return 3

        def group_peer(self, pair_id):
            raise AssertionError("P2POp must not receive group-local peer")

    controller = MappedNCCLEndpoint(
        PairGroups(), torch.zeros(2, 1, 2, 1, 1, 1)
    )
    assert controller._peer("p0--d1") == 3


def test_mapped_nccl_receive_ack_does_not_wait_for_peer_send():
    """Target ACK precedes peer send while the owner retains Work adoption."""
    import threading
    import torch

    class Buffer:
        def numel(self):
            return 1

        def element_size(self):
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

    class Work:
        def is_completed(self):
            return False

    class CudaEvent:
        def query(self):
            return False

    peer_send = threading.Event()

    def batch_isend_irecv(_ops):
        if not peer_send.wait(timeout=1.0):
            raise RuntimeError("peer send was never allowed to start")
        return [Work()]

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(torch, "empty_like", side_effect=lambda value: value), \
         patch.object(torch.distributed, "P2POp", return_value=object()), \
         patch.object(
             torch.distributed, "batch_isend_irecv",
             side_effect=batch_isend_irecv,
         ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
        accepted = controller.prepare_receive(ref, {
            "source_instance": "p0", "target_instance": "d0",
            "src_block_ids": [1], "dst_block_ids": [2], "kv_size_bytes": 1,
        })
        assert accepted.state == OperationState.UNKNOWN
        key = endpoint_ref_key(ref)
        assert key in controller._pending_receive_launches
        assert controller.registry._operations[key].launched is True

        original_launch = controller._pending_receive_launches[key][0]
        replay = controller.prepare_receive(ref, {
            "source_instance": "p0", "target_instance": "d0",
            "src_block_ids": [1], "dst_block_ids": [2], "kv_size_bytes": 1,
        })
        assert replay.state == OperationState.UNKNOWN
        assert controller._pending_receive_launches[key][0] is original_launch

        peer_send.set()
        controller._pending_receive_launches[key][0].result(timeout=1.0)
        controller.refresh(ref)

    assert controller.registry._operations[key].launched is True
    assert key in controller._pending_receives


def test_mapped_nccl_watchdog_advances_coalesced_receive_without_gateway_query():
    """Worker tick owns completion even when Gateway never queries the ref."""
    import threading
    import torch

    copy_calls = []

    class Buffer:
        def __init__(self, block_id):
            self.block_id = block_id

        def numel(self):
            return 1

        def copy_(self, value):
            copy_calls.append((self.block_id, value.block_id))

    class Cache:
        shape = (2, 1, 4, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, key):
            return Buffer(key[2])

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 0

    class Work:
        done = False
        waits = 0

        def is_completed(self):
            return self.done

        def wait(self):
            assert self.done
            self.waits += 1

    class CudaEvent:
        recorded = False

        def record(self):
            self.recorded = True

        def query(self):
            return self.recorded

    work = Work()
    event = CudaEvent()
    peer_send = threading.Event()

    def batch_isend_irecv(ops):
        assert len(ops) == 3
        peer_send.wait(timeout=1.0)
        return [work]

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(
             torch, "empty_like",
             side_effect=lambda value: Buffer(value.block_id + 10),
         ), patch.object(torch.distributed, "P2POp", return_value=object()), \
         patch.object(
             torch.distributed, "batch_isend_irecv",
             side_effect=batch_isend_irecv,
         ), patch.object(torch.cuda, "Event", return_value=event):
        accepted = controller.prepare_receive(ref, {
            "source_instance": "p0", "target_instance": "d0",
            "src_block_ids": [7, 8, 9], "dst_block_ids": [1, 2, 3],
            "kv_size_bytes": 3,
        })
        assert accepted.state == OperationState.UNKNOWN

        key = endpoint_ref_key(ref)
        peer_send.set()
        controller._pending_receive_launches[key][0].result(timeout=1.0)
        work.done = True
        assert controller.poll_watchdog() is False
        completed = controller.registry.status(key)

    assert completed.status.value == "COMPLETED"
    assert work.waits == 1
    assert event.recorded is True
    assert copy_calls == [(1, 11), (2, 12), (3, 13)]


def test_mapped_nccl_watchdog_covers_receive_launch_blocked_before_work():
    """The watchdog deadline starts before NCCL returns a Work handle."""
    import threading
    import torch

    class Buffer:
        def numel(self):
            return 1

        def element_size(self):
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

    release = threading.Event()
    now = [10.0]

    def blocked_receive(_ops):
        release.wait(timeout=1.0)
        return []

    controller = MappedNCCLEndpoint(
        PairGroups(), Cache(), watchdog_timeout_s=1.0,
        clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    try:
        with patch.object(torch.distributed, "is_initialized", return_value=True), \
             patch.object(torch, "empty_like", side_effect=lambda value: value), \
             patch.object(torch.distributed, "P2POp", return_value=object()), \
             patch.object(
                 torch.distributed, "batch_isend_irecv",
                 side_effect=blocked_receive,
             ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
            controller.prepare_receive(ref, {
                "source_instance": "p0", "target_instance": "d0",
                "src_block_ids": [1], "dst_block_ids": [2],
                "kv_size_bytes": 1,
            })
            now[0] = 11.0
            assert controller.poll_watchdog() is True
    finally:
        release.set()
        controller._pending_receive_launches[key][0].result(timeout=1.0)

    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert controller._launch_errors[key] == \
        "RuntimeError: NCCL receive returned no Work handles"
    assert key in controller._failed_receive_buffers
    assert controller.watchdog_evidence["operation_id"] == "op"


def test_mapped_nccl_abort_retains_late_receive_until_terminal_without_kv_copy():
    """Abort cannot fake a fence; late Work remains buffered until terminal."""
    import threading
    import torch

    copy_calls = []

    class Buffer:
        def numel(self):
            return 1

        def copy_(self, _value):
            copy_calls.append(True)

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

    class Work:
        done = False

        def is_completed(self):
            return self.done

        def wait(self):
            assert self.done

    class CudaEvent:
        recorded = False

        def record(self):
            self.recorded = True

        def query(self):
            return self.recorded

    peer_send = threading.Event()
    work = Work()

    def batch_isend_irecv(_ops):
        peer_send.wait(timeout=1.0)
        return [work]

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(torch, "empty_like", side_effect=lambda value: value), \
         patch.object(torch.distributed, "P2POp", return_value=object()), \
         patch.object(
             torch.distributed, "batch_isend_irecv",
             side_effect=batch_isend_irecv,
         ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
        controller.prepare_receive(ref, {
            "source_instance": "p0", "target_instance": "d0",
            "src_block_ids": [1], "dst_block_ids": [2], "kv_size_bytes": 1,
        })
        aborted = controller.abort(ref)
        assert aborted.status.value == "UNKNOWN"

        key = endpoint_ref_key(ref)
        peer_send.set()
        controller._pending_receive_launches[key][0].result(timeout=1.0)
        assert controller.refresh(ref).state == OperationState.UNKNOWN
        assert key in controller._discarded_receives

        work.done = True
        assert controller.refresh(ref).state == OperationState.FENCED

    assert key not in controller._discarded_receives
    assert copy_calls == []


def test_mapped_nccl_prune_retains_pending_launcher_until_terminal_fence():
    """Outer-ref pruning aborts first and retains state until terminal."""
    import threading
    import torch

    class Buffer:
        def numel(self):
            return 1

        def copy_(self, _value):
            raise AssertionError("pruned receive must not copy into KV cache")

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

    class Work:
        done = False

        def is_completed(self):
            return self.done

        def wait(self):
            assert self.done

    class CudaEvent:
        recorded = False

        def record(self):
            self.recorded = True

        def query(self):
            return self.recorded

    peer_send = threading.Event()
    work = Work()

    def batch_isend_irecv(_ops):
        peer_send.wait(timeout=1.0)
        return [work]

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(torch, "empty_like", side_effect=lambda value: value), \
         patch.object(torch.distributed, "P2POp", return_value=object()), \
         patch.object(
             torch.distributed, "batch_isend_irecv",
             side_effect=batch_isend_irecv,
         ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
        controller.prepare_receive(ref, {
            "source_instance": "p0", "target_instance": "d0",
            "src_block_ids": [1], "dst_block_ids": [2], "kv_size_bytes": 1,
        })
        key = endpoint_ref_key(ref)
        controller.prune(set())
        assert key in controller.registry._operations
        assert controller.registry.status(key).status.value == "UNKNOWN"

        peer_send.set()
        controller._pending_receive_launches[key][0].result(timeout=1.0)
        controller.poll_watchdog()
        controller.prune(set())
        assert key in controller.registry._operations

        work.done = True
        controller.poll_watchdog()
        controller.prune(set())

    assert key not in controller.registry._operations


def test_mapped_nccl_receive_launch_error_remains_unknown_until_watchdog():
    """grouped launch exception is ambiguous and must retain buffers until exit."""
    import threading
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
    now = [10.0]

    def batch_isend_irecv(_ops):
        fail_launch.wait(timeout=1.0)
        raise RuntimeError("synthetic launch failure")

    controller = MappedNCCLEndpoint(
        PairGroups(), Cache(), watchdog_timeout_s=1.0,
        clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(torch, "empty_like", side_effect=lambda value: value), \
         patch.object(torch.distributed, "P2POp", return_value=object()), \
         patch.object(
             torch.distributed, "batch_isend_irecv",
             side_effect=batch_isend_irecv,
         ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
        controller.prepare_receive(ref, {
            "source_instance": "p0", "target_instance": "d0",
            "src_block_ids": [1], "dst_block_ids": [2], "kv_size_bytes": 1,
        })
        key = endpoint_ref_key(ref)
        fail_launch.set()
        assert isinstance(
            controller._pending_receive_launches[key][0].exception(timeout=1.0),
            RuntimeError,
        )
        snapshot = controller.refresh(ref)
        assert snapshot.state == OperationState.UNKNOWN
        assert "synthetic launch failure" in snapshot.reason

        assert controller.registry.status(key).status.value == "UNKNOWN"
        assert key in controller._failed_receive_buffers
        now[0] = 11.0
        assert controller.poll_watchdog() is True


def test_mapped_nccl_watchdog_latches_process_termination_for_stuck_work():
    import torch

    from prism_infer.server.runtime import endpoint_ref_key

    class Pending:
        def is_completed(self):
            return False

        def query(self):
            return False

    now = [10.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    controller.registry.register(key, resource_kinds=("TRANSFER_BYTES",))
    controller.registry.mark_launched(key, [Pending()], Pending())
    controller._operation_refs[key] = ref
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0",
        "completed_bytes": 1,
    }
    controller._launched_at[key] = now[0]

    now[0] = 10.999
    assert controller.poll_watchdog() is False
    now[0] = 11.0
    assert controller.poll_watchdog() is True
    assert controller.termination_requested is True
    assert key in controller.watchdog_reason
    assert controller.watchdog_evidence == {
        "kind": "nccl_watchdog_timeout",
        "reason": controller.watchdog_reason,
        "pair_id": "p0--d0",
        "endpoint_key": key,
        "endpoint_ref": ref.__dict__,
        "operation_id": "op",
        "watchdog_timeout_s": 1.0,
    }
    assert controller.refresh(ref).state == OperationState.UNKNOWN


def test_mapped_nccl_watchdog_advances_source_without_gateway_query():
    import torch

    calls = []

    class Work:
        def is_completed(self):
            calls.append("is_completed")
            return True

        def wait(self):
            calls.append("wait")

    class Event:
        def record(self):
            calls.append("record")

        def query(self):
            return "record" in calls

    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1)
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [Work()], Event())
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = controller._clock()
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    snapshot = controller.registry.status(key)

    assert snapshot.status.value == "COMPLETED"
    assert calls.index("wait") < calls.index("record")
    assert controller._transfer_metadata[key]["completed_bytes"] == 8


def test_mapped_nccl_work_terminal_is_latched_after_successful_wait():
    import torch

    calls = []

    class Work:
        waited = False

        def is_completed(self):
            calls.append("is_completed")
            return not self.waited

        def wait(self):
            calls.append("wait")
            self.waited = True

    class Event:
        recorded = False

        def record(self):
            calls.append("record")
            self.recorded = True

        def query(self):
            calls.append("query")
            return self.recorded

    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1)
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], Event())
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = controller._clock()
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    snapshot = controller.registry.status(key)

    assert snapshot.status.value == "COMPLETED"
    assert snapshot.work_terminal is True
    assert calls.count("is_completed") == 1
    assert calls.index("wait") < calls.index("record") < calls.index("query")


def test_mapped_nccl_source_poll_error_stays_unknown_until_watchdog():
    import torch

    class Work:
        def is_completed(self):
            raise RuntimeError("source Work query failed")

    class Event:
        def query(self):
            raise AssertionError("unrecorded event must not be queried")

    now = [0.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    buffer = object()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], Event())
    controller._pending_sends[key] = ([work], [buffer])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = 0.0
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert controller.refresh(ref).reason == "RuntimeError: source Work query failed"
    assert controller._pending_sends[key][1] == [buffer]

    now[0] = 1.0
    assert controller.poll_watchdog() is True
    assert controller.watchdog_evidence["endpoint_key"] == key


def test_mapped_nccl_discarded_receive_wait_error_retains_buffer_until_watchdog():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            raise RuntimeError("discarded receive wait failed")

    class Event:
        def record(self):
            raise AssertionError("event must not record after failed wait")

        def query(self):
            raise AssertionError("unrecorded event must not be queried")

    now = [0.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    buffer = (1, object())
    controller.registry.register(key, resource_kinds=())
    controller.registry.mark_launched(key, [work], Event())
    controller.registry._operations[key].accepting_new_work = False
    controller._receive_keys.add(key)
    controller._discarded_receives[key] = ([work], [buffer], False)
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = 0.0
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    assert key in controller._discarded_receives
    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert controller.refresh(ref).reason == (
        "RuntimeError: discarded receive wait failed"
    )

    now[0] = 1.0
    assert controller.poll_watchdog() is True
    assert key in controller._discarded_receives


def test_mapped_nccl_cuda_query_error_retains_source_until_watchdog():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            return None

    class Event:
        def record(self):
            return None

        def query(self):
            raise RuntimeError("CUDA event query failed")

    now = [0.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    buffer = object()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], Event())
    controller._pending_sends[key] = ([work], [buffer])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = 0.0
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    assert key in controller._pending_sends
    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert controller.refresh(ref).reason == "RuntimeError: CUDA event query failed"

    now[0] = 1.0
    assert controller.poll_watchdog() is True
    assert key in controller._pending_sends
    assert "CUDA event query failed" in controller.watchdog_reason
    assert controller.watchdog_evidence["reason"] == controller.watchdog_reason


def test_mapped_nccl_one_shot_source_query_error_cannot_drop_watchdog():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            return None

    class Event:
        query_calls = 0

        def record(self):
            return None

        def query(self):
            self.query_calls += 1
            if self.query_calls == 1:
                raise RuntimeError("one-shot source CUDA query failed")
            return True

    now = [0.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    event = Event()
    buffer = object()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], event)
    controller._pending_sends[key] = ([work], [buffer])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = 0.0
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    assert event.query_calls == 1
    assert key in controller._launched_at
    assert key in controller._pending_sends
    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert event.query_calls == 1

    now[0] = 1.0
    assert controller.poll_watchdog() is True
    assert key in controller._pending_sends
    assert event.query_calls == 1
    assert "one-shot source CUDA query failed" in controller.watchdog_reason


def test_mapped_nccl_one_shot_discarded_query_error_cannot_drop_watchdog():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            return None

    class Event:
        query_calls = 0

        def record(self):
            return None

        def query(self):
            self.query_calls += 1
            if self.query_calls == 1:
                raise RuntimeError("one-shot discarded CUDA query failed")
            return True

    now = [0.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    event = Event()
    buffer = (1, object())
    controller.registry.register(key, resource_kinds=())
    controller.registry.mark_launched(key, [work], event)
    controller.registry._operations[key].accepting_new_work = False
    controller._receive_keys.add(key)
    controller._discarded_receives[key] = ([work], [buffer], False)
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = 0.0
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    assert controller.poll_watchdog() is False
    assert event.query_calls == 1
    assert key in controller._launched_at
    assert key in controller._discarded_receives
    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert event.query_calls == 1

    now[0] = 1.0
    assert controller.poll_watchdog() is True
    assert key in controller._discarded_receives
    assert event.query_calls == 1
    assert "one-shot discarded CUDA query failed" in controller.watchdog_reason


def test_mapped_nccl_cuda_terminal_replay_does_not_requery_after_response_loss():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            return None

    class Event:
        query_calls = 0

        def record(self):
            return None

        def query(self):
            self.query_calls += 1
            if self.query_calls == 1:
                return True
            raise RuntimeError("latched CUDA terminal was queried again")

    now = [0.0]
    kv_cache = torch.zeros(2, 1, 2, 1, 1, 1)
    controller = MappedNCCLEndpoint(
        object(), kv_cache, watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    event = Event()
    buffer = torch.ones_like(kv_cache[:, :, 1, :, :, :])
    controller.registry.register(key, resource_kinds=())
    controller.registry.mark_launched(key, [work], event)
    controller._receive_keys.add(key)
    controller._pending_receives[key] = ([work], [(1, buffer)])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._launched_at[key] = 0.0
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }

    # The first terminal response is intentionally ignored to model response
    # loss after the endpoint committed its CUDA visibility proof.
    assert controller.poll_watchdog() is False
    assert key not in controller._launched_at
    assert event.query_calls == 1

    replay = controller.refresh(ref)
    now[0] = 2.0

    assert replay.state == OperationState.COMPLETED
    assert controller.poll_watchdog() is False
    assert controller.termination_requested is False
    assert event.query_calls == 1


def test_mapped_nccl_source_retains_staging_buffers_until_cuda_visibility():
    import torch

    class StagingBuffer:
        def __init__(self, block_id):
            self.block_id = block_id

    class CacheSlice:
        def __init__(self, block_id):
            self.block_id = block_id

        def numel(self):
            return 1

        def contiguous(self):
            return StagingBuffer(self.block_id)

    class Cache:
        shape = (2, 1, 4, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, key):
            block_id = key[2] if len(key) > 2 else 0
            return CacheSlice(block_id)

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 1

    class Work:
        done = False
        waits = 0

        def is_completed(self):
            return self.done

        def wait(self):
            assert self.done
            self.waits += 1

    class Event:
        recorded = False
        visible = False

        def record(self):
            self.recorded = True

        def query(self):
            return self.visible

    work = Work()
    event = Event()
    submitted_buffers = []

    def p2p_op(_op, buffer, **_kwargs):
        submitted_buffers.append(buffer)
        return object()

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(torch.distributed, "P2POp", side_effect=p2p_op), \
         patch.object(
             torch.distributed, "batch_isend_irecv", return_value=[work]
         ), patch.object(torch.cuda, "Event", return_value=event):
        accepted = controller.start_transfer(ref, {
            "source_instance": "d0", "target_instance": "d1",
            "src_block_ids": [1, 2], "kv_size_bytes": 2,
        })

    key = endpoint_ref_key(ref)
    assert accepted.state == OperationState.UNKNOWN
    assert controller._pending_sends[key][1] == submitted_buffers
    assert controller.abort(ref).status.value == "UNKNOWN"
    assert key in controller._pending_sends

    work.done = True
    assert controller.poll_watchdog() is False
    assert event.recorded is True
    assert key in controller._pending_sends

    event.visible = True
    assert controller.poll_watchdog() is False
    assert key not in controller._pending_sends
    assert work.waits == 1


def test_mapped_nccl_source_launch_error_retains_buffers_until_fail_stop():
    import torch

    class StagingBuffer:
        def __init__(self, block_id):
            self.block_id = block_id

    class CacheSlice:
        def __init__(self, block_id):
            self.block_id = block_id

        def numel(self):
            return 1

        def contiguous(self):
            return StagingBuffer(self.block_id)

    class Cache:
        shape = (2, 1, 4, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, key):
            block_id = key[2] if len(key) > 2 else 0
            return CacheSlice(block_id)

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 1

    submitted_buffers = []

    def p2p_op(_op, buffer, **_kwargs):
        submitted_buffers.append(buffer)
        return object()

    now = [0.0]
    controller = MappedNCCLEndpoint(
        PairGroups(), Cache(), watchdog_timeout_s=1.0, clock=lambda: now[0]
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    with patch.object(torch.distributed, "is_initialized", return_value=True), \
         patch.object(torch.distributed, "P2POp", side_effect=p2p_op), \
         patch.object(
             torch.distributed,
             "batch_isend_irecv",
             side_effect=RuntimeError("partial grouped launch"),
         ), patch.object(torch.cuda, "Event", return_value=object()):
        accepted = controller.start_transfer(ref, {
            "source_instance": "d0", "target_instance": "d1",
            "src_block_ids": [1, 2], "kv_size_bytes": 2,
        })

    key = endpoint_ref_key(ref)
    assert accepted.state == OperationState.UNKNOWN
    assert accepted.reason == "RuntimeError: partial grouped launch"
    assert controller._pending_sends[key][1] == submitted_buffers
    controller.prune(set())
    assert key in controller.registry._operations
    assert key in controller._pending_sends

    now[0] = 1.0
    assert controller.poll_watchdog() is True
    assert controller.termination_requested is True
    assert controller.watchdog_evidence["endpoint_key"] == key


def test_watchdog_deadline_precedes_blocked_source_staging():
    import threading
    import torch

    entered = threading.Event()
    release = threading.Event()
    now = [0.0]

    class CacheSlice:
        def numel(self):
            return 1

        def contiguous(self):
            entered.set()
            assert release.wait(timeout=2.0)
            return object()

    class Cache:
        shape = (2, 1, 2, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, _key):
            return CacheSlice()

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 3

    class Work:
        def is_completed(self):
            return False

    class CudaEvent:
        def query(self):
            return False

    controller = MappedNCCLEndpoint(
        PairGroups(), Cache(), watchdog_timeout_s=1.0,
        clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    owner = EngineOwnerCommandQueue(
        lambda _operation, endpoint_ref, payload: controller.start_transfer(
            endpoint_ref, payload
        )
    )
    try:
        with patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            torch.distributed, "P2POp", return_value=object()
        ), patch.object(
            torch.distributed, "batch_isend_irecv", return_value=[Work()]
        ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
            future = owner.submit_future("transfer.start", ref, {
                "source_instance": "d0", "target_instance": "d1",
                "src_block_ids": [1], "kv_size_bytes": 1,
            })
            assert entered.wait(timeout=1.0)
            assert future.done() is False
            assert key in controller._watchdog_deadlines
            assert key not in controller.registry._operations
            assert ref.operation_id not in controller._operation_keys
            assert key not in controller._resource_quantities

            now[0] = 1.0
            assert controller.poll_watchdog_deadline() is True
            assert controller.watchdog_evidence["endpoint_key"] == key

            release.set()
            with pytest.raises(
                RuntimeError,
                match="watchdog expired during source staging",
            ):
                future.result(timeout=1.0)
            assert key in controller._watchdog_deadlines
            assert controller.watchdog_evidence["endpoint_key"] == key
    finally:
        release.set()
        owner.close()


def test_source_staging_exception_leaves_no_registry_or_resource_residue():
    import torch

    class CacheSlice:
        def numel(self):
            return 1

        def contiguous(self):
            raise RuntimeError("source staging failed")

    class Cache:
        shape = (2, 1, 2, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, _key):
            return CacheSlice()

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 3

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    with patch.object(
        torch.distributed, "is_initialized", return_value=True
    ):
        with pytest.raises(RuntimeError, match="source staging failed"):
            controller.start_transfer(ref, {
                "source_instance": "d0", "target_instance": "d1",
                "src_block_ids": [1], "kv_size_bytes": 1,
            })

    assert key not in controller._watchdog_deadlines
    assert key not in controller._launched_at
    assert key not in controller.registry._operations
    assert ref.operation_id not in controller._operation_keys
    assert key not in controller._operation_refs
    assert key not in controller._resource_quantities
    assert key not in controller._transfer_metadata
    assert key not in controller._pending_sends


def test_source_prelaunch_failure_rolls_back_registry_and_side_state():
    import torch

    class CacheSlice:
        def numel(self):
            return 1

        def contiguous(self):
            return object()

    class Cache:
        shape = (2, 1, 2, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, _key):
            return CacheSlice()

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 3

    controller = MappedNCCLEndpoint(PairGroups(), Cache())
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    with patch.object(
        torch.distributed, "is_initialized", return_value=True
    ), patch.object(
        torch.distributed, "P2POp", return_value=object()
    ), patch.object(
        torch.distributed, "batch_isend_irecv"
    ) as grouped_launch, patch.object(
        torch.cuda, "Event", return_value=object()
    ), patch.object(
        controller.registry,
        "mark_launched",
        side_effect=RuntimeError("prelaunch registration failed"),
    ):
        with pytest.raises(RuntimeError, match="prelaunch registration failed"):
            controller.start_transfer(ref, {
                "source_instance": "d0", "target_instance": "d1",
                "src_block_ids": [1], "kv_size_bytes": 1,
            })

    grouped_launch.assert_not_called()
    assert key not in controller._watchdog_deadlines
    assert key not in controller._launched_at
    assert key not in controller.registry._operations
    assert ref.operation_id not in controller._operation_keys
    assert key not in controller._operation_refs
    assert key not in controller._resource_quantities
    assert key not in controller._transfer_metadata
    assert key not in controller._pending_sends


def test_out_of_band_watchdog_expires_while_source_launch_blocks_owner():
    import threading
    import torch

    entered = threading.Event()
    release = threading.Event()
    now = [0.0]

    class CacheSlice:
        def numel(self):
            return 1

        def contiguous(self):
            return object()

    class Cache:
        shape = (2, 1, 2, 1, 1, 1)
        is_cuda = True

        def element_size(self):
            return 1

        def __getitem__(self, _key):
            return CacheSlice()

    class PairGroups:
        def pair(self, pair_id):
            return SimpleNamespace(pair_id=pair_id, process_group=object())

        def global_peer(self, _pair_id):
            return 3

    class Work:
        def is_completed(self):
            return False

    class CudaEvent:
        def query(self):
            return False

    def blocked_launch(_ops):
        entered.set()
        assert release.wait(timeout=2.0)
        return [Work()]

    controller = MappedNCCLEndpoint(
        PairGroups(), Cache(), watchdog_timeout_s=1.0,
        clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    owner = EngineOwnerCommandQueue(
        lambda _operation, endpoint_ref, payload: controller.start_transfer(
            endpoint_ref, payload
        )
    )
    try:
        with patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), patch.object(
            torch.distributed, "P2POp", return_value=object()
        ), patch.object(
            torch.distributed, "batch_isend_irecv", side_effect=blocked_launch
        ), patch.object(torch.cuda, "Event", return_value=CudaEvent()):
            future = owner.submit_future("transfer.start", ref, {
                "source_instance": "d0", "target_instance": "d1",
                "src_block_ids": [1], "kv_size_bytes": 1,
            })
            assert entered.wait(timeout=1.0)
            assert future.done() is False

            now[0] = 1.0
            assert controller.poll_watchdog_deadline() is True
            assert controller.watchdog_evidence["endpoint_key"] == key

            release.set()
            assert future.result(timeout=1.0).state == OperationState.UNKNOWN
    finally:
        release.set()
        owner.close()


def test_out_of_band_watchdog_expires_while_terminal_wait_blocks_owner():
    import threading
    import torch

    entered = threading.Event()
    release = threading.Event()
    now = [0.0]

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            entered.set()
            assert release.wait(timeout=2.0)

    class CudaEvent:
        def record(self):
            return None

        def query(self):
            return True

    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], CudaEvent())
    controller._pending_sends[key] = ([work], [object()])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 1,
    }
    controller._register_watchdog_deadline(
        key, ref, "p0--d0", started_at=0.0
    )
    owner = EngineOwnerCommandQueue(
        lambda _operation, _ref, _payload: controller.poll_terminal_progress()
    )
    try:
        future = owner.submit_future("progress", None, {})
        assert entered.wait(timeout=1.0)
        assert future.done() is False

        now[0] = 1.0
        assert controller.poll_watchdog_deadline() is True
        assert controller.watchdog_evidence["endpoint_key"] == key

        release.set()
        assert future.result(timeout=1.0) is None
        assert key in controller._watchdog_deadlines
        assert controller.watchdog_evidence["endpoint_key"] == key
        assert controller.refresh(ref).state == OperationState.UNKNOWN
        assert key in controller._pending_sends
        assert controller.operation_completed(ref.operation_id, ref) is False
        with pytest.raises(ValueError, match="not terminal"):
            controller.release(
                ref.operation_id, ("SOURCE_RETAIN", "TRANSFER_BYTES")
            )
    finally:
        release.set()
        owner.close()


def test_terminal_deadline_clear_wins_before_out_of_band_expiry():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            return None

    class CudaEvent:
        def record(self):
            return None

        def query(self):
            return True

    now = [0.999]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], CudaEvent())
    controller._pending_sends[key] = ([work], [object()])
    controller._operation_refs[key] = ref
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 1,
    }
    controller._register_watchdog_deadline(
        key, ref, "p0--d0", started_at=0.0
    )

    controller.poll_terminal_progress()
    now[0] = 1.0

    assert key not in controller._watchdog_deadlines
    assert controller.poll_watchdog_deadline() is False
    assert controller.termination_requested is False


def test_terminal_claim_detaches_storage_outside_watchdog_lock():
    import threading
    import torch

    entered_cleanup = threading.Event()
    release_cleanup = threading.Event()
    poll_done = threading.Event()
    lock_acquired_during_del = []
    terminal_results = []
    poll_results = []
    now = [0.0]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)

    class Storage:
        def __del__(self):
            entered_cleanup.set()
            release_cleanup.wait(timeout=2.0)
            acquired = controller._watchdog_lock.acquire(blocking=False)
            lock_acquired_during_del.append(acquired)
            if acquired:
                controller._watchdog_lock.release()

    controller._pending_sends[key] = ([], [Storage()])
    controller._register_watchdog_deadline(
        key, ref, "p0--d0", started_at=0.0
    )
    snapshot = SimpleNamespace(
        status=SimpleNamespace(value="COMPLETED"),
        work_terminal=True,
        cuda_visibility_terminal=True,
    )

    terminal_thread = threading.Thread(
        target=lambda: terminal_results.append(
            controller._finalize_terminal_snapshot(key, snapshot)
        )
    )
    terminal_thread.start()
    assert entered_cleanup.wait(timeout=1.0)
    now[0] = 1.0

    def poll_deadline():
        poll_results.append(controller.poll_watchdog_deadline())
        poll_done.set()

    poll_thread = threading.Thread(target=poll_deadline)
    poll_thread.start()
    poll_completed_while_cleanup_blocked = poll_done.wait(timeout=0.2)
    release_cleanup.set()
    terminal_thread.join(timeout=1.0)
    poll_thread.join(timeout=1.0)

    assert poll_completed_while_cleanup_blocked is True
    assert terminal_results == [True]
    assert poll_results == [False]
    assert lock_acquired_during_del == [True]
    assert controller.termination_requested is False
    assert key not in controller._watchdog_deadlines


def test_refresh_finalizes_second_cuda_query_before_publishing_completed():
    import torch

    class Work:
        def is_completed(self):
            return True

        def wait(self):
            return None

    class CudaEvent:
        recorded = False
        query_calls = 0

        def record(self):
            self.recorded = True

        def query(self):
            self.query_calls += 1
            return self.recorded and self.query_calls >= 2

    now = [0.5]
    controller = MappedNCCLEndpoint(
        object(), torch.zeros(2, 1, 2, 1, 1, 1),
        watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    event = CudaEvent()
    controller.registry.register(
        key, resource_kinds=("SOURCE_RETAIN", "TRANSFER_BYTES")
    )
    controller.registry.mark_launched(key, [work], event)
    controller._pending_sends[key] = ([work], [object()])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._resource_quantities[key] = {
        "SOURCE_RETAIN": 1, "TRANSFER_BYTES": 8,
    }
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }
    controller._register_watchdog_deadline(
        key, ref, "p0--d0", started_at=0.0
    )

    completed = controller.refresh(ref)

    assert completed.state == OperationState.COMPLETED
    assert event.query_calls == 2
    assert key not in controller._watchdog_deadlines
    assert key not in controller._pending_sends
    assert controller.operation_completed(ref.operation_id, ref) is True

    now[0] = 1.5
    assert controller.poll_watchdog_deadline() is False
    assert controller.refresh(ref).state == OperationState.COMPLETED
    assert event.query_calls == 2


def test_receive_buffer_is_retained_until_cuda_event_is_terminal():
    import torch

    class Work:
        waits = 0

        def is_completed(self):
            return True

        def wait(self):
            self.waits += 1

    class CudaEvent:
        recorded = False
        visible = False

        def record(self):
            self.recorded = True

        def query(self):
            return self.visible

    now = [0.0]
    kv_cache = torch.zeros(2, 1, 2, 1, 1, 1)
    controller = MappedNCCLEndpoint(
        object(), kv_cache, watchdog_timeout_s=1.0, clock=lambda: now[0],
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )
    key = endpoint_ref_key(ref)
    work = Work()
    event = CudaEvent()
    buffer = torch.ones_like(kv_cache[:, :, 1, :, :, :])
    controller.registry.register(key, resource_kinds=())
    controller.registry.mark_launched(key, [work], event)
    controller._receive_keys.add(key)
    controller._pending_receives[key] = ([work], [(1, buffer)])
    controller._operation_keys[ref.operation_id] = key
    controller._operation_refs[key] = ref
    controller._transfer_metadata[key] = {
        "pair_id": "p0--d0", "completed_bytes": 8,
    }
    controller._register_watchdog_deadline(
        key, ref, "p0--d0", started_at=0.0
    )

    controller.poll_terminal_progress()

    assert event.recorded is True
    assert work.waits == 1
    assert torch.equal(
        kv_cache[:, :, 1, :, :, :], torch.ones_like(buffer)
    )
    assert key in controller._pending_receives
    assert key in controller._receive_copy_enqueued
    assert key in controller._watchdog_deadlines
    assert controller.refresh(ref).state == OperationState.UNKNOWN
    assert work.waits == 1

    event.visible = True
    controller.poll_terminal_progress()

    assert key not in controller._pending_receives
    assert key not in controller._receive_copy_enqueued
    assert key not in controller._watchdog_deadlines
    assert controller.refresh(ref).state == OperationState.COMPLETED


def test_mapped_nccl_reports_and_releases_real_block_and_byte_quantities():
    import torch

    class PairGroups:
        def pair(self, pair_id):
            return type("Pair", (), {"pair_id": pair_id, "process_group": None})()

    controller = MappedNCCLEndpoint(
        PairGroups(), torch.zeros(2, 1, 4, 1, 1, 2)
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )
    controller.start_transfer(ref, {
        "source_instance": "p0", "target_instance": "d0",
        "src_block_ids": [1, 2], "kv_size_bytes": 32,
    })

    assert controller.resource_quantities() == {
        "SOURCE_RETAIN": 2, "TRANSFER_BYTES": 32,
    }
    assert controller.release(
        "op", ("SOURCE_RETAIN", "TRANSFER_BYTES")
    ) == {"SOURCE_RETAIN": 2, "TRANSFER_BYTES": 32}
    assert controller.resource_quantities() == {}


def test_mapped_nccl_rejects_declared_bytes_that_do_not_match_kv_slices():
    import torch

    class PairGroups:
        def pair(self, pair_id):
            return type("Pair", (), {"pair_id": pair_id, "process_group": None})()

    controller = MappedNCCLEndpoint(
        PairGroups(), torch.zeros(2, 1, 4, 1, 1, 2)
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "op", "sha256:x"
    )

    with pytest.raises(ValueError, match="declared transfer bytes"):
        controller.start_transfer(ref, {
            "source_instance": "p0",
            "target_instance": "d0",
            "src_block_ids": [1, 2],
            "kv_size_bytes": 33,
        })

    assert controller.resource_quantities() == {}
    assert controller.registry._operations == {}
    assert controller._operation_keys == {}


@pytest.mark.parametrize(
    ("src_block_ids", "dst_block_ids"),
    [
        ([1], [2, 3]),
        ([1, 1], [2, 3]),
        ([1, 2], [3, 3]),
    ],
)
def test_mapped_nccl_rejects_non_bijective_block_mapping_before_registration(
    src_block_ids, dst_block_ids,
):
    import torch

    class PairGroups:
        def pair(self, pair_id):
            return type("Pair", (), {"pair_id": pair_id, "process_group": None})()

    controller = MappedNCCLEndpoint(
        PairGroups(), torch.zeros(2, 1, 4, 1, 1, 2)
    )
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:x"
    )

    with pytest.raises(ValueError, match="bijective"):
        controller.prepare_receive(ref, {
            "source_instance": "p0",
            "target_instance": "d0",
            "src_block_ids": src_block_ids,
            "dst_block_ids": dst_block_ids,
        })

    assert controller.registry._operations == {}
    assert controller._operation_keys == {}


def test_terminal_eviction_bounds_all_worker_side_registries():
    import torch

    from prism_infer.engine.prefix_cache import PrefixCacheService
    from prism_infer.server.app import WorkerControlRuntime
    from prism_infer.server.operation_registry import OperationRegistry

    registry = OperationRegistry(
        instance_id="d0", instance_epoch="epoch", topology_generation="world",
        terminal_snapshot_cap=2,
    )
    registry.activate_owner("owner")
    mapped = MappedNCCLEndpoint(object(), torch.zeros(2, 1, 2, 1, 1, 1))
    prefix = SimpleNamespace(_operations={})
    driver = PDExecutionDriver(SimpleNamespace(), role="decode")

    def prune(refs, operation_ids):
        mapped.prune(refs)
        PrefixCacheService.prune_operations(prefix, operation_ids)
        driver.prune(operation_ids)

    runtime = WorkerControlRuntime(
        identity={}, capabilities={}, registry=registry, prune_handler=prune,
    )
    for seq in range(1, 11):
        ref = EndpointOperationRef(
            "world", "owner", seq, "d0", "epoch", f"op-{seq}", f"sha256:{seq}"
        )
        key = endpoint_ref_key(ref)
        mapped.registry.register(key, resource_kinds=())
        mapped._operation_keys[ref.operation_id] = key
        prefix._operations[ref.operation_id] = object()
        sequence = SimpleNamespace()
        driver._operations[ref.operation_id] = sequence
        driver._requests[f"request-{seq}"] = sequence
        runtime.operation_kinds[ref] = "transfer.prepare_receive"
        registry.accept(
            ref, lambda ref=ref: OperationSnapshot(ref, OperationState.COMPLETED)
        )
        runtime.prune_side_state()

    assert len(registry.snapshots()) == 2
    assert len(runtime.operation_kinds) == 2
    assert len(mapped.registry._operations) == 2
    assert len(mapped._operation_keys) == 2
    assert len(prefix._operations) == 2
    assert len(driver._operations) == 2
    assert len(driver._requests) == 2


@pytest.mark.parametrize("terminal_kind", ["max_tokens", "eos"])
def test_cold_commit_first_token_terminal_publishes_once_without_decode(
    monkeypatch, terminal_kind,
):
    from prism_infer.engine.prefix_cache import PrefixCacheService
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import Sequence
    from prism_infer.sampling_params import SamplingParams

    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=8, eos=9,
        kvcache_block_size=4, num_kvcache_blocks=8,
    ))
    engine = SimpleNamespace(scheduler=scheduler)
    engine.prefix_cache = PrefixCacheService(scheduler.block_manager)
    prepared = engine.prefix_cache.prepare(
        "op", "request", mode="remote_transfer", block_count=1,
        token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(
            max_tokens=1 if terminal_kind == "max_tokens" else 4,
            ignore_eos=False,
        ),
    )
    assert prepared is not None
    driver = PDExecutionDriver(engine, role="decode")
    control = EngineControlRouter(
        engine, transfer_terminal=lambda operation_id, ref=None: True,
        request_committed=driver.request_committed,
    )
    transfer_ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "transfer", "sha256:t"
    )
    commit_ref = EndpointOperationRef(
        "world", "owner", 2, "d0", "epoch", "op", "sha256:c"
    )
    first_token = 9 if terminal_kind == "eos" else 7
    snapshot = control("request.commit", commit_ref, {
        "req_id": "request", "operation_id": "op",
        "transfer_operation_id": "transfer",
        "transfer_endpoint_ref": transfer_ref.__dict__,
        "first_token": first_token, "cached_prefix_tokens": 4,
        "first_token_subject": "first", "decode_progress_subject": "progress",
        "decode_done_subject": "done",
    })
    sequence = prepared.sequence

    assert snapshot.state == OperationState.COMPLETED
    assert sequence is not None and sequence.is_finished
    assert sequence.completion_token_ids == [first_token]
    assert sequence not in scheduler.running and scheduler.is_finished()
    events = [driver.events.get_nowait() for _ in range(3)]
    assert [subject for subject, _ in events] == ["first", "progress", "done"]
    assert all(value["token_ids"] == [first_token] for _, value in events)
    driver.idle_step()
    assert driver.events.empty()


@pytest.mark.parametrize("prefix_path", ["same", "mapped"])
@pytest.mark.parametrize("terminal_token", [7, 9])
def test_suffix_first_token_terminal_same_and_mapped_paths_publish_once(
    monkeypatch, prefix_path, terminal_token,
):
    from prism_infer.engine.sequence import SequenceStatus

    class Sequence:
        num_cached_tokens = 4
        num_prompt_tokens = 5
        completion_token_ids = []
        block_table = [0, 1]
        status = SequenceStatus.WAITING

        @property
        def is_finished(self):
            return self.status == SequenceStatus.FINISHED

    sequence = Sequence()
    scheduler = SimpleNamespace(
        waiting=deque([sequence]), running=deque(),
        add=lambda value: scheduler.waiting.append(value),
        is_finished=lambda: not scheduler.waiting and not scheduler.running,
    )

    def step():
        sequence.num_cached_tokens = sequence.num_prompt_tokens
        sequence.completion_token_ids = [terminal_token]
        sequence.status = SequenceStatus.FINISHED
        scheduler.waiting.remove(sequence)

    engine = SimpleNamespace(
        scheduler=scheduler, step=step, model_runner=SimpleNamespace(kv_cache=None),
        prefix_cache=SimpleNamespace(
            instance_epoch="d0-epoch",
            _operations={"op": SimpleNamespace(
                sequence=sequence,
                mode="local_reuse" if prefix_path == "same" else "remote_transfer",
            )},
        ),
    )
    driver = PDExecutionDriver(engine, role="decode")
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "op", "sha256:s"
    )
    driver("suffix", ref, {
        "req_id": "request", "first_token_subject": "first",
        "decode_progress_subject": "progress", "decode_done_subject": "done",
    })

    assert sequence.is_finished and scheduler.is_finished()
    events = [driver.events.get_nowait() for _ in range(3)]
    assert [subject for subject, _ in events] == ["first", "progress", "done"]
    driver.idle_step()
    assert driver.events.empty()


def test_decode_idle_loop_publishes_cumulative_progress_and_terminal_events():
    class Sequence:
        completion_token_ids = []
        is_finished = False

    sequence = Sequence()

    class Scheduler:
        def is_finished(self):
            return False

    class Engine:
        scheduler = Scheduler()
        prefix_cache = type("Prefix", (), {"instance_epoch": "d0-epoch"})()

        def step(self):
            sequence.completion_token_ids = [7]
            sequence.is_finished = True

    driver = PDExecutionDriver(Engine(), role="decode")
    driver.request_committed("r1", "op-1", sequence, {
        "operation_id": "op-1", "first_token_subject": "first_token.owner",
        "decode_progress_subject": "decode_progress.owner",
        "decode_done_subject": "decode_done.owner",
    })
    driver.idle_step()
    events = [driver.events.get_nowait() for _ in range(3)]
    assert [subject for subject, _ in events] == [
        "first_token.owner", "decode_progress.owner", "decode_done.owner"
    ]
    assert all(event["token_ids"] == [7] for _, event in events)


def test_idle_step_exception_aborts_sequence_and_releases_resources(monkeypatch):
    from prism_infer.engine.prefix_cache import PrefixCacheService
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import Sequence, SequenceStatus
    from prism_infer.sampling_params import SamplingParams

    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=8, eos=999,
        kvcache_block_size=4, num_kvcache_blocks=4,
    ))
    engine = SimpleNamespace(scheduler=scheduler)
    engine.prefix_cache = PrefixCacheService(scheduler.block_manager)
    prepared = engine.prefix_cache.prepare(
        "target-op", "request-1", mode="remote_transfer", block_count=2,
        token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=4, ignore_eos=True),
    )
    assert prepared is not None
    sequence = engine.prefix_cache.commit(
        "target-op", namespace="", kv_compatibility_id="",
        request_context_digest="", cached_prefix_tokens=5,
        transfer_proven=True,
    )
    sequence.status = SequenceStatus.RUNNING
    scheduler.running.append(sequence)

    def fail_step():
        raise IndexError("block_table[2]")

    engine.step = fail_step
    driver = PDExecutionDriver(engine, role="decode")
    driver.request_committed("request-1", "target-op", sequence, {})

    with pytest.raises(IndexError, match=r"block_table\[2\]"):
        driver.idle_step()

    assert sequence.status == SequenceStatus.ABORTED
    assert sequence not in scheduler.waiting
    assert sequence not in scheduler.running
    assert driver.resource_details()["active_request_ids"] == []
    assert len(sequence.block_table) == 2
    assert set(sequence.block_table) <= scheduler.block_manager.used_block_ids

    assert engine.prefix_cache.finalize_release(
        "target-op", ("TARGET_SEQUENCE",)
    ) == {"TARGET_SEQUENCE": 2}
    assert sequence.block_table == []
    assert scheduler.block_manager.used_block_ids == set()


def test_idle_step_batch_exception_aborts_all_active_sequences_without_freeing_blocks():
    from prism_infer.engine.sequence import SequenceStatus

    sequence_a = SimpleNamespace(
        completion_token_ids=[], block_table=[3, 4],
        status=SequenceStatus.RUNNING, is_finished=False,
    )
    sequence_b = SimpleNamespace(
        completion_token_ids=[], block_table=[7],
        status=SequenceStatus.WAITING, is_finished=False,
    )
    scheduler = SimpleNamespace(
        waiting=deque([sequence_b]), running=deque([sequence_a]),
        is_finished=lambda: False,
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        prefix_cache=SimpleNamespace(instance_epoch="d0-epoch"),
        step=lambda: (_ for _ in ()).throw(RuntimeError("batched decode failed")),
    )
    driver = PDExecutionDriver(engine, role="decode")
    driver.request_committed("request-a", "operation-a", sequence_a, {})
    driver.request_committed("request-b", "operation-b", sequence_b, {})

    with pytest.raises(RuntimeError, match="batched decode failed"):
        driver.idle_step()

    assert sequence_a.status == SequenceStatus.ABORTED
    assert sequence_b.status == SequenceStatus.ABORTED
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == []
    assert driver._requests == {}
    assert driver._operations == {
        "operation-a": sequence_a,
        "operation-b": sequence_b,
    }
    assert sequence_a.block_table == [3, 4]
    assert sequence_b.block_table == [7]


def test_suffix_step_exception_aborts_exact_sequence_before_rethrow():
    from prism_infer.engine.sequence import SequenceStatus

    sequence = SimpleNamespace(
        num_cached_tokens=4, num_prompt_tokens=5,
        completion_token_ids=[], block_table=[0, 1],
        status=SequenceStatus.WAITING, is_finished=False,
    )
    scheduler = SimpleNamespace(
        waiting=deque(), running=deque(),
        add=lambda value: scheduler.waiting.append(value),
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        prefix_cache=SimpleNamespace(
            instance_epoch="d0-epoch",
            _operations={"suffix-op": SimpleNamespace(sequence=sequence)},
        ),
        model_runner=SimpleNamespace(kv_cache=None),
        step=lambda: (_ for _ in ()).throw(IndexError("block_table[2]")),
    )
    driver = PDExecutionDriver(engine, role="decode")
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "suffix-op", "sha256:x"
    )

    with pytest.raises(IndexError, match=r"block_table\[2\]"):
        driver("suffix", ref, {"req_id": "request-1"})

    assert sequence.status == SequenceStatus.ABORTED
    assert sequence not in scheduler.waiting
    assert sequence not in scheduler.running
    assert driver._requests == {}
    assert sequence.block_table == [0, 1]


def test_request_commit_prune_uses_commit_endpoint_operation_identity():
    """Transfer cleanup cannot remove output while its commit ref is live."""
    from prism_infer.engine.sequence import SequenceStatus

    class Sequence:
        defer_deallocation = False
        num_cached_tokens = 0
        num_prompt_tokens = 4
        num_completion_tokens = 0
        completion_token_ids = []
        ignore_eos = True
        max_tokens = 2
        is_prefill = True
        status = SequenceStatus.WAITING
        seq_id = 1
        block_table = [0]

        def append_token(self, token):
            self.completion_token_ids = [token]
            self.num_completion_tokens = 1

        @property
        def is_finished(self):
            return self.status == SequenceStatus.FINISHED

    sequence = Sequence()

    class PrefixCache:
        instance_epoch = "d1-epoch"

        def commit(self, operation_id, **_kwargs):
            assert operation_id == "commit-operation"
            return sequence

    scheduler = SimpleNamespace(
        waiting=[sequence], running=[], eos=9,
    )
    engine = SimpleNamespace(
        scheduler=scheduler, prefix_cache=PrefixCache(),
    )
    driver = PDExecutionDriver(engine, role="decode")
    control = EngineControlRouter(
        engine,
        transfer_terminal=lambda _operation_id, _ref=None: True,
        request_committed=driver.request_committed,
    )
    transfer_ref = EndpointOperationRef(
        "world", "owner", 1, "d1", "epoch", "transfer-operation", "sha256:t"
    )
    commit_ref = EndpointOperationRef(
        "world", "owner", 2, "d1", "epoch", "commit-operation", "sha256:c"
    )

    control("request.commit", commit_ref, {
        "req_id": "request-1",
        "operation_id": "transfer-operation",
        "transfer_operation_id": "transfer-operation",
        "transfer_endpoint_ref": transfer_ref.__dict__,
        "first_token": 7,
        "cached_prefix_tokens": 4,
    })

    assert driver._operations == {"commit-operation": sequence}
    driver.prune({"commit-operation"})
    output = driver.output("request-1", 0)
    assert output["token_ids"] == [7]
    assert output["operation_id"] == "transfer-operation"


def test_suffix_abort_uses_operation_identity_removes_sequence_and_owns_no_blocks():
    from prism_infer.engine.sequence import SequenceStatus

    sequence = SimpleNamespace(
        num_cached_tokens=4, num_prompt_tokens=4,
        completion_token_ids=[], block_table=[3], status=SequenceStatus.RUNNING,
        is_finished=False,
    )
    scheduler = SimpleNamespace(
        waiting=deque([sequence]), running=deque(),
        add=lambda value: scheduler.waiting.append(value),
        is_finished=lambda: not scheduler.waiting and not scheduler.running,
        block_manager=SimpleNamespace(blocks=[object()], used_block_ids={3}),
        max_num_seqs=1,
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        prefix_cache=SimpleNamespace(
            _operations={"suffix-op": SimpleNamespace(sequence=sequence)},
            instance_epoch="d0-epoch",
        ),
        model_runner=SimpleNamespace(kv_cache=None),
        step=lambda: (_ for _ in ()).throw(AssertionError("suffix already prefetched")),
    )
    driver = PDExecutionDriver(engine, role="decode")
    ref = EndpointOperationRef(
        "world", "owner", 1, "d0", "epoch", "suffix-op", "sha256:x"
    )

    snapshot = driver("suffix", ref, {
        "req_id": "request-1", "first_token_subject": "first.owner",
        "decode_progress_subject": "progress.owner",
        "decode_done_subject": "done.owner",
    })
    assert snapshot.state == OperationState.COMPLETED
    assert snapshot.resources_held is False
    assert snapshot.held_resource_kinds == ()

    assert driver.abort_request("suffix-op") is True
    assert sequence.status == SequenceStatus.ABORTED
    assert sequence not in scheduler.waiting
    assert sequence not in scheduler.running
    driver.idle_step()
    assert driver.events.empty()


def test_prefill_failure_before_allocation_has_zero_count_finalize():
    scheduler = SimpleNamespace(
        waiting=deque(), running=deque(),
        block_manager=SimpleNamespace(
            deallocate=lambda _sequence: (_ for _ in ()).throw(
                AssertionError("zero-count release must not deallocate")
            )
        ),
    )
    driver = PDExecutionDriver(
        SimpleNamespace(scheduler=scheduler), role="prefill"
    )

    assert driver.release_source_blocks("failed-before-sequence") == {
        "SOURCE_BLOCKS": 0
    }


def test_prefill_max_tokens_one_retains_source_blocks_and_stops_after_one_step(
    monkeypatch,
):
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import Sequence, SequenceStatus

    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=8, eos=9,
        kvcache_block_size=4, num_kvcache_blocks=8,
    ))

    class Engine:
        model_runner = SimpleNamespace(kv_cache=None)
        prefix_cache = SimpleNamespace(instance_epoch="p0-epoch")

        def __init__(self):
            self.scheduler = scheduler
            self.step_count = 0

        def add_request(self, token_ids, sampling_params):
            self.scheduler.add(Sequence(token_ids, sampling_params))

        def step(self):
            self.step_count += 1
            if self.step_count > 1:
                raise AssertionError("terminal seed prefill must not spin")
            seqs, is_prefill = self.scheduler.schedule()
            assert len(seqs) == 1 and is_prefill is True
            self.scheduler.postprocess(seqs, [7], is_prefill)

    engine = Engine()
    driver = PDExecutionDriver(engine, role="prefill")
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "seed", "sha256:x"
    )
    snapshot = driver("prefill", ref, {
        "req_id": "seed-request", "token_ids": [1, 2, 3, 4],
        "sampling_params": {"max_tokens": 1, "ignore_eos": True},
    })
    sequence = driver._requests["seed-request"]

    assert engine.step_count == 1
    assert sequence.defer_deallocation is True
    assert sequence.num_cached_tokens == sequence.num_prompt_tokens == 4
    assert sequence.status == SequenceStatus.KV_TRANSFERRING
    assert len(sequence.block_table) == 1
    assert snapshot.state == OperationState.COMPLETED
    assert snapshot.held_resource_kinds == ("SOURCE_BLOCKS",)
    assert snapshot.result["first_token"] == 7


def test_prefill_without_scheduler_progress_fails_instead_of_spinning(monkeypatch):
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import Sequence

    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=8, eos=9,
        kvcache_block_size=4, num_kvcache_blocks=0,
    ))

    class Engine:
        model_runner = SimpleNamespace(kv_cache=None)
        prefix_cache = SimpleNamespace(instance_epoch="p0-epoch")

        def __init__(self):
            self.scheduler = scheduler
            self.step_count = 0

        def add_request(self, token_ids, sampling_params):
            self.scheduler.add(Sequence(token_ids, sampling_params))

        def step(self):
            self.step_count += 1
            assert self.scheduler.schedule() == ([], False)

    engine = Engine()
    driver = PDExecutionDriver(engine, role="prefill")
    ref = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "oom", "sha256:x"
    )

    with pytest.raises(RuntimeError, match="prefill engine made no progress"):
        driver("prefill", ref, {
            "req_id": "oom-request", "token_ids": [1, 2, 3, 4],
            "sampling_params": {"max_tokens": 1, "ignore_eos": True},
        })

    assert engine.step_count == 1
    assert "oom-request" not in driver._requests
    assert driver._operations["oom"].defer_deallocation is True


def test_prefill_no_progress_fences_sequence_until_generic_finalize(monkeypatch):
    from prism_infer.engine.scheduler import Scheduler
    from prism_infer.engine.sequence import Sequence, SequenceStatus

    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(SimpleNamespace(
        max_num_seqs=2, max_num_batched_tokens=8, eos=9,
        kvcache_block_size=4, num_kvcache_blocks=2,
    ))

    class Engine:
        model_runner = SimpleNamespace(kv_cache=None)
        prefix_cache = SimpleNamespace(instance_epoch="p0-epoch")

        def __init__(self):
            self.scheduler = scheduler
            self.scheduled = []

        def add_request(self, token_ids, sampling_params):
            self.scheduler.add(Sequence(token_ids, sampling_params))

        def step(self):
            seqs, is_prefill = self.scheduler.schedule()
            self.scheduled.append(list(seqs))
            assert len(seqs) == 1 and is_prefill is True
            if len(self.scheduled) == 1:
    # The owner step makes no progress after allocating the sequence block.
                return
            self.scheduler.postprocess(seqs, [7], is_prefill)

    engine = Engine()
    driver = PDExecutionDriver(engine, role="prefill")
    ref_a = EndpointOperationRef(
        "world", "owner", 1, "p0", "epoch", "prefill-a", "sha256:a"
    )

    with pytest.raises(RuntimeError, match="prefill engine made no progress"):
        driver("prefill", ref_a, {
            "req_id": "request-a", "token_ids": [1, 2, 3, 4],
            "sampling_params": {"max_tokens": 1, "ignore_eos": True},
        })

    assert "request-a" not in driver._requests
    sequence_a = driver._operations["prefill-a"]
    assert sequence_a.status == SequenceStatus.ABORTED
    assert sequence_a not in scheduler.waiting
    assert sequence_a not in scheduler.running
    assert len(sequence_a.block_table) == 1
    assert set(sequence_a.block_table) <= scheduler.block_manager.used_block_ids

    ref_b = EndpointOperationRef(
        "world", "owner", 2, "p0", "epoch", "prefill-b", "sha256:b"
    )
    snapshot_b = driver("prefill", ref_b, {
        "req_id": "request-b", "token_ids": [5, 6, 7, 8],
        "sampling_params": {"max_tokens": 1, "ignore_eos": True},
    })
    sequence_b = driver._requests["request-b"]

    assert engine.scheduled[1] == [sequence_b]
    assert sequence_a not in engine.scheduled[1]
    assert snapshot_b.state == OperationState.COMPLETED
    assert len(scheduler.block_manager.used_block_ids) == 2

    assert driver.release_source_blocks("prefill-a") == {"SOURCE_BLOCKS": 1}
    assert sequence_a.block_table == []
    assert len(scheduler.block_manager.used_block_ids) == 1
    assert driver.release_source_blocks("prefill-b") == {"SOURCE_BLOCKS": 1}
    assert len(scheduler.block_manager.used_block_ids) == 0


def test_prefill_failure_with_empty_sequence_finalizes_and_removes_owner_state():
    from prism_infer.engine.sequence import SequenceStatus

    sequence = SimpleNamespace(block_table=[], status=SequenceStatus.WAITING)
    scheduler = SimpleNamespace(
        waiting=deque([sequence]), running=deque(),
        block_manager=SimpleNamespace(
            deallocate=lambda _sequence: (_ for _ in ()).throw(
                AssertionError("empty sequence must not deallocate")
            )
        ),
    )
    driver = PDExecutionDriver(
        SimpleNamespace(scheduler=scheduler), role="prefill"
    )
    driver._operations["prefill-op"] = sequence
    driver._requests["request-1"] = sequence

    assert driver.release_source_blocks("prefill-op") == {"SOURCE_BLOCKS": 0}
    assert sequence.status == SequenceStatus.ABORTED
    assert list(scheduler.waiting) == []
    assert driver._operations == {}
    assert driver._requests == {}
