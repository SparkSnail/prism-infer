import pytest

from prism_infer.engine.kv_transfer import (
    MappedPrefixTransferReq,
    MappedTransferRegistry,
    MappedTransferStatus,
)


def _request(op_id="op1"):
    return MappedPrefixTransferReq(
        op_id, "r1", "s0", "se", "d0", "de", (1, 2), (8, 9),
        "ns", "compat", "text",
    )


def test_mapping_requires_equal_source_and_target_blocks():
    with pytest.raises(AssertionError):
        MappedPrefixTransferReq(
            "op", "r", "s", "se", "d", "de", (1,), (2, 3),
            "ns", "compat", "text",
        )


def test_registry_is_idempotent_and_rejects_operation_reuse():
    registry = MappedTransferRegistry()
    request = _request()
    assert registry.prepare(request) is registry.prepare(request)
    with pytest.raises(ValueError):
        registry.prepare(_request("op1").__class__(
            "op1", "different", "s0", "se", "d0", "de", (1, 2), (8, 9),
            "ns", "compat", "text",
        ))


def test_abort_requires_both_endpoint_fences():
    registry = MappedTransferRegistry()
    registry.prepare(_request())
    registry.mark_running("op1")
    assert registry.abort_result("op1", source_fenced=True, target_fenced=False) == MappedTransferStatus.UNKNOWN


def test_completion_racing_abort_returns_completed():
    registry = MappedTransferRegistry()
    registry.prepare(_request())
    registry.mark_running("op1")
    registry.mark_completed("op1")
    assert registry.abort_result("op1", source_fenced=False, target_fenced=False) == MappedTransferStatus.COMPLETED
