"""Cached-prefix transaction owner for one infer process."""

from dataclasses import dataclass
from enum import Enum
import time

import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.kv_snapshot import pre_alloc_blocks
from prism_infer.engine.kv_transfer import (
    MappedPrefixTransferReq,
    MappedTransferRegistry,
    MappedTransferStatus,
)
from prism_infer.engine.sequence import Sequence, SequenceStatus
from prism_infer.sampling_params import SamplingParams


class PrefixOperationStatus(str, Enum):
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class PrefixOperation:
    operation_id: str
    req_id: str
    mode: str
    token_ids: list[int]
    sampling_params: SamplingParams
    dst_block_ids: tuple[int, ...]
    status: PrefixOperationStatus = PrefixOperationStatus.PREPARED
    sequence: Sequence | None = None
    created_at: float = 0.0
    resources_held: bool = True


class PrefixCacheService:
    """Own pending target blocks and commit them exactly once by operation id."""

    def __init__(self, block_manager: BlockManager, kv_cache: torch.Tensor | None = None):
        self.block_manager = block_manager
        self.kv_cache = kv_cache
        self.transfers = MappedTransferRegistry()
        self._operations: dict[str, PrefixOperation] = {}

    @property
    def instance_id(self) -> str:
        return self.block_manager.instance_id

    @property
    def instance_epoch(self) -> str:
        return self.block_manager.instance_epoch

    def resolve_prefix(
        self,
        operation_id: str,
        expected_blocks: list[tuple[int, list[int]]],
        *,
        namespace: str,
        kv_compatibility_id: str,
        request_context_digest: str,
    ) -> tuple[int, ...] | None:
        return self.block_manager.resolve_and_pin_prefix(
            operation_id, expected_blocks,
            namespace=namespace,
            kv_compatibility_id=kv_compatibility_id,
            request_context_digest=request_context_digest,
        )

    def prepare(
        self,
        operation_id: str,
        req_id: str,
        *,
        mode: str,
        block_count: int,
        token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> PrefixOperation | None:
        assert mode in {"local_reuse", "remote_transfer"}, f"invalid mode: {mode!r}"
        existing = self._operations.get(operation_id)
        if existing is not None:
            if existing.req_id != req_id or existing.mode != mode:
                raise ValueError("operation id reused with different prefix prepare")
            return existing
        if mode == "local_reuse":
            pinned = self.block_manager._transfer_pins.get(operation_id)
            if pinned is None or len(pinned) != block_count:
                return None
            dst_blocks: tuple[int, ...] = ()
        else:
            allocated = pre_alloc_blocks(
                req_id, block_count, token_ids, self.block_manager
            )
            if allocated is None:
                return None
            dst_blocks = tuple(allocated)
        operation = PrefixOperation(
            operation_id, req_id, mode, list(token_ids), sampling_params,
            dst_blocks, created_at=time.monotonic(),
        )
        self._operations[operation_id] = operation
        return operation

    def transfer_from(
        self,
        source: "PrefixCacheService",
        request: MappedPrefixTransferReq,
    ) -> MappedTransferStatus:
        assert request.src_instance == source.instance_id
        assert request.src_instance_epoch == source.instance_epoch
        assert request.dst_instance == self.instance_id
        assert request.dst_instance_epoch == self.instance_epoch
        operation = self._operations.get(request.op_id)
        if operation is None or operation.dst_block_ids != request.dst_block_ids:
            raise ValueError("mapped transfer target was not prepared")
        self.transfers.prepare(request)
        self.transfers.mark_running(request.op_id)
        if source.kv_cache is None or self.kv_cache is None:
            self.transfers.abort_result(
                request.op_id, source_fenced=False, target_fenced=False
            )
            return MappedTransferStatus.UNKNOWN
        with torch.no_grad():
            for src, dst in zip(request.src_block_ids, request.dst_block_ids):
                self.kv_cache[:, :, dst].copy_(source.kv_cache[:, :, src])
        self.transfers.mark_completed(request.op_id)
        return MappedTransferStatus.COMPLETED

    def commit(
        self,
        operation_id: str,
        *,
        namespace: str,
        kv_compatibility_id: str,
        request_context_digest: str,
        cached_prefix_tokens: int,
        transfer_proven: bool = False,
    ) -> Sequence:
        operation = self._operations[operation_id]
        if operation.status == PrefixOperationStatus.COMMITTED:
            assert operation.sequence is not None
            return operation.sequence
        if operation.status != PrefixOperationStatus.PREPARED:
            raise ValueError(f"cannot commit prefix operation: {operation.status}")
        if operation.mode == "local_reuse":
            blocks = self.block_manager.commit_pinned_prefix(operation_id)
        else:
            if not transfer_proven and self.transfers.status(operation_id) != MappedTransferStatus.COMPLETED:
                raise ValueError("mapped transfer is not complete")
            blocks = operation.dst_block_ids
            # The final request block may be partial.  It owns real KV bytes
            # for decode but cannot become a reusable prefix location until a
            # complete token block exists.
            reusable = min(
                len(blocks), len(operation.token_ids) // self.block_manager.block_size
            )
            self.block_manager.install_prefix_metadata(
                blocks[:reusable], operation.token_ids,
                namespace=namespace,
                kv_compatibility_id=kv_compatibility_id,
                request_context_digest=request_context_digest,
            )
        sequence = Sequence(operation.token_ids, operation.sampling_params)
        sequence.defer_deallocation = True
        sequence.block_table = list(blocks)
        sequence.num_cached_tokens = cached_prefix_tokens
        sequence.status = SequenceStatus.WAITING
        operation.sequence = sequence
        operation.status = PrefixOperationStatus.COMMITTED
        return sequence

    def abort(self, operation_id: str) -> PrefixOperationStatus:
        operation = self._operations.get(operation_id)
        if operation is None:
            return PrefixOperationStatus.UNKNOWN
        if operation.status == PrefixOperationStatus.COMMITTED:
            return operation.status
        if operation.status == PrefixOperationStatus.ABORTED:
            return operation.status
        # Abort fences the writer but retains owned resources for generic finalize.
        operation.status = PrefixOperationStatus.ABORTED
        return operation.status

    def status(self, operation_id: str) -> PrefixOperationStatus:
        operation = self._operations.get(operation_id)
        return operation.status if operation is not None else PrefixOperationStatus.UNKNOWN

    def unpin(self, operation_id: str) -> bool:
        return self.block_manager.unpin_prefix(operation_id)

    def resource_counts(self) -> dict[str, int]:
        with self.block_manager._prefix_state_lock:
            pins = len(self.block_manager._transfer_pins)
        pending = sum(
            operation.resources_held
            and operation.mode == "remote_transfer"
            and operation.sequence is None
            and bool(operation.dst_block_ids)
            for operation in self._operations.values()
        )
        return {"transfer_pins": pins, "pending_allocations": pending}

    def expire_unstarted(self, timeout_s: float, now: float | None = None) -> list[str]:
        """Abort PREPARED operations that never handed data to the transport."""
        current = time.monotonic() if now is None else now
        expired = []
        for operation_id, operation in list(self._operations.items()):
            if operation.status != PrefixOperationStatus.PREPARED:
                continue
            if self.transfers.contains(operation_id):
                continue
            if current - operation.created_at <= timeout_s:
                continue
            self.abort(operation_id)
            expired.append(operation_id)
        return expired

    def abort_sequence(self, operation_id: str) -> bool:
        """Fence the local Sequence and release committed block refs."""
        operation = self._operations.get(operation_id)
        if operation is None:
            return False
        if operation.status == PrefixOperationStatus.ABORTED:
            return True
        if operation.status != PrefixOperationStatus.COMMITTED \
                or operation.sequence is None:
            return False
        operation.sequence.status = SequenceStatus.ABORTED
        operation.status = PrefixOperationStatus.ABORTED
        return True

    def finalize_release(
        self, operation_id: str, resource_kinds: tuple[str, ...]
    ) -> dict[str, int]:
        """Release held resources after the first successful generic finalize."""
        operation = self._operations.get(operation_id)
        requested = set(resource_kinds)
        released: dict[str, int] = {}
        if operation is None:
            if requested == {"SOURCE_PIN"}:
                released["SOURCE_PIN"] = int(
                    self.block_manager.unpin_prefix(operation_id)
                )
                return released
            raise ValueError("unknown prefix operation")
        if operation.status not in {
            PrefixOperationStatus.ABORTED, PrefixOperationStatus.COMMITTED,
        }:
            raise ValueError("prefix operation is not terminal")
        if not operation.resources_held:
            raise ValueError("prefix resources already released")
        if operation.mode == "remote_transfer" and operation.sequence is None:
            if requested != {"TARGET_PENDING"}:
                raise ValueError("pending prefix owns TARGET_PENDING only")
            for block_id in operation.dst_block_ids:
                self.block_manager.release_block(block_id)
            released["TARGET_PENDING"] = len(operation.dst_block_ids)
        elif operation.sequence is not None:
            if requested != {"TARGET_SEQUENCE"}:
                raise ValueError("committed prefix owns TARGET_SEQUENCE only")
            block_count = len(operation.sequence.block_table)
            self.block_manager.deallocate(operation.sequence)
            operation.sequence.defer_deallocation = False
            released["TARGET_SEQUENCE"] = block_count
        else:
            if requested != {"SOURCE_PIN"}:
                raise ValueError("local prefix owns SOURCE_PIN only")
            released["SOURCE_PIN"] = int(
                self.block_manager.unpin_prefix(operation_id)
            )
        operation.resources_held = False
        return released

    def prune_operations(self, operation_ids: set[str]) -> None:
        self._operations = {
            operation_id: operation
            for operation_id, operation in self._operations.items()
            if operation_id in operation_ids
        }
