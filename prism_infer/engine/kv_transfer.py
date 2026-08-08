from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

import torch


@dataclass
class TransferReq:
    """KV transfer instruction from serve (interface contract §03)."""
    op_id: str               # idempotent id, safe to retry on failure
    seq_id: str
    src_instance: str
    dst_instance: str
    block_table: list[int]   # all block ids for this sequence
    block_hint: list[int]    # block ids already present on D side (delta transfer)
    priority: int = 1
    on_fail: str = "recompute"
    # Fallback policy on transfer failure:
    #   "recompute" -> D side re-runs prefill locally; request succeeds at higher latency
    #   "fail"      -> request fails immediately
    timeout_ms: int = 5000


@dataclass(frozen=True)
class MappedPrefixTransferReq:
    op_id: str
    req_id: str
    src_instance: str
    src_instance_epoch: str
    dst_instance: str
    dst_instance_epoch: str
    src_block_ids: tuple[int, ...]
    dst_block_ids: tuple[int, ...]
    namespace: str
    kv_compatibility_id: str
    request_context_digest: str

    def __post_init__(self):
        assert self.op_id, "mapped transfer requires op_id"
        assert len(self.src_block_ids) == len(self.dst_block_ids), (
            f"mapped block count mismatch: {self.src_block_ids=} {self.dst_block_ids=}"
        )


class MappedTransferStatus(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FENCED = "FENCED"
    UNKNOWN = "UNKNOWN"


@dataclass
class MappedTransferOperation:
    request: MappedPrefixTransferReq
    status: MappedTransferStatus = MappedTransferStatus.PREPARED
    source_fenced: bool = False
    target_fenced: bool = False


class MappedTransferRegistry:
    """Track control-plane state while the backend owns the write fence."""

    def __init__(self):
        self._operations: dict[str, MappedTransferOperation] = {}

    def prepare(self, request: MappedPrefixTransferReq) -> MappedTransferOperation:
        existing = self._operations.get(request.op_id)
        if existing is not None:
            if existing.request != request:
                raise ValueError("operation id reused with different mapped transfer")
            return existing
        operation = MappedTransferOperation(request)
        self._operations[request.op_id] = operation
        return operation

    def mark_running(self, op_id: str) -> None:
        operation = self._operations[op_id]
        if operation.status == MappedTransferStatus.PREPARED:
            operation.status = MappedTransferStatus.RUNNING

    def mark_completed(self, op_id: str) -> None:
        operation = self._operations[op_id]
        if operation.status not in {MappedTransferStatus.FENCED, MappedTransferStatus.UNKNOWN}:
            operation.status = MappedTransferStatus.COMPLETED

    def abort_result(
        self, op_id: str, *, source_fenced: bool, target_fenced: bool
    ) -> MappedTransferStatus:
        operation = self._operations.get(op_id)
        if operation is None:
            return MappedTransferStatus.UNKNOWN
        if operation.status == MappedTransferStatus.COMPLETED:
            return MappedTransferStatus.COMPLETED
        operation.source_fenced = source_fenced
        operation.target_fenced = target_fenced
        operation.status = (
            MappedTransferStatus.FENCED
            if source_fenced and target_fenced else MappedTransferStatus.UNKNOWN
        )
        return operation.status

    def status(self, op_id: str) -> MappedTransferStatus:
        operation = self._operations.get(op_id)
        return operation.status if operation is not None else MappedTransferStatus.UNKNOWN

    def contains(self, op_id: str) -> bool:
        return op_id in self._operations


class EndpointFenceStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FENCED = "FENCED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EndpointFenceSnapshot:
    endpoint_ref_key: str
    status: EndpointFenceStatus
    work_terminal: bool
    cuda_visibility_terminal: bool
    resources_held: bool
    held_resource_kinds: tuple[str, ...]


@dataclass
class EndpointTransferOperation:
    endpoint_ref_key: str
    held_resource_kinds: tuple[str, ...]
    accepting_new_work: bool = True
    launched: bool = False
    work_handles: tuple[object, ...] = ()
    work_terminal: bool = False
    completion_event: object | None = None
    completion_event_recorded: bool = False
    cuda_visibility_terminal: bool = False
    data_complete: bool = False


class EndpointTransferRegistry:
    """Track transport termination separately from resource ownership.

    Terminal states fence writers; generic finalize releases held resources.
    """

    def __init__(self):
        self._operations: dict[str, EndpointTransferOperation] = {}

    def register(
        self, endpoint_ref_key: str, *, resource_kinds: tuple[str, ...]
    ) -> EndpointTransferOperation:
        kinds = tuple(sorted(set(resource_kinds)))
        existing = self._operations.get(endpoint_ref_key)
        if existing is not None:
            if existing.held_resource_kinds != kinds:
                raise ValueError("endpoint transfer ref reused with different resources")
            return existing
        operation = EndpointTransferOperation(endpoint_ref_key, kinds)
        self._operations[endpoint_ref_key] = operation
        return operation

    def mark_launched(
        self,
        endpoint_ref_key: str,
        work_handles: list[object],
        completion_event: object,
    ) -> None:
        operation = self._operations[endpoint_ref_key]
        if not operation.accepting_new_work:
            raise ValueError("fenced endpoint cannot launch new work")
        operation.launched = True
        operation.work_handles = tuple(work_handles)
        operation.completion_event = completion_event

    def mark_completion_event_recorded(self, endpoint_ref_key: str) -> None:
        operation = self._operations[endpoint_ref_key]
        if operation.completion_event is None:
            raise ValueError("endpoint transfer has no completion event")
        if not operation.work_terminal:
            raise ValueError("endpoint transfer Work handles are not terminal")
        operation.completion_event_recorded = True

    def mark_work_terminal(self, endpoint_ref_key: str) -> None:
        """Latch successful ``Work.wait`` completion as monotonic authority."""

        operation = self._operations[endpoint_ref_key]
        if not operation.launched or not operation.work_handles:
            raise ValueError("endpoint transfer has no launched Work handles")
        operation.work_terminal = True

    def mark_data_complete(self, endpoint_ref_key: str) -> None:
        operation = self._operations[endpoint_ref_key]
        if not operation.completion_event_recorded:
            raise ValueError("endpoint transfer completion event was not recorded")
        operation.data_complete = True

    def abort(self, endpoint_ref_key: str) -> EndpointFenceSnapshot:
        operation = self._operations[endpoint_ref_key]
        operation.accepting_new_work = False
        return self.status(endpoint_ref_key)

    def status(self, endpoint_ref_key: str) -> EndpointFenceSnapshot:
        operation = self._operations[endpoint_ref_key]
        if not operation.launched and not operation.accepting_new_work:
            status = EndpointFenceStatus.FENCED
            work_terminal = True
            cuda_terminal = True
        else:
            work_terminal = operation.work_terminal
            if operation.completion_event_recorded \
                    and not operation.cuda_visibility_terminal \
                    and operation.completion_event is not None \
                    and operation.completion_event.query():
                operation.cuda_visibility_terminal = True
            cuda_terminal = operation.cuda_visibility_terminal
            if work_terminal and cuda_terminal:
                status = (
                    EndpointFenceStatus.COMPLETED
                    if operation.data_complete else EndpointFenceStatus.FENCED
                )
            else:
                status = EndpointFenceStatus.UNKNOWN
        return EndpointFenceSnapshot(
            endpoint_ref_key=endpoint_ref_key,
            status=status,
            work_terminal=work_terminal,
            cuda_visibility_terminal=cuda_terminal,
            resources_held=bool(operation.held_resource_kinds),
            held_resource_kinds=operation.held_resource_kinds,
        )


@dataclass
class ChunkedBlock:
    """Unit of transfer: one or more numerically adjacent block_ids merged into one chunk.

    Merging reduces RDMA startup count (~10us RTT per call). See _coalesce.
    """
    block_ids:  list[int]  # [first..last], numerically contiguous
    gpu_ptr:    int        # kv_cache[:, :, block_ids[0]].data_ptr()
    size_bytes: int        # = len(block_ids) * block_bytes
    seq_id:     str
    op_id:      str


class KVTransportBackend(Protocol):
    """Unified transport interface. KVBlockPusher depends only on this protocol.

    ack() does not own source block lifetime or send an RPC to serve.
    serve detects KV readiness by polling per-request state (KV_TRANSFERRING -> RUNNING).
    """

    def send_async(
        self,
        dst: str,
        chunk: ChunkedBlock,
        on_complete: Callable[[], None],
    ) -> None: ...

    def send_batch_async(
        self,
        dst: str,
        chunks: list[ChunkedBlock],
        on_complete: Callable[[], None],
    ) -> None:
        # Default: fall back to sequential send_async.
        for i, chunk in enumerate(chunks):
            cb = on_complete if i == len(chunks) - 1 else (lambda: None)
            self.send_async(dst, chunk, on_complete=cb)

    def ack(self, op_id: str, dst: str, bytes_sent: int) -> None: ...
    def has_pending(self) -> bool: ...


def _calc_block_bytes(kv_cache: torch.Tensor, block_size: int) -> int:
    """Bytes per logical block across all layers and K/V.

    kv_cache shape: [2, num_layers, num_blocks, block_size, kv_heads, head_dim]
    """
    return (2 * kv_cache.shape[1] * block_size *
            kv_cache.shape[4] * kv_cache.shape[5] * kv_cache.element_size())


class KVBlockPusher:
    """Batch KV blocks and push to a decode instance with per-dst flow control.

    Borrows three mechanisms from Spark ShuffleBlockPusher:
      1. Coalescing: merge adjacent blocks into one chunk to reduce RDMA ops.
      2. Flow control: dual cap (bytes_inflight + blocks_per_peer) to avoid overloading dst.
      3. Deferred queue: FIFO per dst; blocks must arrive in order to avoid D-side
         reads of uninitialised GPU memory.

    Deadlock guard (_coalesce): a single chunk is capped at max_bytes_inflight * 0.9
    so it can always pass the flow-control check and never get permanently stuck.
    """

    def __init__(
        self,
        transport: KVTransportBackend,
        kv_cache: torch.Tensor,
        block_size: int,
        max_bytes_inflight: int = 256 * 1024 * 1024,
        max_blocks_per_peer: int = 64,
    ):
        self.transport   = transport
        self.kv_cache    = kv_cache
        self.block_bytes = _calc_block_bytes(kv_cache, block_size)
        assert self.block_bytes <= max_bytes_inflight, (
            f"single block {self.block_bytes/1e6:.0f}MB > max_bytes_inflight "
            f"{max_bytes_inflight/1e6:.0f}MB; increase max_bytes_inflight"
        )
        self.max_bytes_inflight  = max_bytes_inflight
        self.max_blocks_per_peer = max_blocks_per_peer
        self.bytes_inflight: int = 0
        self.blocks_inflight: dict[str, int] = defaultdict(int)
        self.deferred: dict[str, deque] = defaultdict(deque)
        # op_id -> [total_chunks, done_chunks, dst, bytes_done]
        self._op_tracker: dict[str, list] = {}
        self._op_callbacks: dict[str, Callable[[], None]] = {}

    def required_blocks(self, req: TransferReq) -> list[int]:
        hint = set(req.block_hint)
        return [block_id for block_id in req.block_table if block_id not in hint]

    def transfer(
        self,
        req: TransferReq,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Initiate KV transfer for one sequence (called after prefill completes)."""
        dst = req.dst_instance
        delta = self.required_blocks(req)
        if not delta:
            self.transport.ack(req.op_id, dst, bytes_sent=0)
            if on_complete is not None:
                on_complete()
            return
        chunks = self._coalesce(delta, req.seq_id, req.op_id)
        assert req.op_id not in self._op_tracker, (
            f"duplicate local transfer operation: {req.op_id!r}"
        )
        self._op_tracker[req.op_id] = [len(chunks), 0, dst, 0]
        if on_complete is not None:
            self._op_callbacks[req.op_id] = on_complete
        try:
            self._flush_or_defer(dst, chunks)
        except Exception:
            self._cancel_local_op(req.op_id)
            raise

    def poll(self) -> None:
        poll_completions = getattr(self.transport, "poll_completions", None)
        if poll_completions is not None:
            poll_completions()

    def has_pending(self) -> bool:
        transport_pending = getattr(self.transport, "has_pending", None)
        return (
            bool(self._op_tracker)
            or any(self.deferred.values())
            or self.bytes_inflight != 0
            or any(self.blocks_inflight.values())
            or (transport_pending is not None and transport_pending())
        )

    def _cancel_local_op(self, op_id: str) -> None:
        for dst, queue in list(self.deferred.items()):
            self.deferred[dst] = deque(
                item for item in queue if item[1].op_id != op_id
            )
        self._op_tracker.pop(op_id, None)
        self._op_callbacks.pop(op_id, None)

    def _fail_local_op(self, op_id: str) -> None:
        callback = self._op_callbacks.get(op_id)
        self._cancel_local_op(op_id)
        if callback is not None:
            callback()

    def _flush_or_defer(self, dst: str, new_chunks: list[ChunkedBlock]) -> None:
        for c in new_chunks:
            self.deferred[dst].append((dst, c))
        self._flush_deferred(dst)

    def _flush_deferred(self, dst: str) -> None:
        """Greedily collect pushable chunks into one grouped API submission."""
        q = self.deferred[dst]
        batch: list[ChunkedBlock] = []
        tb  = self.bytes_inflight
        tbl = self.blocks_inflight[dst]
        while q:
            _, chunk = q[0]
            if (tb  + chunk.size_bytes     <= self.max_bytes_inflight and
                tbl + len(chunk.block_ids) <= self.max_blocks_per_peer):
                q.popleft()
                batch.append(chunk)
                tb  += chunk.size_bytes
                tbl += len(chunk.block_ids)
            else:
                break
        if not batch:
            return
        tot_b  = sum(c.size_bytes for c in batch)
        tot_bl = sum(len(c.block_ids) for c in batch)
        self.bytes_inflight       += tot_b
        self.blocks_inflight[dst] += tot_bl
        try:
            self.transport.send_batch_async(
                dst, batch,
                on_complete=lambda: self._on_batch_complete(
                    dst, tot_b, tot_bl, batch
                ),
            )
        except Exception:
            self.bytes_inflight -= tot_b
            self.blocks_inflight[dst] -= tot_bl
            for op_id in {chunk.op_id for chunk in batch}:
                self._fail_local_op(op_id)
            raise

    def _on_batch_complete(
        self, dst: str, bytes_sent: int, blocks_sent: int,
        chunks: list[ChunkedBlock],
    ) -> None:
        self.bytes_inflight       -= bytes_sent
        self.blocks_inflight[dst] -= blocks_sent
        for chunk in chunks:
            op = chunk.op_id
            if op in self._op_tracker:
                t = self._op_tracker[op]
                t[1] += 1
                t[3] += chunk.size_bytes
                if t[1] == t[0]:
                    del self._op_tracker[op]
                    self.transport.ack(op, t[2], t[3])
                    callback = self._op_callbacks.pop(op, None)
                    if callback is not None:
                        callback()
        self._flush_deferred(dst)

    def _coalesce(self, block_ids: list[int], seq_id: str, op_id: str) -> list[ChunkedBlock]:
        """Merge numerically adjacent block_ids into chunks.

        Split on two conditions:
          1. block_ids are not contiguous (gap > 1)
          2. merging would exceed max_bytes_inflight * 0.9 (deadlock guard)
        """
        if not block_ids:
            return []
        max_bpc = max(1, int(self.max_bytes_inflight * 0.9) // self.block_bytes)
        chunks, start = [], 0
        for i in range(1, len(block_ids)):
            if block_ids[i] != block_ids[i - 1] + 1 or (i - start) >= max_bpc:
                chunks.append(self._make_chunk(block_ids[start:i], seq_id, op_id))
                start = i
        chunks.append(self._make_chunk(block_ids[start:], seq_id, op_id))
        return chunks

    def _make_chunk(self, block_ids: list[int], seq_id: str, op_id: str) -> ChunkedBlock:
        size = len(block_ids) * self.block_bytes
        assert size <= self.max_bytes_inflight, (
            f"chunk {block_ids}: {size/1e6:.0f}MB > max_bytes_inflight "
            f"{self.max_bytes_inflight/1e6:.0f}MB; increase max_bytes_inflight"
        )
        return ChunkedBlock(
            block_ids=block_ids,
            gpu_ptr=self.kv_cache[:, :, block_ids[0]].data_ptr(),
            size_bytes=size, seq_id=seq_id, op_id=op_id,
        )


class KVReceiver:
    """Track KV block readiness on the D side for the scheduler to poll.

    The actual data movement is handled by the transport backend.
    """

    def __init__(self) -> None:
        self._pending: dict[str, list[int]] = {}
        self._ready: set[str] = set()

    def expect(self, seq_id: str, block_ids: list[int]) -> None:
        self._pending[seq_id] = block_ids

    def mark_received(self, seq_id: str) -> None:
        self._pending.pop(seq_id, None)
        self._ready.add(seq_id)

    def is_ready(self, seq_id: str) -> bool:
        return seq_id in self._ready

    def consume_ready(self, seq_id: str) -> None:
        self._ready.discard(seq_id)


class NCCLTransport:
    """NCCL P2P transport backend.

    send_batch_async uses dist.batch_isend_irecv: N chunks in one kernel launch
    instead of N separate isend calls. Each isend call carries ~5-50us Python->NCCL->CUDA
    overhead; batching amortises that to a single fixed cost regardless of N.

    The send side issues one P2POp per block (via _block_slices) to match the
    per-block irecv ops issued by recv_kv on the destination.

    recv_kv receives KV from the P side into pre-allocated local blocks (handshake Step 3).
    Receives into a temporary contiguous buffer then copy_ into kv_cache, because
    kv_cache[:, :, block_id] is non-contiguous and NCCL requires contiguous tensors.
    """

    def __init__(self, pd_group, decode_rank: int, kv_cache: torch.Tensor):
        self.pd_group    = pd_group
        self.decode_rank = decode_rank  # group-local rank of the decode side
        self.kv_cache    = kv_cache
        try:
            self.stream = torch.cuda.Stream()
        except Exception:
            self.stream = None  # CPU test environment
        # Each entry: (reqs, chunks, slices, on_complete)
        # slices holds tensor references to prevent GC during in-flight transfer.
        self._pending: list = []

    def send_async(
        self,
        dst: str,
        chunk: ChunkedBlock,
        on_complete: Callable[[], None],
    ) -> None:
        self.send_batch_async(dst, [chunk], on_complete)

    def send_batch_async(
        self,
        dst: str,
        chunks: list[ChunkedBlock],
        on_complete: Callable[[], None],
    ) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            on_complete()
            return

        ctx = torch.cuda.stream(self.stream) if self.stream else _null_ctx()
        with ctx:
            ops    = []
            slices = []
            for chunk in chunks:
                # Send one message per block so the count and shape match the
                # per-block irecv ops issued by recv_kv on the destination.
                # All ops are submitted in a single batch_isend_irecv call.
                for kv_slice in self._block_slices(chunk.block_ids):
                    slices.append(kv_slice)
                    ops.append(dist.P2POp(
                        dist.isend, kv_slice,
                        peer=self.decode_rank,
                        group=self.pd_group,
                    ))
            reqs = dist.batch_isend_irecv(ops)
            self._pending.append((reqs, chunks, slices, on_complete))

    def _block_slices(self, block_ids: list[int]) -> list[torch.Tensor]:
        """One contiguous tensor per block, matching the recv_kv irecv shape."""
        return [
            self.kv_cache[:, :, block_id, :, :, :].contiguous()
            for block_id in block_ids
        ]

    def recv_kv(self, src_rank: int, block_ids: list[int]) -> None:
        """Receive KV from src_rank into kv_cache block slots (handshake Step 3).

        Args:
            src_rank:  group-local rank of the P side (typically 0)
            block_ids: pre-allocated local block ids from pre_alloc_blocks
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return

        ctx = torch.cuda.stream(self.stream) if self.stream else _null_ctx()
        with ctx:
            ops  = []
            bufs = []
            for block_id in block_ids:
                # kv_cache[:, :, block_id] is non-contiguous after dim-2 indexing;
                # recv into a contiguous tmp buffer then copy_ back.
                buf = torch.empty_like(self.kv_cache[:, :, block_id, :, :, :])
                bufs.append((block_id, buf))
                ops.append(dist.P2POp(
                    dist.irecv, buf,
                    peer=src_rank,
                    group=self.pd_group,
                ))
            if ops:
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()
                for block_id, buf in bufs:
                    self.kv_cache[:, :, block_id, :, :, :].copy_(buf)

    def poll_completions(self) -> None:
        """Check pending sends; fire callbacks for completed batches."""
        still = []
        for reqs, chunks, slices, cb in self._pending:
            if all(r.is_completed() for r in reqs):
                cb()
            else:
                still.append((reqs, chunks, slices, cb))
        self._pending = still

    def has_pending(self) -> bool:
        return bool(self._pending)

    def ack(self, op_id: str, dst: str, bytes_sent: int) -> None:
        pass


class CUDAIPCTransport:
    """CUDA IPC transport: share GPU memory handles across processes on the same host."""

    def __init__(self, shm_channel, kv_cache: torch.Tensor):
        self.shm      = shm_channel
        self.kv_cache = kv_cache

    def send_async(
        self,
        dst: str,
        chunk: ChunkedBlock,
        on_complete: Callable[[], None],
    ) -> None:
        # IPC handle transfer is near-zero cost; fire callback immediately.
        on_complete()

    def send_batch_async(
        self,
        dst: str,
        chunks: list[ChunkedBlock],
        on_complete: Callable[[], None],
    ) -> None:
        for i, chunk in enumerate(chunks):
            cb = on_complete if i == len(chunks) - 1 else (lambda: None)
            self.send_async(dst, chunk, on_complete=cb)

    def ack(self, op_id: str, dst: str, bytes_sent: int) -> None:
        pass

    def has_pending(self) -> bool:
        return False


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def build_transport(
    config,
    pd_group,
    kv_cache: torch.Tensor,
) -> KVTransportBackend:
    """Select transport backend from config.kv_transfer_backend."""
    backend = config.kv_transfer_backend
    if backend == "auto":
        same_host = (not config.pd_decode_addr or
                     config.pd_decode_addr.startswith(("localhost", "127.")))
        backend = "ipc" if same_host else "nccl"
    if backend == "nccl":
        return NCCLTransport(pd_group, decode_rank=1, kv_cache=kv_cache)
    if backend == "ipc":
        return CUDAIPCTransport(shm_channel=None, kv_cache=kv_cache)
    raise ValueError(f"Unknown kv_transfer_backend: {backend!r}, "
                     f"expected 'nccl' | 'ipc' | 'auto'")
