from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
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

    ack() triggers P-side block release only -- no RPC to serve.
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

    def transfer(self, req: TransferReq) -> None:
        """Initiate KV transfer for one sequence (called after prefill completes)."""
        dst   = req.dst_instance
        hint  = set(req.block_hint)
        delta = [b for b in req.block_table if b not in hint]
        if not delta:
            # Full prefix hit: all blocks already on D side.
            self.transport.ack(req.op_id, dst, bytes_sent=0)
            return
        chunks = self._coalesce(delta, req.seq_id, req.op_id)
        self._op_tracker[req.op_id] = [len(chunks), 0, dst, 0]
        self._flush_or_defer(dst, chunks)

    def _flush_or_defer(self, dst: str, new_chunks: list[ChunkedBlock]) -> None:
        for c in new_chunks:
            self.deferred[dst].append((dst, c))
        self._flush_deferred(dst)

    def _flush_deferred(self, dst: str) -> None:
        """Greedily collect pushable chunks and send as one batch (one kernel launch)."""
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
        self.transport.send_batch_async(
            dst, batch,
            on_complete=lambda: self._on_batch_complete(dst, tot_b, tot_bl, batch),
        )

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
                first, last = chunk.block_ids[0], chunk.block_ids[-1]
                kv_slice = self.kv_cache[:, :, first:last + 1, :, :, :].contiguous()
                slices.append(kv_slice)
                ops.append(dist.P2POp(
                    dist.isend, kv_slice,
                    peer=self.decode_rank,
                    group=self.pd_group,
                ))
            reqs = dist.batch_isend_irecv(ops)
            self._pending.append((reqs, chunks, slices, on_complete))

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

    def ack(self, op_id: str, dst: str, bytes_sent: int) -> None:
        pass  # P-side block release only; no RPC to serve


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
