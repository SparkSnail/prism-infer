from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from prism_infer.engine.sequence import Sequence
    from prism_infer.engine.kv_transfer import KVBlockPusher, KVReceiver
    from prism_infer.config import Config


class KVConnector(Protocol):
    """Single choke-point for PD-role behaviour. LLMEngine calls only these two hooks."""

    def on_prefill_done(self, seq: "Sequence") -> None: ...
    def on_before_decode(self, seq: "Sequence") -> bool: ...


class UnifiedConnector:
    """unified mode: both hooks are no-ops; behaviour identical to pre-I4."""
    def on_prefill_done(self, seq: "Sequence") -> None: pass
    def on_before_decode(self, seq: "Sequence") -> bool: return True


class PrefillConnector:
    """prefill-only mode: push KV to decode instance after prefill completes."""

    def __init__(self, pusher: "KVBlockPusher", config: "Config"):
        self.pusher = pusher
        self.config = config

    def on_prefill_done(self, seq: "Sequence") -> None:
        # After pushing KV, mark FINISHED so the scheduler cleans up this seq.
        # The decode side owns the sequence from this point; prefill-only engine
        # does not need to track it further.
        import uuid as _uuid
        from prism_infer.engine.kv_transfer import TransferReq
        from prism_infer.engine.sequence import SequenceStatus
        req = TransferReq(
            op_id=f"{seq.seq_id}-{_uuid.uuid4().hex[:8]}",
            seq_id=str(seq.seq_id),
            src_instance=self.config.instance_id or "prefill-0",
            dst_instance=self.config.pd_decode_addr,
            block_table=seq.block_table[:],
            block_hint=[],  # no hint yet: send full KV (serve will populate on S1/KV-affinity route)
            on_fail=getattr(self.config, "kv_transfer_on_fail", "recompute"),
        )
        self.pusher.transfer(req)
        seq.status = SequenceStatus.FINISHED

    def on_before_decode(self, seq: "Sequence") -> bool:
        raise RuntimeError(
            f"PrefillConnector.on_before_decode called for seq {seq.seq_id}; "
            "prefill-only engine must not run decode steps"
        )


class DecodeConnector:
    """decode-only mode: wait for remote KV before scheduling decode."""

    def __init__(self, receiver: "KVReceiver", transport=None):
        self.receiver = receiver
        self.transport = transport
        self._pending_recv: dict[str, list[int]] = {}

    def on_prefill_done(self, seq: "Sequence") -> None:
        pass  # D-side does not run prefill

    def on_before_decode(self, seq: "Sequence") -> bool:
        """Non-blocking check: return True if KV has arrived for this seq.

        Also registers KV_TRANSFERRING seqs into _pending_recv on first sight,
        so poll_recv() knows which block_ids to receive.
        """
        from prism_infer.engine.sequence import SequenceStatus
        seq_id = str(seq.seq_id)

        if self.receiver.is_ready(seq_id):
            self.receiver.consume_ready(seq_id)
            return True

        if (seq.status == SequenceStatus.KV_TRANSFERRING
                and seq_id not in self._pending_recv):
            self._pending_recv[seq_id] = seq.block_table[:]

        return False  # KV not yet arrived; skip this seq for now

    def poll_recv(self) -> None:
        """Drain pending KV receives at the end of each step.

        recv_kv is a synchronous blocking call; calling it here avoids
        blocking the GPU compute path. Without a transport (CPU tests / IPC
        stub) we go straight to mark_received so tests can run on CPU.
        """
        if not self._pending_recv:
            return
        done = []
        for seq_id, block_ids in self._pending_recv.items():
            if self.transport is not None and hasattr(self.transport, "recv_kv"):
                # src_rank=0: prefill process is rank 0 in the pd group
                self.transport.recv_kv(src_rank=0, block_ids=block_ids)
            self.receiver.mark_received(seq_id)
            done.append(seq_id)
        for sid in done:
            del self._pending_recv[sid]


def _build_connector(config: "Config", kv_cache=None) -> KVConnector:
    """Return the appropriate connector for the configured engine_mode."""
    mode = getattr(config, "engine_mode", "unified")

    if mode == "unified":
        return UnifiedConnector()

    if mode == "prefill-only":
        from prism_infer.engine.kv_transfer import KVBlockPusher, build_transport
        pd_group = getattr(config, "_pd_group", None)
        decode_rank = getattr(config, "_pd_rank", 1)
        transport = build_transport(config, pd_group=pd_group, kv_cache=kv_cache)
        # Inject decode_rank so NCCLTransport knows where to send
        if pd_group is not None and hasattr(transport, "decode_rank"):
            transport.decode_rank = decode_rank
        pusher = KVBlockPusher(
            transport=transport,
            kv_cache=kv_cache,
            block_size=config.kvcache_block_size,
            max_bytes_inflight=config.max_bytes_inflight,
            max_blocks_per_peer=config.max_blocks_per_peer,
        )
        return PrefillConnector(pusher=pusher, config=config)

    if mode == "decode-only":
        from prism_infer.engine.kv_transfer import KVReceiver, build_transport
        receiver = KVReceiver()
        pd_group = getattr(config, "_pd_group", None)
        transport = build_transport(config, pd_group=pd_group, kv_cache=kv_cache) \
            if pd_group is not None else None
        return DecodeConnector(receiver=receiver, transport=transport)

    raise ValueError(
        f"Unknown engine_mode: {mode!r}, expected 'unified' | 'prefill-only' | 'decode-only'"
    )
