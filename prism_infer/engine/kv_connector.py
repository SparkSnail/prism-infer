from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from prism_infer.engine.sequence import Sequence
    from prism_infer.engine.kv_transfer import KVBlockPusher, KVReceiver, TransferReq
    from prism_infer.config import Config


class KVConnector(Protocol):
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
        import uuid as _uuid
        from prism_infer.engine.kv_transfer import TransferReq
        req = TransferReq(
            op_id=f"{seq.seq_id}-{_uuid.uuid4().hex[:8]}",
            seq_id=str(seq.seq_id),
            src_instance=self.config.instance_id or "prefill-0",
            dst_instance=self.config.pd_decode_addr,
            block_table=seq.block_table[:],
            block_hint=[],  # populated by serve in Week 10; send full KV for now
            on_fail=getattr(self.config, "kv_transfer_on_fail", "recompute"),
        )
        self.pusher.transfer(req)

    def on_before_decode(self, seq: "Sequence") -> bool:
        raise RuntimeError(
            f"PrefillConnector.on_before_decode called for seq {seq.seq_id}; "
            "prefill-only engine must not run decode steps"
        )


class DecodeConnector:
    """decode-only mode: wait for remote KV before scheduling decode."""

    def __init__(self, receiver: "KVReceiver"):
        self.receiver = receiver

    def on_prefill_done(self, seq: "Sequence") -> None:
        pass  # D-side does not run prefill

    def on_before_decode(self, seq: "Sequence") -> bool:
        seq_id = str(seq.seq_id)
        if self.receiver.is_ready(seq_id):
            self.receiver.consume_ready(seq_id)
            return True
        return False  # KV not yet arrived; skip this seq for now


def _build_connector(config: "Config", kv_cache=None) -> KVConnector:
    """Return the appropriate connector for the configured engine_mode."""
    mode = getattr(config, "engine_mode", "unified")
    if mode == "unified":
        return UnifiedConnector()
    if mode == "prefill-only":
        from prism_infer.engine.kv_transfer import KVBlockPusher, build_transport
        transport = build_transport(config, pd_group=None, kv_cache=kv_cache)
        pusher = KVBlockPusher(
            transport=transport,
            kv_cache=kv_cache,
            block_size=config.kvcache_block_size,
            max_bytes_inflight=config.max_bytes_inflight,
            max_blocks_per_peer=config.max_blocks_per_peer,
        )
        return PrefillConnector(pusher=pusher, config=config)
    if mode == "decode-only":
        from prism_infer.engine.kv_transfer import KVReceiver
        return DecodeConnector(receiver=KVReceiver())
    raise ValueError(f"Unknown engine_mode: {mode!r}")
