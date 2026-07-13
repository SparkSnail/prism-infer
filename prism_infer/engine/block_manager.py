from collections import deque, OrderedDict
from enum import Enum, auto
import threading
import time
import uuid
import xxhash
import numpy as np

from dataclasses import dataclass

from prism_infer.engine.sequence import Sequence


@dataclass(slots=True)
class _CPUEntry:
    slot:      int        # index into KVOffloader.cpu_pool
    token_ids: list[int]  # used to verify on recall


class FullReportRequired(RuntimeError):
    pass


class ConsumerStatus(Enum):
    ACTIVE = auto()
    EXPIRED = auto()


@dataclass(slots=True, frozen=True)
class PrefixEvent:
    kind: str
    namespace: str
    kv_compatibility_id: str
    request_context_digest: str
    chain_hash: int
    block_index: int
    block_id: int
    prefix_tokens: int
    instance_id: str
    instance_epoch: str
    seq_no: int


@dataclass(slots=True)
class ConsumerState:
    generation: str
    status: ConsumerStatus
    acked_seq: int
    last_delivered_seq: int
    lease_deadline: float


@dataclass(slots=True, frozen=True)
class PrefixFullReport:
    instance_id: str
    instance_epoch: str
    snapshot_seq_no: int
    locations: tuple[PrefixEvent, ...]



class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.namespace = "legacy"
        self.kv_compatibility_id = "legacy"
        self.request_context_digest = "text-only"
        self.block_index = -1

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.namespace = "legacy"
        self.kv_compatibility_id = "legacy"
        self.request_context_digest = "text-only"
        self.block_index = -1


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        instance_id: str = "local",
        instance_epoch: str | None = None,
        prefix_event_log_capacity: int = 65536,
        prefix_consumer_lease_s: float = 30.0,
    ):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.evictable: OrderedDict[int, None] = OrderedDict()
        self.evict_count = 0
        self.offloader = None
        self.gpu_to_cpu: dict[int, _CPUEntry] = {}
        self.recall_hit = 0
        self.recall_miss = 0
        self.instance_id = instance_id
        self.instance_epoch = instance_epoch or uuid.uuid4().hex
        self._prefix_state_lock = threading.RLock()
        self._event_log: deque[PrefixEvent] = deque(maxlen=prefix_event_log_capacity)
        self._event_overflow = False
        self._next_seq_no = 1
        self._consumers: dict[str, ConsumerState] = {}
        self._consumer_lease_s = prefix_consumer_lease_s
        self._transfer_pins: dict[str, tuple[int, ...]] = {}
        self._transfer_pin_counts: dict[int, int] = {}

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _num_available(self) -> int:
        return len(self.free_block_ids) + sum(
            block_id not in self._transfer_pin_counts for block_id in self.evictable
        )

    def _evict_one(self) -> int:
        block_id = next(
            (item for item in self.evictable if item not in self._transfer_pin_counts),
            None,
        )
        if block_id is None:
            raise KeyError("no evictable block: cache is empty or transfer-pinned")
        self.evictable.pop(block_id)
        block = self.blocks[block_id]
        if (self.offloader is not None and block.hash != -1
                and block.hash not in self.gpu_to_cpu and self.offloader.has_room()):
            slot = self.offloader.copy_gpu_to_cpu(block_id)
            self.gpu_to_cpu[block.hash] = _CPUEntry(slot, block.token_ids)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        if block.hash != -1:
            self._record_block_event("evicted", block)
        self.evict_count += 1
        return block_id

    def _cpu_has(self, chain_hash: int, token_ids: list[int]) -> bool:
        if self.offloader is None:
            return False
        entry = self.gpu_to_cpu.get(chain_hash)
        return entry is not None and entry.token_ids == token_ids

    def _recall_from_cpu(self, chain_hash: int, token_ids: list[int]) -> int:
        assert self.offloader is not None
        slot = self.gpu_to_cpu.pop(chain_hash).slot
        block_id = self._allocate_block()
        self.offloader.copy_cpu_to_gpu(slot, block_id)
        block = self.blocks[block_id]
        block.update(chain_hash, token_ids)
        self.hash_to_block_id[chain_hash] = block_id
        self.recall_hit += 1
        return block_id

    def _allocate_block(self) -> int:
        with self._prefix_state_lock:
            if self.free_block_ids:
                block_id = self.free_block_ids.popleft()
            else:
                block_id = self._evict_one()
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
            self.used_block_ids.add(block_id)
            return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        if self.blocks[block_id].hash != -1:
            self.evictable[block_id] = None
        else:
            self.free_block_ids.append(block_id)

    def release_block(self, block_id: int) -> None:
        """Decrement ref_count for a single block; return it to the pool when it reaches zero."""
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"block {block_id} ref_count is already 0"
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                num_cached_blocks += 1
                if block_id in self.used_block_ids:
                    num_new_blocks -= 1
            elif self._cpu_has(h, token_ids):
                num_cached_blocks += 1
            else:
                break
        if self._num_available() < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        hashes = []
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            hashes.append((h, token_ids))
        recall_idx = []
        for i, (h, token_ids) in enumerate(hashes):
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                block = self.blocks[block_id]
                if block_id in self.used_block_ids:
                    block.ref_count += 1
                else:
                    block.ref_count = 1
                    del self.evictable[block_id]
                    self.used_block_ids.add(block_id)
                seq.block_table.append(block_id)
            else:
                seq.block_table.append(-1)
                recall_idx.append(i)
        for i in recall_idx:
            h, token_ids = hashes[i]
            block_id = self._recall_from_cpu(h, token_ids)
            seq.block_table[i] = block_id
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return self._num_available() >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(
        self,
        seq: Sequence,
        *,
        namespace: str = "legacy",
        kv_compatibility_id: str = "legacy",
        request_context_digest: str = "text-only",
    ):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        with self._prefix_state_lock:
            h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
            for i in range(start, end):
                block = self.blocks[seq.block_table[i]]
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block.update(h, token_ids)
                block.namespace = namespace
                block.kv_compatibility_id = kv_compatibility_id
                block.request_context_digest = request_context_digest
                block.block_index = i
                self.hash_to_block_id[h] = block.block_id
                self._record_block_event("hash_added", block)

    def _record_block_event(self, kind: str, block: Block) -> PrefixEvent:
        with self._prefix_state_lock:
            if len(self._event_log) == self._event_log.maxlen:
                self._event_overflow = True
            event = PrefixEvent(
                kind, block.namespace, block.kv_compatibility_id,
                block.request_context_digest, block.hash, block.block_index,
                block.block_id, (block.block_index + 1) * self.block_size,
                self.instance_id, self.instance_epoch, self._next_seq_no,
            )
            self._next_seq_no += 1
            self._event_log.append(event)
            return event

    def full_report_and_register(
        self, consumer_id: str, generation: str
    ) -> PrefixFullReport:
        with self._prefix_state_lock:
            previous = self._consumers.get(consumer_id)
            if previous is not None and previous.status == ConsumerStatus.EXPIRED \
                    and previous.generation == generation:
                raise ValueError("expired consumer generation cannot re-register")
            snapshot_seq_no = self._next_seq_no - 1
            locations = tuple(
                PrefixEvent(
                    "hash_added", block.namespace, block.kv_compatibility_id,
                    block.request_context_digest, block.hash, block.block_index,
                    block.block_id, (block.block_index + 1) * self.block_size,
                    self.instance_id, self.instance_epoch, snapshot_seq_no,
                )
                for block in self.blocks if block.hash != -1
            )
            self._consumers[consumer_id] = ConsumerState(
                generation, ConsumerStatus.ACTIVE, snapshot_seq_no,
                snapshot_seq_no, time.monotonic() + self._consumer_lease_s,
            )
            return PrefixFullReport(
                self.instance_id, self.instance_epoch, snapshot_seq_no, locations
            )

    def peek_events(
        self, consumer_id: str, generation: str, after_seq: int, limit: int = 1024
    ) -> list[PrefixEvent]:
        with self._prefix_state_lock:
            state = self._active_consumer(consumer_id, generation)
            if after_seq != state.acked_seq:
                raise ValueError("consumer cursor must equal last ACK")
            if self._event_overflow and self._event_log \
                    and after_seq < self._event_log[0].seq_no - 1:
                raise FullReportRequired("prefix event log overflow")
            events = [event for event in self._event_log if event.seq_no > after_seq][:limit]
            state.last_delivered_seq = events[-1].seq_no if events else state.acked_seq
            state.lease_deadline = time.monotonic() + self._consumer_lease_s
            return events

    def ack_events(self, consumer_id: str, generation: str, up_to_seq: int) -> None:
        with self._prefix_state_lock:
            state = self._active_consumer(consumer_id, generation)
            if not state.acked_seq <= up_to_seq <= state.last_delivered_seq:
                raise ValueError("ACK outside delivered range")
            state.acked_seq = up_to_seq
            state.lease_deadline = time.monotonic() + self._consumer_lease_s
            now = time.monotonic()
            active = [
                item.acked_seq for item in self._consumers.values()
                if item.status == ConsumerStatus.ACTIVE and item.lease_deadline > now
            ]
            trim_through = min(active, default=self._next_seq_no - 1)
            while self._event_log and self._event_log[0].seq_no <= trim_through:
                self._event_log.popleft()

    def _active_consumer(self, consumer_id: str, generation: str) -> ConsumerState:
        state = self._consumers[consumer_id]
        if state.status == ConsumerStatus.EXPIRED \
                or state.lease_deadline <= time.monotonic():
            state.status = ConsumerStatus.EXPIRED
            raise FullReportRequired("prefix consumer lease expired")
        if state.generation != generation:
            raise FullReportRequired("prefix consumer generation changed")
        return state

    def resolve_and_pin_prefix(
        self,
        operation_id: str,
        expected_blocks: list[tuple[int, list[int]]],
        *,
        namespace: str,
        kv_compatibility_id: str,
        request_context_digest: str,
    ) -> tuple[int, ...] | None:
        """Validate a complete chain and pin every source block atomically."""
        with self._prefix_state_lock:
            existing = self._transfer_pins.get(operation_id)
            if existing is not None:
                return existing
            resolved = []
            for chain_hash, token_ids in expected_blocks:
                block_id = self.hash_to_block_id.get(chain_hash)
                if block_id is None:
                    return None
                block = self.blocks[block_id]
                if (
                    block.token_ids != token_ids
                    or block.namespace != namespace
                    or block.kv_compatibility_id != kv_compatibility_id
                    or block.request_context_digest != request_context_digest
                ):
                    return None
                resolved.append(block_id)
            pinned = tuple(resolved)
            self._transfer_pins[operation_id] = pinned
            for block_id in pinned:
                self._transfer_pin_counts[block_id] = (
                    self._transfer_pin_counts.get(block_id, 0) + 1
                )
            return pinned

    def unpin_prefix(self, operation_id: str) -> bool:
        with self._prefix_state_lock:
            block_ids = self._transfer_pins.pop(operation_id, None)
            if block_ids is None:
                return False
            for block_id in block_ids:
                count = self._transfer_pin_counts[block_id] - 1
                if count == 0:
                    self._transfer_pin_counts.pop(block_id)
                else:
                    self._transfer_pin_counts[block_id] = count
            return True

    def commit_pinned_prefix(self, operation_id: str) -> tuple[int, ...]:
        """Convert local-reuse pins into Sequence block references."""
        with self._prefix_state_lock:
            block_ids = self._transfer_pins.get(operation_id)
            if block_ids is None:
                raise KeyError(f"prefix operation is not pinned: {operation_id!r}")
            for block_id in block_ids:
                block = self.blocks[block_id]
                if block.ref_count == 0:
                    self.evictable.pop(block_id, None)
                    self.used_block_ids.add(block_id)
                    block.ref_count = 1
                else:
                    block.ref_count += 1
            self.unpin_prefix(operation_id)
            return block_ids

    def install_prefix_metadata(
        self,
        block_ids: tuple[int, ...],
        token_ids: list[int],
        *,
        namespace: str,
        kv_compatibility_id: str,
        request_context_digest: str,
    ) -> None:
        """Install target metadata after the mapped tensor copy completes."""
        assert len(token_ids) >= len(block_ids) * self.block_size, (
            f"insufficient prefix tokens: {len(token_ids)=} {len(block_ids)=}"
        )
        with self._prefix_state_lock:
            chain_hash = -1
            for index, block_id in enumerate(block_ids):
                block_tokens = token_ids[
                    index * self.block_size:(index + 1) * self.block_size
                ]
                chain_hash = self.compute_hash(block_tokens, chain_hash)
                block = self.blocks[block_id]
                block.update(chain_hash, block_tokens)
                block.namespace = namespace
                block.kv_compatibility_id = kv_compatibility_id
                block.request_context_digest = request_context_digest
                block.block_index = index
                self.hash_to_block_id[chain_hash] = block_id
                self._record_block_event("hash_added", block)
