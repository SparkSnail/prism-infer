from collections import deque, OrderedDict
import xxhash
import numpy as np

from dataclasses import dataclass

from prism_infer.engine.sequence import Sequence


@dataclass(slots=True)
class _CPUEntry:
    slot:      int        # index into KVOffloader.cpu_pool
    token_ids: list[int]  # used to verify on recall



class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
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

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _num_available(self) -> int:
        return len(self.free_block_ids) + len(self.evictable)
    
    def _evict_one(self) -> int:
        block_id, _ = self.evictable.popitem(last=False)
        block = self.blocks[block_id]
        if (self.offloader is not None and block.hash != -1 and block.hash not in self.gpu_to_cpu and self.offloader.has_room()):
            slot = self.offloader.copy_gpu_to_cpu(block_id)
            self.gpu_to_cpu[block.hash] = _CPUEntry(slot, block.token_ids)

        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
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

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
