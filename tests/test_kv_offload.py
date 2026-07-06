from collections import deque

import pytest
import torch

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.kv_offloader import KVOffloader
from prism_infer.engine.sequence import Sequence


class FakeOffloader:
    def __init__(self, capacity):
        self.free = deque(range(capacity))
        self.gpu_to_cpu_calls = 0
        self.cpu_to_gpu_calls = 0

    def has_room(self):
        return bool(self.free)

    def copy_gpu_to_cpu(self, gpu_block_id):
        self.gpu_to_cpu_calls += 1
        return self.free.popleft()

    def copy_cpu_to_gpu(self, slot, gpu_block_id):
        self.cpu_to_gpu_calls += 1
        self.free.append(slot)


def _make_evictable(bm, n):
    ids = [bm._allocate_block() for _ in range(n)]
    for k, bid in enumerate(ids):
        blk = bm.blocks[bid]
        blk.hash = 1000 + k
        blk.token_ids = [k]
        bm.hash_to_block_id[blk.hash] = bid
    for bid in ids:
        bm.blocks[bid].ref_count -= 1
        bm._deallocate_block(bid)
    return ids


def test_evict_offloads_not_drops():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.offloader = FakeOffloader(capacity=4)
    ids = _make_evictable(bm, 3)               # 3 evictable (hash 1000/1001/1002), 5 free
    for _ in range(5):
        bm._allocate_block()                   # drain free
    evicted = bm._allocate_block()             # force eviction of LRU = ids[0]
    assert evicted == ids[0]
    assert bm.offloader.gpu_to_cpu_calls == 1  # offloaded, not dropped
    assert 1000 in bm.gpu_to_cpu               # indexed by its chain hash
    assert bm.gpu_to_cpu[1000].token_ids == [0]  # token_ids stored for validation/restore
    assert 1000 not in bm.hash_to_block_id     # GPU index cleared


def test_evict_drops_when_cpu_full():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.offloader = FakeOffloader(capacity=0)   # no CPU room
    _make_evictable(bm, 3)
    for _ in range(5):
        bm._allocate_block()
    bm._allocate_block()                       # evict, CPU full -> drop
    assert bm.offloader.gpu_to_cpu_calls == 0
    assert bm.gpu_to_cpu == {}


def test_no_offload_when_offloader_none():
    bm = BlockManager(num_blocks=8, block_size=4)
    _make_evictable(bm, 3)
    for _ in range(5):
        bm._allocate_block()
    bm._allocate_block()
    assert bm.gpu_to_cpu == {}
    assert bm.recall_hit == 0


@pytest.fixture
def small_block():
    prev = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = prev


def test_recall_restores_prefix(small_block):
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.offloader = FakeOffloader(capacity=4)

    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
    bm.allocate(a, bm.can_allocate(a))
    a.num_scheduled_tokens = 8
    bm.hash_blocks(a)
    bm.deallocate(a)
    assert len(bm.evictable) == 2

    big = Sequence(list(range(100, 116)))
    assert bm.can_allocate(big) == 0
    bm.allocate(big, 0)
    assert bm.offloader.gpu_to_cpu_calls == 2
    assert len(bm.gpu_to_cpu) == 2             # A's 2 prefix blocks now live on CPU
    bm.deallocate(big)

    b = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])      # 10 tokens -> 3 blocks
    n = bm.can_allocate(b)
    assert n == 2                              # both prefix blocks found in CPU
    bm.allocate(b, n)
    assert bm.recall_hit == 2
    assert b.num_cached_tokens == 8
    assert len(bm.gpu_to_cpu) == 0             # both recalled, CPU slots freed
    # recalled blocks are valid prefix-cache entries again
    for bid in b.block_table[:2]:
        assert bm.blocks[bid].hash != -1
        assert bm.blocks[bid].ref_count >= 1


def test_kv_offloader_roundtrip():
    kv = torch.randn(2, 2, 4, 2, 2, 3)
    off = KVOffloader(kv, num_cpu_blocks=2)
    orig = kv[:, :, 1].clone()
    slot = off.copy_gpu_to_cpu(1)
    kv[:, :, 1].zero_()                        # simulate the GPU block being reused/overwritten
    assert not torch.allclose(kv[:, :, 1], orig)
    off.copy_cpu_to_gpu(slot, 1)               # recall
    assert torch.allclose(kv[:, :, 1], orig)   # content restored exactly
    assert off.has_room()                      # slot freed after copy_cpu_to_gpu


def test_kv_offloader_has_room():
    kv = torch.randn(2, 1, 4, 2, 1, 2)
    off = KVOffloader(kv, num_cpu_blocks=1)
    assert off.has_room()
    off.copy_gpu_to_cpu(0)
    assert not off.has_room()                  # only slot taken
    off.copy_cpu_to_gpu(0, 0)
    assert off.has_room()


def test_recall_real_kvoffloader_moves_bytes(small_block):
    kv = torch.zeros(2, 2, 4, 4, 2, 3)
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.offloader = KVOffloader(kv, num_cpu_blocks=4)

    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8])             # 2 blocks
    bm.allocate(a, bm.can_allocate(a))
    markers = {}
    for i, bid in enumerate(a.block_table):            # write a distinct marker per A block
        kv[:, :, bid] = float(i + 1)
        markers[i] = float(i + 1)
    a.num_scheduled_tokens = 8
    bm.hash_blocks(a)
    bm.deallocate(a)

    big = Sequence(list(range(100, 116)))              # 4 fresh blocks -> evicts (offloads) A's 2
    bm.allocate(big, 0)
    assert bm.offloader.free_slots and bm.recall_hit == 0
    for bid in big.block_table:                        # big overwrites the reused GPU slots
        kv[:, :, bid] = -99.0
    bm.deallocate(big)

    b = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])      # shares A's first 8 tokens -> recall
    assert bm.can_allocate(b) == 2
    bm.allocate(b, 2)
    assert bm.recall_hit == 2
    for i in range(2):                                 # recalled GPU blocks hold A's original bytes
        bid = b.block_table[i]
        assert torch.allclose(kv[:, :, bid], torch.full_like(kv[:, :, bid], markers[i]))
