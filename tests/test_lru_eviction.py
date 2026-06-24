import random

import pytest

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.sequence import Sequence


def _make_evictable(bm, n):
    # Allocate n blocks, give them fake prefix hashes (simulate hash_blocks), then release
    # so they land in `evictable` in allocation order (LRU -> MRU = ids[0] -> ids[-1]).
    ids = [bm._allocate_block() for _ in range(n)]
    for k, bid in enumerate(ids):
        blk = bm.blocks[bid]
        blk.hash = 1000 + k
        blk.token_ids = [k]
        bm.hash_to_block_id[blk.hash] = bid
    for bid in ids:
        bm.blocks[bid].ref_count -= 1      # ref 1 -> 0
        bm._deallocate_block(bid)
    return ids


def test_evictable_is_access_order_lru():
    bm = BlockManager(num_blocks=8, block_size=4)
    ids = _make_evictable(bm, 3)               # evictable order = ids[0], ids[1], ids[2]
    assert list(bm.evictable.keys()) == ids
    # free still has the other 5 blocks; _allocate_block prefers free first (no eviction)
    for _ in range(5):
        assert bm._allocate_block() not in ids
    # free now empty -> next allocate evicts the LRU end = ids[0]
    evicted = bm._allocate_block()
    assert evicted == ids[0]
    assert ids[0] not in bm.evictable
    assert 1000 not in bm.hash_to_block_id    # its prefix-cache index entry was cleared
    assert bm.evict_count == 1


def test_pin_active_never_evicted():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = bm._allocate_block()                   # ref_count=1 -> pinned
    bm.blocks[a].hash = 50
    bm.hash_to_block_id[50] = a
    assert a not in bm.evictable               # pinned blocks are never evictable
    [bm._allocate_block() for _ in range(3)]   # exhaust the rest
    assert bm._num_available() == 0            # nothing free, nothing evictable
    assert a in bm.used_block_ids              # active block untouched
    assert bm.hash_to_block_id.get(50) == a


def test_num_available_counts_evictable():
    bm = BlockManager(num_blocks=8, block_size=4)
    assert bm._num_available() == 8
    _make_evictable(bm, 3)                      # 3 -> evictable, 5 -> free
    assert len(bm.evictable) == 3
    assert len(bm.free_block_ids) == 5
    assert bm._num_available() == 8          # evictable still counts as reclaimable


def test_deallocate_routes_by_hash():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = bm._allocate_block()
    bm.blocks[a].hash = 7                       # filled block (has hash)
    b = bm._allocate_block()                    # unfilled block (hash == -1)
    for bid in (a, b):
        bm.blocks[bid].ref_count -= 1
        bm._deallocate_block(bid)
    assert a in bm.evictable                    # hash != -1 -> evictable
    assert b in bm.free_block_ids               # hash == -1 -> free
    assert a not in bm.free_block_ids


@pytest.fixture
def small_block():
    # BlockManager uses its own block_size, but Sequence.block(i)/num_blocks use the
    # class attribute. Shrink it so short token lists span multiple blocks.
    prev = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = prev


def test_prefix_reuse_pulls_from_evictable(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8])              # 8 tokens -> 2 full blocks
    assert bm.can_allocate(a) == 0
    bm.allocate(a, 0)
    a.num_scheduled_tokens = 8
    bm.hash_blocks(a)                                   # register block0/block1 hashes
    bm.deallocate(a)                                    # both blocks -> evictable
    assert len(bm.evictable) == 2

    b = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])       # shares the first 8 tokens
    n = bm.can_allocate(b)
    assert n == 2                                       # 2 prefix blocks hit
    bm.allocate(b, n)
    assert len(bm.evictable) == 0                       # both reused blocks pulled out (O(1))
    assert b.num_cached_tokens == 8
    assert all(bm.blocks[bid].ref_count >= 1 for bid in b.block_table)


def test_invariant_after_random_ops():
    random.seed(0)
    n = 16
    bm = BlockManager(num_blocks=n, block_size=4)
    held = []
    for _ in range(300):
        if held and random.random() < 0.5:
            bid = held.pop(random.randrange(len(held)))
            bm.blocks[bid].ref_count -= 1
            if bm.blocks[bid].ref_count == 0:
                bm._deallocate_block(bid)
        elif bm._num_available() > 0:
            bid = bm._allocate_block()
            if random.random() < 0.5:          # some blocks get a hash -> land in evictable
                bm.blocks[bid].hash = random.randint(1, 10_000)
            held.append(bid)
        free = set(bm.free_block_ids)
        evic = set(bm.evictable.keys())
        used = set(bm.used_block_ids)
        assert free.isdisjoint(evic) and free.isdisjoint(used) and evic.isdisjoint(used)
        assert free | evic | used == set(range(n))
