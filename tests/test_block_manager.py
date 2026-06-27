import pytest

from prism_infer.engine.block_manager import BlockManager
from prism_infer.engine.sequence import Sequence


@pytest.fixture
def small_block():
    # Sequence.block(i)/num_blocks use the class attribute; shrink it so short token
    # lists span several blocks (BlockManager takes its own block_size separately).
    prev = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = prev

def test_compute_hash_is_deterministic():
    assert BlockManager.compute_hash([1, 2, 3, 4]) == BlockManager.compute_hash([1, 2, 3, 4])


def test_compute_hash_depends_on_prefix():
    h0 = BlockManager.compute_hash([1, 2, 3, 4])              # first block, no prefix
    chained = BlockManager.compute_hash([5, 6, 7, 8], h0)     # chained on h0
    standalone = BlockManager.compute_hash([5, 6, 7, 8])      # same tokens, no prefix
    assert chained != standalone                              # prefix changes the hash

def test_can_allocate_no_cache_returns_zero(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8])     # 2 full blocks, nothing cached yet
    assert bm.can_allocate(seq) == 0


def test_can_allocate_insufficient_returns_minus_one(small_block):
    bm = BlockManager(num_blocks=1, block_size=4)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])   # needs 3 blocks, only 1
    assert bm.can_allocate(seq) == -1


def test_allocate_sets_block_table_and_cached_tokens(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # 3 blocks
    n = bm.can_allocate(seq)
    bm.allocate(seq, n)
    assert len(seq.block_table) == seq.num_blocks == 3
    assert seq.num_cached_tokens == 0
    assert all(bm.blocks[bid].ref_count == 1 for bid in seq.block_table)

def test_hash_blocks_registers_full_blocks_only(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # 3 blocks; block2 partial (2 tok)
    bm.allocate(seq, 0)
    seq.num_scheduled_tokens = seq.num_tokens
    bm.hash_blocks(seq)
    # block0 + block1 are full -> registered; block2 is partial -> not registered
    assert len(bm.hash_to_block_id) == 2
    assert bm.blocks[seq.block_table[0]].hash != -1
    assert bm.blocks[seq.block_table[1]].hash != -1
    assert bm.blocks[seq.block_table[2]].hash == -1


def test_prefix_hit_shares_block_refcount(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])     # 3 blocks; first 2 full
    bm.allocate(a, 0)
    a.num_scheduled_tokens = a.num_tokens
    bm.hash_blocks(a)                                  # register block0/block1
    first_two = a.block_table[:2]

    b = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 99, 100])    # shares the first 8 tokens
    n = bm.can_allocate(b)
    assert n == 2                                       # 2 prefix blocks hit
    bm.allocate(b, n)
    assert b.block_table[:2] == first_two              # reused the same physical blocks
    assert b.num_cached_tokens == 8
    for bid in first_two:
        assert bm.blocks[bid].ref_count == 2           # shared by A and B


def test_prefix_miss_on_content_mismatch(small_block):
    # Same block count but different first-block content -> no prefix hit.
    bm = BlockManager(num_blocks=8, block_size=4)
    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
    bm.allocate(a, 0)
    a.num_scheduled_tokens = a.num_tokens
    bm.hash_blocks(a)
    b = Sequence([9, 9, 9, 9, 5, 6, 7, 8])             # different block0 content
    assert bm.can_allocate(b) == 0                     # no cached blocks reused

def test_may_append_allocates_on_block_boundary(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    seq = Sequence([1, 2, 3, 4])          # exactly 1 full block
    bm.allocate(seq, 0)
    assert len(seq.block_table) == 1
    seq.append_token(5)                   # now 5 tokens -> 5 % 4 == 1 -> needs new block
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    seq.append_token(6)                   # 6 % 4 == 2 -> no new block
    bm.may_append(seq)
    assert len(seq.block_table) == 2


def test_can_append_false_when_full(small_block):
    bm = BlockManager(num_blocks=1, block_size=4)
    seq = Sequence([1, 2, 3, 4])          # uses the only block
    bm.allocate(seq, 0)
    seq.append_token(5)                   # 5 % 4 == 1 -> would need a new block
    assert bm.can_append(seq) is False    # but no block available

def test_deallocate_decrements_refcount_and_frees(small_block):
    bm = BlockManager(num_blocks=8, block_size=4)
    a = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bm.allocate(a, 0)
    a.num_scheduled_tokens = a.num_tokens
    bm.hash_blocks(a)
    b = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 99, 100])    # shares first 2 blocks
    bm.allocate(b, bm.can_allocate(b))
    shared = b.block_table[:2]

    bm.deallocate(a)                                   # A releases; shared blocks ref 2->1
    for bid in shared:
        assert bm.blocks[bid].ref_count == 1           # still held by B
    bm.deallocate(b)                                   # B releases; ref 1->0
    for bid in shared:
        assert bm.blocks[bid].ref_count == 0
    assert b.block_table == []
    assert b.num_cached_tokens == 0