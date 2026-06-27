import pytest

from prism_infer.engine.sequence import Sequence, SequenceStatus
from prism_infer.sampling_params import SamplingParams


@pytest.fixture
def small_block():
    prev = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = prev


def test_initial_state():
    seq = Sequence([1, 2, 3, 4, 5])
    assert seq.status == SequenceStatus.WAITING
    assert seq.is_finished is False
    assert seq.num_tokens == 5
    assert seq.num_prompt_tokens == 5
    assert seq.num_completion_tokens == 0
    assert seq.last_token == 5
    assert seq.is_prefill is True
    assert seq.block_table == []


def test_seq_id_is_unique_and_increasing():
    a = Sequence([1])
    b = Sequence([2])
    assert b.seq_id == a.seq_id + 1


def test_num_blocks_and_block_slicing(small_block):
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])    # 10 tokens, block_size 4
    assert seq.num_blocks == 3                          # ceil(10/4)
    assert seq.block(0) == [1, 2, 3, 4]
    assert seq.block(1) == [5, 6, 7, 8]
    assert seq.block(2) == [9, 10]                      # partial last block


def test_last_block_num_tokens(small_block):
    seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])    # 10 tokens -> 3 blocks
    assert seq.last_block_num_tokens == 10 - 2 * 4      # == 2
    full = Sequence([1, 2, 3, 4, 5, 6, 7, 8])           # 8 tokens -> 2 full blocks
    assert full.last_block_num_tokens == 4


def test_append_token_updates_counters():
    seq = Sequence([1, 2, 3])
    seq.append_token(42)
    assert seq.token_ids == [1, 2, 3, 42]
    assert seq.last_token == 42
    assert seq.num_tokens == 4
    assert seq.num_completion_tokens == 1
    assert seq.num_prompt_tokens == 3                   # prompt length unchanged


def test_prompt_and_completion_views():
    seq = Sequence([1, 2, 3])
    seq.append_token(42)
    seq.append_token(99)
    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.completion_token_ids == [42, 99]


def test_len_and_getitem():
    seq = Sequence([1, 2, 3, 4, 5])
    assert len(seq) == 5
    assert seq[1:3] == [2, 3]


def test_constructor_copies_token_ids():
    src = [1, 2, 3]
    seq = Sequence(src)
    src.append(999)                                     # mutate caller's list
    assert seq.token_ids == [1, 2, 3]                   # sequence is unaffected


def test_getstate_prefill_sends_full_tokens():
    seq = Sequence([1, 2, 3, 4, 5])
    seq.is_prefill = True
    state = seq.__getstate__()
    # state = (num_tokens, num_prompt_tokens, num_cached_tokens,
    #          num_scheduled_tokens, block_table, last_state)
    assert state[-1] == [1, 2, 3, 4, 5]                 # full token_ids during prefill


def test_getstate_decode_sends_last_token_only():
    seq = Sequence([1, 2, 3, 4, 5])
    seq.is_prefill = False
    state = seq.__getstate__()
    assert state[-1] == 5                                # only last_token during decode


def test_max_tokens_from_sampling_params():
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
    assert seq.max_tokens == 2
    assert seq.ignore_eos is True