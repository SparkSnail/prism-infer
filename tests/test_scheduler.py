from types import SimpleNamespace

import pytest

from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.sequence import Sequence, SequenceStatus
from prism_infer.sampling_params import SamplingParams


@pytest.fixture
def small_block():
    prev = Sequence.block_size
    Sequence.block_size = 4
    yield
    Sequence.block_size = prev


def make_scheduler(num_blocks=16, block_size=4, max_num_seqs=4,
                   max_num_batched_tokens=8, eos=999):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)

def test_prefill_respects_token_budget(small_block):
    sch = make_scheduler(max_num_batched_tokens=8)
    a = Sequence([1, 2, 3, 4, 5, 6])      # 6 tokens
    b = Sequence([7, 8, 9])               # 3 tokens -> won't fit in remaining 2
    sch.add(a)
    sch.add(b)
    seqs, is_prefill = sch.schedule()
    assert is_prefill is True
    assert seqs == [a]                    # only A fits this step
    assert a.status == SequenceStatus.RUNNING
    assert b.status == SequenceStatus.WAITING
    assert b in sch.waiting


def test_chunked_prefill_of_first_request(small_block):
    sch = make_scheduler(max_num_batched_tokens=4)
    a = Sequence(list(range(1, 11)))      # 10 tokens, budget only 4 -> chunked
    sch.add(a)
    seqs, is_prefill = sch.schedule()
    assert is_prefill is True
    assert seqs == [a]
    assert a.num_scheduled_tokens == 4    # only first chunk scheduled
    assert a.status == SequenceStatus.WAITING   # prefill not finished yet
    assert a in sch.waiting

def test_decode_step_after_prefill(small_block):
    sch = make_scheduler(max_num_batched_tokens=8)
    a = Sequence([1, 2, 3, 4])
    sch.add(a)
    seqs, is_prefill = sch.schedule()                  # prefill
    sch.postprocess(seqs, [1], is_prefill)             # A appends 1 token -> 5 tokens
    seqs2, is_prefill2 = sch.schedule()                # now decode
    assert is_prefill2 is False
    assert seqs2 == [a]
    assert a.num_scheduled_tokens == 1
    assert a.is_prefill is False

def test_preempt_releases_blocks_and_requeues(small_block):
    sch = make_scheduler()
    a = Sequence([1, 2, 3, 4])
    sch.block_manager.allocate(a, 0)
    a.status = SequenceStatus.RUNNING
    sch.preempt(a)
    assert a.status == SequenceStatus.WAITING
    assert a.is_prefill is True
    assert a.block_table == []
    assert sch.waiting[0] is a                          # requeued at the head


def test_decode_oom_preempts_running_tail(small_block):
    # Only 2 blocks: A and B each take one during prefill. After each appends a token
    # (5 tokens -> needs a 2nd block), decoding A finds no free block and preempts B.
    sch = make_scheduler(num_blocks=2, block_size=4, max_num_batched_tokens=8)
    a = Sequence([1, 2, 3, 4])
    b = Sequence([5, 6, 7, 8])
    sch.add(a)
    sch.add(b)
    seqs, is_prefill = sch.schedule()                  # prefill both
    assert is_prefill is True
    assert set(seqs) == {a, b}
    sch.postprocess(seqs, [1, 1], is_prefill)          # each -> 5 tokens, 1 block each

    seqs2, is_prefill2 = sch.schedule()                # decode: A needs a block, none free
    assert is_prefill2 is False
    assert seqs2 == [a]                                # only A scheduled
    assert b.status == SequenceStatus.WAITING          # B got preempted (running tail)
    assert b in sch.waiting
    assert b.block_table == []
    assert len(a.block_table) == 2                     # A grew into the freed block

def test_decode_self_preempt_returns_empty_and_requeues(small_block):
    # Only 1 block available: A fills it during prefill (4 tokens, block_size=4).
    # On the next decode step A needs a 2nd block but none exist and the running
    # queue is already empty -> A preempts itself.
    # Expected: schedule() returns ([], False) without crashing, A is back in waiting.
    sch = make_scheduler(num_blocks=1, block_size=4, max_num_batched_tokens=8)
    a = Sequence([1, 2, 3, 4])
    sch.add(a)
    seqs, is_prefill = sch.schedule()          # prefill: allocates the 1 block
    assert is_prefill is True
    sch.postprocess(seqs, [1], is_prefill)     # A -> 5 tokens, needs 2nd block next

    seqs2, is_prefill2 = sch.schedule()        # decode: only A, no free block -> self-preempt
    assert seqs2 == []                         # must not crash, must return empty
    assert is_prefill2 is False
    assert a.status == SequenceStatus.WAITING  # A requeued at head of waiting
    assert a in sch.waiting
    assert sch.waiting[0] is a
    assert a.block_table == []                 # KV cache released
    assert a.is_prefill is True                # reset to prefill mode


def test_decode_self_preempt_recovers_on_next_step(small_block):
    # After self-preemption the scheduler must be able to re-prefill A on the
    # very next step (the released block is back in the free pool).
    # Use 2 blocks so that after self-preemption (A has 5 tokens = 2 blocks needed)
    # there are enough free blocks to re-prefill A.
    sch = make_scheduler(num_blocks=2, block_size=4, max_num_batched_tokens=8)
    a = Sequence([1, 2, 3, 4])
    sch.add(a)
    seqs, is_prefill = sch.schedule()          # prefill: allocates block 0
    sch.postprocess(seqs, [1], is_prefill)     # A -> 5 tokens

    # Force OOM by temporarily exhausting all blocks so the first decode triggers
    # self-preemption. Borrow block 1 via a dummy sequence, then release it after
    # self-preemption so the re-prefill can proceed.
    dummy = Sequence([10, 11, 12, 13])
    sch.block_manager.allocate(dummy, 0)       # occupy block 1 (the only free one)

    seqs2, is_prefill2 = sch.schedule()        # decode: A needs block, none free -> self-preempt
    assert seqs2 == []
    assert a.status == SequenceStatus.WAITING
    assert a.block_table == []

    sch.block_manager.deallocate(dummy)        # free block 1 again

    seqs3, is_prefill3 = sch.schedule()        # A should be re-prefilled successfully
    assert is_prefill3 is True
    assert seqs3 == [a]
    assert a.status == SequenceStatus.RUNNING


def test_postprocess_finishes_on_eos(small_block):
    sch = make_scheduler(eos=0)
    a = Sequence([1, 2, 3])
    sch.add(a)
    seqs, is_prefill = sch.schedule()
    sch.postprocess(seqs, [0], is_prefill)             # token 0 == eos
    assert a.is_finished is True
    assert a.status == SequenceStatus.FINISHED
    assert sch.is_finished() is True                   # both queues empty


def test_postprocess_finishes_on_max_tokens(small_block):
    sch = make_scheduler(eos=999)
    a = Sequence([1, 2, 3], SamplingParams(max_tokens=1, ignore_eos=True))
    sch.add(a)
    seqs, is_prefill = sch.schedule()
    sch.postprocess(seqs, [7], is_prefill)             # 1 completion token == max_tokens
    assert a.is_finished is True
    assert a.status == SequenceStatus.FINISHED