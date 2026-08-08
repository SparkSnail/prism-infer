from unittest.mock import MagicMock, patch
import pytest

from prism_infer.utils.distributed import DistributedContext, PairGroupRegistry


def _make_ctx(rank: int, world_size: int = 4, tp_size: int = 1) -> DistributedContext:
    return DistributedContext(world_size=world_size, rank=rank, tp_size=tp_size)


def test_create_pd_groups_prefill_rank_sets_peer():
    """prefill rank 0 gets pd_peer_rank == decode rank 2 (TP=1, 2P2D)."""
    ctx = _make_ctx(rank=0, world_size=4)
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        ctx.create_pd_groups(prefill_ranks=[0, 1], decode_ranks=[2, 3])
    assert ctx.pd_peer_rank == 2
    assert len(ctx.pd_groups) == 2


def test_create_pd_groups_decode_rank_sets_peer():
    """decode rank 3 gets pd_peer_rank == prefill rank 1 (TP=1, 2P2D)."""
    ctx = _make_ctx(rank=3, world_size=4)
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        ctx.create_pd_groups(prefill_ranks=[0, 1], decode_ranks=[2, 3])
    assert ctx.pd_peer_rank == 1


def test_create_pd_groups_non_member_rank_stays_minus_one():
    """A rank that is neither prefill nor decode keeps pd_peer_rank == -1."""
    ctx = _make_ctx(rank=5, world_size=6)
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        ctx.create_pd_groups(prefill_ranks=[0, 1], decode_ranks=[2, 3])
    assert ctx.pd_peer_rank == -1


def test_create_pd_groups_new_group_called_for_each_pair():
    """new_group is called once per TP shard (i.e. once per P/D pair)."""
    ctx = _make_ctx(rank=0, world_size=8)
    with patch("torch.distributed.new_group", return_value=MagicMock()) as ng:
        ctx.create_pd_groups(
            prefill_ranks=[0, 1, 2, 3],
            decode_ranks=[4, 5, 6, 7],
        )
    assert ng.call_count == 4
    assert len(ctx.pd_groups) == 4


def test_create_pd_groups_new_group_receives_correct_pair():
    """new_group is called with the correct [prefill_rank, decode_rank] per pair."""
    ctx = _make_ctx(rank=0, world_size=4)
    calls_received = []

    def fake_new_group(ranks, **kw):
        calls_received.append(ranks)
        return MagicMock()

    with patch("torch.distributed.new_group", side_effect=fake_new_group):
        ctx.create_pd_groups(prefill_ranks=[0, 1], decode_ranks=[2, 3])

    assert calls_received == [[0, 2], [1, 3]]


def test_create_pd_groups_length_mismatch_raises():
    """Mismatched prefill_ranks / decode_ranks lengths raise AssertionError."""
    ctx = _make_ctx(rank=0, world_size=4)
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        with pytest.raises(AssertionError):
            ctx.create_pd_groups(prefill_ranks=[0], decode_ranks=[1, 2])


def test_create_pd_groups_tp1_single_pair():
    """TP=1 minimal config (1P1D) produces pd_groups of length 1."""
    ctx = _make_ctx(rank=0, world_size=2)
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        ctx.create_pd_groups(prefill_ranks=[0], decode_ranks=[1])
    assert len(ctx.pd_groups) == 1
    assert ctx.pd_peer_rank == 1


def test_pair_group_registry_uses_canonical_five_group_order():
    calls_received = []

    def fake_new_group(ranks, **kwargs):
        calls_received.append(tuple(ranks))
        return f"group-{len(calls_received)}"

    registry = PairGroupRegistry(global_rank=2)
    with patch("torch.distributed.new_group", side_effect=fake_new_group):
        registry.create_all()

    assert calls_received == [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert registry.group_peer("p0--d0", 0) == 1
    assert registry.global_peer("p0--d0", 0) == 2
    assert registry.group_peer("d0--d1", 2) == 1


def test_pair_group_registry_rejects_nonmember_rank_translation():
    registry = PairGroupRegistry(global_rank=0)
    with patch("torch.distributed.new_group", return_value=MagicMock()):
        registry.create_all()
    with pytest.raises(ValueError, match="not a member"):
        registry.group_peer("d0--d1", 0)
