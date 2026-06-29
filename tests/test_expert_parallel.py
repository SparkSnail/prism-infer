import torch

from prism_infer.engine.parallel.expert_parallel import (
    ExpertParallelMoE,
    build_dispatch_plan,
    combine_weighted_sum,
    expert_rank,
)

def test_expert_rank_contiguous_sharding():
    # 8 experts, ep_size=4 -> 2 per rank: [0,1]->r0 [2,3]->r1 [4,5]->r2 [6,7]->r3
    num_local = 2
    assert [expert_rank(e, num_local) for e in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]

def test_build_dispatch_plan_groups_by_rank():
    # num_experts=8, ep_size=4, num_local=2; flat ids: [0,5,2,3]
    topk_ids = torch.tensor([[0, 5], [2, 3]], dtype=torch.int64)
    perm, input_splits, sorted_local_eid = build_dispatch_plan(topk_ids, 2, 4)
    # dst_rank = [0//2, 5//2, 2//2, 3//2] = [0, 2, 1, 1]
    assert input_splits.tolist() == [1, 2, 1, 0]   # sent to r0,r1,r2,r3
    assert perm.tolist() == [0, 2, 3, 1]            # stable argsort of [0,2,1,1]
    assert sorted_local_eid.tolist() == [0, 0, 1, 1]

def test_build_dispatch_plan_splits_sum_to_routed_items():
    torch.manual_seed(0)
    T, k, num_experts, ep_size = 7, 3, 8, 4
    topk_ids = torch.randint(0, num_experts, (T, k), dtype=torch.int64)
    perm, input_splits, sorted_local_eid = build_dispatch_plan(
        topk_ids, num_experts // ep_size, ep_size)
    assert int(input_splits.sum()) == T * k
    assert perm.numel() == T * k
    assert sorted(perm.tolist()) == list(range(T * k))  # valid permutation

def test_combine_weighted_sum_matches_manual():
    T, k, H = 2, 2, 3
    unperm = torch.arange(T * k * H, dtype=torch.float32).view(T * k, H)
    weights = torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=torch.float32)
    y = combine_weighted_sum(unperm, weights, T, k)
    expected = torch.stack([
        0.5 * unperm[0] + 0.5 * unperm[1],
        1.0 * unperm[2] + 0.0 * unperm[3],
    ])
    assert torch.allclose(y, expected)

@torch.no_grad()
def _ref_forward(ep: ExpertParallelMoE, x, topk_ids, topk_weights):
    T, k = topk_ids.shape
    y = torch.zeros_like(x)
    for t in range(T):
        for j in range(k):
            e = int(topk_ids[t, j])
            le = e - ep.local_expert_base   # ep_size=1: base=0, le==e
            out = ep.local_experts[le](x[t:t+1]).squeeze(0)
            y[t] += topk_weights[t, j] * out
    return y


def test_ep_size_one_matches_reference():
    torch.manual_seed(0)
    hidden, inter, num_experts, top_k = 16, 8, 8, 2
    ep = ExpertParallelMoE(hidden, inter, num_experts, ep_size=1, ep_rank=0)
    for p in ep.parameters():
        p.data.normal_(0, 0.1)
    ep.eval()

    T = 5
    x = torch.randn(T, hidden)
    topk_ids = torch.stack([
        torch.randperm(num_experts)[:top_k] for _ in range(T)
    ]).to(torch.int64)
    topk_weights = torch.rand(T, top_k)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    got = ep(x, topk_ids, topk_weights)
    expected = _ref_forward(ep, x, topk_ids, topk_weights)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_ep_requires_divisible_experts():
    import pytest
    with pytest.raises(AssertionError):
        ExpertParallelMoE(16, 8, num_experts=7, ep_size=4, ep_rank=0)
