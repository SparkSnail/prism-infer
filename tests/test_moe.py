from types import SimpleNamespace

import torch
import torch.nn.functional as F

from prism_infer.layers.moe import MoE


def _cfg(hidden=16, num_experts=8, top_k=2, moe_inter=8, norm=True):
    # Build a config object for testing with the given parameters; maps to the args of MoE.
    return SimpleNamespace(
        hidden_size=hidden,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter,
        norm_topk_prob=norm,
    )


def _build_moe(cfg):
    # Build a MoE layer with the given config, and initialize weights to normal(0, 0.1) for testing.
    moe = MoE(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
    )
    # Initialize weights to normal(0, 0.1) for testing;
    # this is important to avoid empty experts and ensure stable outputs for reference comparison.
    for p in moe.parameters():
        p.data.normal_(0, 0.1)
    moe.eval()
    return moe


@torch.no_grad()
def _ref_forward(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    # Naive reference implementation of MoE forward pass for testing correctness.
    logits = moe.gate(x)
    probs = F.softmax(logits, dim=-1, dtype=torch.float)
    weights, idx = torch.topk(probs, moe.top_k, dim=-1)
    if moe.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(moe.top_k):
            e = int(idx[t, k])
            y = moe.experts[e](x[t : t + 1])  # [1, hidden]
            out[t] += weights[t, k] * y.squeeze(0)
    return out


@torch.no_grad()
def test_output_matches_naive_reference():
    # Test that the MoE output matches the naive reference implementation for a random input.
    cfg = _cfg()
    moe = _build_moe(cfg)
    x = torch.randn(6, cfg.hidden_size)
    out = moe(x)
    ref = _ref_forward(moe, x)
    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_renorm_weights_sum_to_one():
    # Test that the re-normalized top-k weights sum to 1 for each token.
    cfg = _cfg(norm=True)
    moe = _build_moe(cfg)
    x = torch.randn(10, cfg.hidden_size)
    probs = F.softmax(moe.gate(x), dim=-1, dtype=torch.float)
    weights, _ = torch.topk(probs, moe.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(x.shape[0]), atol=1e-5)


@torch.no_grad()
def test_no_renorm_still_matches_reference():
    # Test that the MoE output still matches the reference implementation when norm_topk_prob is False.
    cfg = _cfg(norm=False)
    moe = _build_moe(cfg)
    assert moe.norm_topk_prob is False
    x = torch.randn(5, cfg.hidden_size)
    out = moe(x)
    ref = _ref_forward(moe, x)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_empty_expert_no_crash():
    # Test that the MoE layer does not crash when some experts are not selected by any token.
    cfg = _cfg(hidden=16, num_experts=32, top_k=2, moe_inter=8)
    moe = _build_moe(cfg)
    x = torch.randn(2, cfg.hidden_size)
    out = moe(x)
    ref = _ref_forward(moe, x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_single_token():
    # Test that the MoE layer handles a single token input correctly.
    cfg = _cfg()
    moe = _build_moe(cfg)
    x = torch.randn(1, cfg.hidden_size)
    out = moe(x)
    ref = _ref_forward(moe, x)
    assert out.shape == (1, cfg.hidden_size)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


@torch.no_grad()
def test_multidim_input_shape_preserved():
    # Test that the MoE layer preserves the shape of multi-dimensional input.
    cfg = _cfg()
    moe = _build_moe(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    out = moe(x)
    assert out.shape == x.shape
