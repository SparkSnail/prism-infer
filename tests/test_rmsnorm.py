import pytest
import torch

from prism_infer.layers.layernorm import RMSNorm


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    out = x32 * torch.rsqrt(var + eps)
    return out.to(orig_dtype) * weight


def test_rms_forward_matches_reference():
    dim, eps = 128, 1e-6
    norm = RMSNorm(dim, eps=eps)
    norm.weight.data.normal_()
    x = torch.randn(16, dim)
    # rms_forward mutates its input in place (x.float() is a no-op for float32);
    # pass a clone so the reference below sees the pristine x.
    out = norm(x.clone())
    ref = _ref_rmsnorm(x, norm.weight.data, eps)
    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_add_rms_forward_equivalence():
    # add_rms_forward is only used with bf16 in practice. On float32 the internal
    # `residual = x.to(orig_dtype)` would alias the in-place buffer (no copy), so we
    # exercise it in bf16 (its real dtype), where .to() makes a real copy.
    dim, eps = 64, 1e-6
    norm = RMSNorm(dim, eps=eps).to(torch.bfloat16)
    norm.weight.data.normal_()
    x = torch.randn(8, dim, dtype=torch.bfloat16)
    residual = torch.randn(8, dim, dtype=torch.bfloat16)

    out, new_residual = norm(x.clone(), residual.clone())

    # new_residual is the (x + residual) sum cast back to bf16, before norm
    expected_residual = (x.float() + residual.float()).to(torch.bfloat16)
    expected_out = _ref_rmsnorm(expected_residual, norm.weight.data, eps)

    assert torch.allclose(new_residual.float(), expected_residual.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(out.float(), expected_out.float(), atol=2e-2, rtol=2e-2)


def test_fp32_accumulation_no_overflow():
    dim = 256
    norm = RMSNorm(dim)
    x = (torch.randn(4, dim) * 200).to(torch.bfloat16)
    out = norm(x)
    assert torch.isfinite(out.float()).all()


def test_qk_norm_over_head_dim():
    head_dim = 128
    qk_norm = RMSNorm(head_dim, eps=1e-6)
    qk_norm.weight.data.normal_()
    q = torch.randn(10, 4, head_dim)  # [tokens, heads, head_dim]
    out = qk_norm(q.clone())  # clone: norm mutates input in place for float32
    ref = _ref_rmsnorm(q, qk_norm.weight.data, 1e-6)
    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_against_hf_qwen3_rmsnorm():
    mod = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
    HfRMSNorm = getattr(mod, "Qwen3RMSNorm", None)
    if HfRMSNorm is None:
        pytest.skip("Qwen3RMSNorm not available in this transformers version")
    dim, eps = 128, 1e-6
    norm = RMSNorm(dim, eps=eps)
    norm.weight.data.normal_()
    hf = HfRMSNorm(dim, eps=eps)
    hf.weight.data.copy_(norm.weight.data)
    x = torch.randn(16, dim)
    out = norm(x.clone())  # clone: norm mutates input in place for float32
    ref = hf(x)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
