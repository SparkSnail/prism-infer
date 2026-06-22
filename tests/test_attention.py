import pytest
import torch

pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("attention UT requires CUDA GPU", allow_module_level=True)

from prism_infer.layers.attention import Attention  # noqa: E402
from prism_infer.utils.context import set_context, reset_context  # noqa: E402


def _ref_gqa_causal(q, k, v, scale):
    # q: [T, nh, hd], k/v: [T, nkv, hd]
    T, nh, hd = q.shape
    nkv = k.shape[1]
    group = nh // nkv
    k_rep = k.repeat_interleave(group, dim=1)  # [T, nh, hd]
    v_rep = v.repeat_interleave(group, dim=1)
    qh = q.transpose(0, 1).float()             # [nh, T, hd]
    kh = k_rep.transpose(0, 1).float()
    vh = v_rep.transpose(0, 1).float()
    scores = (qh @ kh.transpose(-1, -2)) * scale  # [nh, T, T]
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh                              # [nh, T, hd]
    return out.transpose(0, 1)                   # [T, nh, hd]


def test_gqa_prefill_matches_reference():
    torch.manual_seed(0)
    T, nh, nkv, hd = 16, 8, 2, 64
    scale = hd ** -0.5
    dev = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(T, nh, hd, device=dev, dtype=dtype)
    k = torch.randn(T, nkv, hd, device=dev, dtype=dtype)
    v = torch.randn(T, nkv, hd, device=dev, dtype=dtype)

    attn = Attention(nh, hd, scale, nkv)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=T,
        max_seqlen_k=T,
        slot_mapping=None,
        block_tables=None,
    )
    try:
        out = attn(q, k, v)  # [T, nh, hd]
    finally:
        reset_context()

    ref = _ref_gqa_causal(q, k, v, scale).to(dtype)
    assert out.shape == (T, nh, hd)
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
