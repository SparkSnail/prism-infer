import torch

from prism_infer.layers.rotary_embedding import RotaryEmbedding


def _make_rope(head_dim=64, max_pos=512, base=1e6):
    return RotaryEmbedding(head_dim, head_dim, max_pos, base)


def _apply_at(rope, x, pos):
    positions = torch.tensor([pos], dtype=torch.long)
    out_q, _ = rope(positions, x, x)
    return out_q


def test_shape_and_dtype_preserved():
    rope = _make_rope()
    q = torch.randn(5, 4, 64)          # [tokens, heads, head_dim]
    k = torch.randn(5, 4, 64)
    positions = torch.arange(5, dtype=torch.long)
    out_q, out_k = rope(positions, q, k)
    assert out_q.shape == q.shape
    assert out_k.shape == k.shape
    assert out_q.dtype == q.dtype


def test_zero_position_is_identity():
    rope = _make_rope()
    x = torch.randn(1, 2, 64)
    out = _apply_at(rope, x, 0)
    assert torch.allclose(out, x, atol=1e-5)


def test_relative_position_invariance():
    rope = _make_rope()
    q = torch.randn(1, 1, 64)
    k = torch.randn(1, 1, 64)

    m, n, s = 3, 7, 5
    dot1 = (_apply_at(rope, q, m) * _apply_at(rope, k, n)).sum()
    dot2 = (_apply_at(rope, q, m + s) * _apply_at(rope, k, n + s)).sum()
    assert torch.allclose(dot1, dot2, atol=1e-4)


def test_different_relative_distance_changes_dot():
    rope = _make_rope()
    q = torch.randn(1, 1, 64)
    k = torch.randn(1, 1, 64)
    dot_close = (_apply_at(rope, q, 1) * _apply_at(rope, k, 2)).sum() 
    dot_far = (_apply_at(rope, q, 1) * _apply_at(rope, k, 100)).sum()
    assert not torch.allclose(dot_close, dot_far, atol=1e-3)
