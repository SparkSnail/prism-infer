import contextlib

import torch
import torch.distributed as dist

from prism_infer.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from prism_infer.layers.embed_head import VocabParallelEmbedding


@contextlib.contextmanager
def simulate_tp(rank: int, world: int):
    # Layers capture tp_rank/tp_size from dist at construction; patch them so we can build
    # a "rank r of a tp_size=world" layer in this single process. Restore afterwards.
    orig_rank, orig_world = dist.get_rank, dist.get_world_size
    dist.get_rank = lambda *a, **k: rank
    dist.get_world_size = lambda *a, **k: world
    try:
        yield
    finally:
        dist.get_rank = orig_rank
        dist.get_world_size = orig_world


def test_column_parallel_partitions_rows():
    O, I = 8, 4
    full = torch.arange(O * I, dtype=torch.float32).reshape(O, I)
    shards = []
    for r in range(2):
        with simulate_tp(r, 2):
            layer = ColumnParallelLinear(I, O)
        assert layer.weight.shape == (O // 2, I)   # column-parallel splits output (dim 0)
        layer.weight_loader(layer.weight, full)
        shards.append(layer.weight.data.clone())
    assert torch.equal(shards[0], full[: O // 2])
    assert torch.equal(shards[1], full[O // 2 :])
    assert torch.equal(torch.cat(shards, dim=0), full)   # no gap / no overlap


def test_column_parallel_forward_concat_matches_full():
    # Each rank computes its output shard; concatenating them must equal the full linear.
    O, I, T = 8, 4, 3
    full = torch.randn(O, I)
    x = torch.randn(T, I)
    ref = x @ full.t()
    outs = []
    for r in range(2):
        with simulate_tp(r, 2):
            layer = ColumnParallelLinear(I, O)
        layer.weight_loader(layer.weight, full)
        outs.append(layer(x))                  # [T, O/2]; F.linear, no collective
    assert torch.allclose(torch.cat(outs, dim=1), ref, atol=1e-5)


def test_row_parallel_partitions_input_cols():
    O, I = 4, 8
    full = torch.arange(O * I, dtype=torch.float32).reshape(O, I)
    shards = []
    for r in range(2):
        with simulate_tp(r, 2):
            layer = RowParallelLinear(I, O)
        assert layer.weight.shape == (O, I // 2)   # row-parallel splits input (dim 1)
        layer.weight_loader(layer.weight, full)
        shards.append(layer.weight.data.clone())
    assert torch.equal(shards[0], full[:, : I // 2])
    assert torch.equal(shards[1], full[:, I // 2 :])
    assert torch.equal(torch.cat(shards, dim=1), full)


def test_merged_column_parallel_gate_up_layout():
    # gate_up_proj fuses gate (shard_id=0) and up (shard_id=1); each rank holds the upper
    # half = its gate chunk, lower half = its up chunk.
    inter, I = 8, 4
    gate_full = torch.arange(inter * I, dtype=torch.float32).reshape(inter, I)
    up_full = torch.arange(inter * I, dtype=torch.float32).reshape(inter, I) + 1000
    for r in range(2):
        with simulate_tp(r, 2):
            layer = MergedColumnParallelLinear(I, [inter, inter])
        assert layer.weight.shape == ((inter + inter) // 2, I)   # [8, 4]
        layer.weight_loader(layer.weight, gate_full, 0)
        layer.weight_loader(layer.weight, up_full, 1)
        w = layer.weight.data
        assert torch.equal(w[: inter // 2], gate_full.chunk(2, 0)[r])
        assert torch.equal(w[inter // 2 :], up_full.chunk(2, 0)[r])


def test_qkv_parallel_gqa_layout():
    # GQA: 4 Q heads, 2 KV heads, head_size=2. Per rank (tp=2): 2 Q heads, 1 KV head.
    hidden, head, n_q, n_kv = 4, 2, 4, 2
    q_full = torch.arange(n_q * head * hidden, dtype=torch.float32).reshape(n_q * head, hidden)
    k_full = torch.arange(n_kv * head * hidden, dtype=torch.float32).reshape(n_kv * head, hidden) + 100
    v_full = torch.arange(n_kv * head * hidden, dtype=torch.float32).reshape(n_kv * head, hidden) + 200
    for r in range(2):
        with simulate_tp(r, 2):
            layer = QKVParallelLinear(hidden, head, n_q, n_kv)
        q_sz = layer.num_heads * layer.head_size       # 2*2 = 4
        k_sz = layer.num_kv_heads * layer.head_size    # 1*2 = 2
        assert layer.weight.shape == (q_sz + 2 * k_sz, hidden)   # [8, 4]
        layer.weight_loader(layer.weight, q_full, "q")
        layer.weight_loader(layer.weight, k_full, "k")
        layer.weight_loader(layer.weight, v_full, "v")
        w = layer.weight.data
        assert torch.equal(w[:q_sz], q_full.chunk(2, 0)[r])               # Q region
        assert torch.equal(w[q_sz : q_sz + k_sz], k_full.chunk(2, 0)[r])  # K region
        assert torch.equal(w[q_sz + k_sz :], v_full.chunk(2, 0)[r])       # V region


def test_vocab_parallel_embedding_partitions_vocab():
    V, D = 8, 3
    full = torch.arange(V * D, dtype=torch.float32).reshape(V, D)
    shards = []
    for r in range(2):
        with simulate_tp(r, 2):
            emb = VocabParallelEmbedding(V, D)
        assert emb.weight.shape == (V // 2, D)
        assert (emb.vocab_start_idx, emb.vocab_end_idx) == (r * V // 2, (r + 1) * V // 2)
        emb.weight_loader(emb.weight, full)
        shards.append(emb.weight.data.clone())
    assert torch.equal(torch.cat(shards, dim=0), full)
