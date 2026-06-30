# Benchmarks

`bench.py` is the prism-infer benchmark suite. It measures the metrics that matter for an inference engine, for both Dense and MoE models, by driving the engine's public step loop (`add_request` / `step` / `is_finished`), without depending on engine internals. Model path comes from the `PRISM_MODEL` env var (a local model directory).

## Metrics

| Metric | Meaning |
|---|---|
| TTFT | Time to first token, single request (latency) |
| Prefill throughput | Prompt tokens processed per second |
| Decode TPS | Steady-state generated tokens per second |
| End-to-end throughput | Total output tokens / wall time |
| Throughput vs batch | `sweep` curve of decode TPS / e2e over batch sizes |

## Running

Model path comes from `PRISM_MODEL` (a local model directory).

```bash
# Dense (CUDA graph)
PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py ttft
PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py throughput
PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py sweep

# MoE MUST run with --eager (the MoE path is not CUDA-graph compatible yet)
PRISM_MODEL=~/models/Qwen3-30B-A3B python bench/bench.py throughput --eager
PRISM_MODEL=~/models/Qwen3-30B-A3B python bench/bench.py sweep --eager --batch-sizes 1,4,16,32
```

Common flags: `--num-seqs` (concurrency), `--input-len`, `--output-len`, `--max-model-len`, `--batch-sizes` (sweep), `--eager`.

## Multi-GPU (tensor parallel)

Pass `--tp N` to run on N GPUs (needs N visible GPUs). `tp_parity.py` checks that
TP=N produces the same tokens as TP=1.

```bash
# TP scaling throughput (1 vs 2 GPUs)
PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py throughput --tp 1
PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py throughput --tp 2

# TP=1 vs TP=2 token parity (first-token + full-sequence)
python bench/tp_parity.py --model ~/models/Qwen3-0.6B --max-tokens 32
```

> [!Note]
> On hosts where GPU peer-to-peer (P2P) over PCIe is not available, NCCL **hangs** during the first collective (both GPUs stuck at 100% util).   
> Prefix multi-GPU runs with `NCCL_P2P_DISABLE=1`:
> ```bash
> NCCL_P2P_DISABLE=1 python bench/tp_parity.py --model ~/models/Qwen3-0.6B
> ```

### TP parity result (2xA800)

TP=1 vs TP=2, `temperature=1e-6` (near-argmax), eager:

- **Dense (Qwen3-0.6B)**: first token matches (both ` Paris`), first 5 tokens identical, diverge at index 5 at a near-tie branch.
- **MoE (Qwen3-30B-A3B)**: first token matches, first 13 tokens identical, diverge at index 14, this also exercises the expert-sharding TP path (MergedColumn/Row parallel experts), confirming MoE TP forward is correct.


### TP scaling (2xA800, NCCL_P2P_DISABLE=1)

**Dense (Qwen3-0.6B, cuda-graph)**: num_seqs=64, input 512 / output 256:

| | prefill tok/s | decode TPS | e2e tok/s | batch-TTFT |
|---|---|---|---|---|
| TP=1 | 68 213 | 10 670 | 8 150 | 487 ms |
| TP=2 | 41 918 | 7 495 | 5 536 | 791 ms |

**MoE (Qwen3-30B-A3B, eager)**: num_seqs=32, input 512 / output 64:

| | prefill tok/s | decode TPS | e2e tok/s | batch-TTFT |
|---|---|---|---|---|
| TP=1 | 5 360 | 28.6 | 27.8 | 3 981 ms |
| TP=2 | 2 259 | 21.6 | 20.4 | 8 403 ms |

#### Analysis
**TP=2 is slower than TP=1 in both cases (~0.68x Dense, ~0.73x MoE)**, it's expected instead of a regression. At these sizes the per-layer all_reduce communication outweighs the compute saved by splitting, and with P2P disabled the cross-GPU path falls back to the slower host/SHM transport. Both models fit on a single 80GB A800, so TP is unnecessary here. TP pays off only when a single GPU is compute- or memory-bound: it is a capacity enabler (fit bigger models), not a small-model speedup.

## Baseline results

### Environment

| | |
|---|---|
| GPU | NVIDIA A800-80G PCIe |
| Precision | BF16, tensor_parallel_size=1 |
| Stack | torch 2.5.1+cu124, Python 3.12, flash-attn 2.7.4.post1 |
| Workload | input_len=512, output_len=256 (throughput/sweep); TTFT = single request, input 512 |

`Dense` runs use CUDA graph (default). `MoE` runs use `--eager` (required).

### Dense: Qwen3-0.6B (CUDA graph)

- **TTFT** (single, input 512): **29.7 ms**
- **throughput** (num_seqs=64): prefill **69 977 tok/s**, decode **10 707 TPS**, e2e **8 221 tok/s**, batch-TTFT 475 ms (16 384 out tok / 1.99 s)

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| decode TPS | 402 | 615 | 1 070 | 2 110 | 4 025 | 7 024 | 10 714 |
| e2e tok/s | 386 | 453 | 1 040 | 2 051 | 3 888 | 6 609 | 9 730 |

### MoE: Qwen3-30B-A3B (eager)

- **TTFT** (single, input 512): **1 403 ms**
- **throughput** (num_seqs=64): prefill **5 809 tok/s**, decode **47.9 TPS**, e2e **47.3 tok/s**, batch-TTFT 6 650 ms (16 384 out tok / 346 s)

| batch | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| decode TPS | 2.4 | 6.6 | 17.9 | 26.8 | 47.9 |

### Observations

- **Dense (CUDA graph)** scales decode throughput near-linearly up to batch 64 (402 -> 10 714 TPS), as expected when per-step kernel launch overhead is amortized by the captured graph.
- **MoE (eager)** is **per-step-overhead bound**: decode TPS also scales near-linearly with batch (2.4-> 47.9 TPS), and single-stream decode is only ~2.4 tok/s (~0.42 s/token). The eager path has no CUDA graph and runs Python-side expert routing every step.
- This MoE eager number is the **"before" baseline** for future work on CUDA-graph support / routing optimization for the MoE path.

## Multi-GPU (expert parallel)

For MoE models, `--ep N` shards experts across N GPUs (true expert parallelism: all-to-all dispatch/combine), as opposed to TP-of-experts.

```bash
# EP throughput
NCCL_P2P_DISABLE=1 PRISM_MODEL=~/models/Qwen3-30B-A3B python bench/bench.py throughput --eager --ep 2 --num-seqs 32 --output-len 64
```

### EP scaling (Qwen3-30B-A3B, 2xA800, eager)

num_seqs=32, input 512 / output 64:

| | prefill tok/s | decode TPS | e2e tok/s | batch-TTFT |
|---|---|---|---|---|
| EP=1 | 5 633 | 31.6 | 30.7 | 3 792 ms |
| EP=2 | 3 369 | 43.0 | 39.6 | 5 429 ms |

**EP=2 decode TPS +36%, e2e +29%** vs EP=1, expert sharding halves per-card MoE compute. Prefill and TTFT are slower (~-40%) because each EP step requires an all-to-all collective; with P2P disabled the cross-GPU path uses the slower host/SHM transport, and prefill has more tokens per step so the communication overhead is higher. EP is a capacity enabler and decode accelerator, not a prefill speedup.
