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
