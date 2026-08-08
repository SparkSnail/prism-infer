<p align="center">
  <img src="assets/banner.png" width="640" alt="Prism-Infer"/>
</p>

<h3 align="center">A minimal LLM inference engine</h3>

<p align="center">
  <a href="#features"><b>Features</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#quick-start"><b>Quick Start</b></a>
</p>

<p align="center">
  <a href="https://github.com/SparkSnail/prism-infer/actions/workflows/ci.yml">
    <img src="https://github.com/SparkSnail/prism-infer/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  </a>
</p>

**prism-infer** is a minimal LLM inference engine focused on getting single-instance inference **correct, fast, and memory-efficient**: KV cache management, two-phase scheduling, and Qwen3 MoE/Dense forward.

It started as a fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and extends it with **scheduling / state / failure-handling and fencing** mechanisms at the inference engine layer.

## Features

- [x] **PagedAttention**: fixed-size KV blocks, prefix caching via rolling hash
- [x] **Two-phase scheduling**: prefill + decode continuous batching with preemption
- [x] **Tensor parallelism**: column/row-parallel linear, fused QKV / gate-up projections
- [x] **Expert parallelism**: all-to-all dispatch/combine for MoE across multiple GPUs
- [x] **CUDA graph**: capture for decode, Torch compilation for fused ops
- [x] **KV cache LRU + CPU offload**: access-order eviction, pinned CPU pool, recall on prefix hit
- [x] **Qwen3 Dense** forward: GQA + QK-norm + RoPE (`theta=1e6`)
- [x] **Qwen3-MoE** forward: router top-k of N + SwiGLU experts with re-norm
- [x] **PD disaggregation**: prefill-only / decode-only engine modes, KV transfer via NCCL P2P
- [x] **KV snapshot and migration primitives**: aligned/unaligned/incremental helpers, three-way handshake, watchdog

## Installation

Runtime support:

- Supported: Linux (Ubuntu), or Windows + WSL2 (Ubuntu)
- Not supported: native Windows Python runtime, macOS

### Environment setup (prerequisites)

prism-infer needs an NVIDIA CUDA GPU plus a matching PyTorch and `flash-attn`.

1. **PyTorch** install the build matching your CUDA version, see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).
2. **flash-attn** CUDA-only, and **not** installed by `pip install -e .` 
 Pick one:
   - *Prebuilt wheel (recommended, fast):* download the wheel matching your torch / CUDA / Python / C++ ABI from the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases), then `pip install <downloaded-wheel>.whl`.
   - *From source (needs CUDA toolkit / `nvcc`):* `pip install flash-attn --no-build-isolation`.

Verify the environment:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import flash_attn, triton; print('ok')"
```

### Install

Same commands on Linux and WSL2 (Ubuntu):

```bash
git clone https://github.com/SparkSnail/prism-infer.git
cd prism-infer
pip install -e .
```

Windows users: run the commands above inside a WSL2 Ubuntu shell.

## Model Download

| Model | Type | Use |
|-------|------|-----|
| **Qwen3-0.6B** | Dense | smoke test / Dense path / `example.py` / `bench/bench.py` |
| **Qwen3-30B-A3B** | MoE | exercises the MoE code path via `example.py` / `bench/bench.py --eager` (needs large GPU memory) |

```bash
pip install modelscope   # or use huggingface-cli

# Dense (default)
modelscope download --model Qwen/Qwen3-0.6B --local_dir ~/models/Qwen3-0.6B

# MoE (large, ~60GB BF16)
modelscope download --model Qwen/Qwen3-30B-A3B --local_dir ~/models/Qwen3-30B-A3B
```

## Quick Start

Download a model (see [Model Download](#model-download)), then run the example:

```bash
PRISM_MODEL=~/models/Qwen3-0.6B python example.py
```

`example.py` read the model directory from the `PRISM_MODEL` env var (a local model folder; defaults to `~/models/Qwen3-0.6B/`).

## PD Disaggregation

prism-infer supports **prefill-decode disaggregation**: the prefill and decode phases run in separate processes, with KV cache transferred between them via NCCL P2P.

### Experimental 2P2D snapshot

The multi-node worker and control-plane path is experimental, not production-ready. A frozen two-node, four-GPU 2P2D campaign executed all 35 test slots: 31 passed, 3 failed, and 1 was blocked. Four of five semantic evidence packets passed; final cleanup failed.

Gateway restart remains a known limitation. One decode worker exited with `SIGSEGV` (exit 139), and tunnel recovery failed while port `18080` remained bound. Both issues are unresolved in this snapshot.

### Single-node (two GPUs)

```bash
# Terminal 1: prefill process (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m prism_infer.engine.pd_runner \
    --mode prefill-only --model ~/models/Qwen3-0.6B \
    --master-addr localhost --master-port 29500 \
    --world-size 2 --rank 0

# Terminal 2: decode process (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m prism_infer.engine.pd_runner \
    --mode decode-only --model ~/models/Qwen3-0.6B \
    --master-addr localhost --master-port 29500 \
    --world-size 2 --rank 1
```

### Multi-node

```bash
# Node A (rank 0, prefill)
NCCL_SOCKET_IFNAME=eth0 python -m prism_infer.engine.pd_runner \
    --mode prefill-only --model /shared/Qwen3-0.6B \
    --master-addr <node-A-ip> --master-port 29500 \
    --world-size 2 --rank 0

# Node B (rank 1, decode)
NCCL_SOCKET_IFNAME=eth0 python -m prism_infer.engine.pd_runner \
    --mode decode-only --model /shared/Qwen3-0.6B \
    --master-addr <node-A-ip> --master-port 29500 \
    --world-size 2 --rank 1
```

### Unified baseline (parity check)

```bash
python -m prism_infer.engine.pd_runner \
    --mode unified --model ~/models/Qwen3-0.6B
```

The parity test checks whether greedy output (`temperature=0`) from PD mode matches unified mode token-for-token.

## Testing

**Unit tests run on CPU.**

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

The CPU suite covers scheduling, block management, KV transfer, KV snapshot/migration,
PD connector logic, distributed context, and tensor/expert parallel math.

**E2E Parity Tests**

The parity tests (single-step and end-to-end token parity vs HuggingFace) need a CUDA GPU, flash-attn, and a local model. They are skipped unless `PRISM_TEST_MODEL` points to a local model directory:

```bash
PRISM_TEST_MODEL=~/models/Qwen3-0.6B python -m pytest tests/ -q
```

**MoE E2E Parity Tests**

MoE-specific tests use a separate env var so they don't interfere with the Dense test suite. Run them after the standard suite, not together:

```bash
# Single-GPU MoE forward + scheduler parity (needs >=40 GB VRAM)
PRISM_TEST_MOE_MODEL=~/models/Qwen3-30B-A3B python -m pytest tests/test_parity_moe_e2e.py -v

# EP=2 logits parity (needs >=2 GPUs; run after the above to avoid OOM)
PRISM_TEST_MOE_MODEL=~/models/Qwen3-30B-A3B python -m pytest tests/test_parity_ep.py -v
```


## Benchmark

`bench/bench.py` measures **TTFT**, **prefill throughput**, **decode TPS**, and a **throughput-vs-batch-size** curve, for both Dense and MoE models.

More details refer [bench/README.md](bench/README.md).

## License

[MIT](LICENSE)
