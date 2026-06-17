<p align="center">
  <img src="assets/banner.png" width="640" alt="Prism-Infer"/>
</p>

<h3 align="center">A from-scratch LLM inference engine</h3>

<p align="center">
  <a href="#status"><b>Status</b></a> ·
  <a href="#installation"><b>Installation</b></a> ·
  <a href="#quick-start"><b>Quick Start</b></a>
</p>

**prism-infer** is a from-scratch LLM inference engine focused on getting single-instance inference **correct, fast, and memory-efficient**: KV cache management, two-phase scheduling, and Qwen3 MoE/Dense forward.

It started as a fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and evolves it by porting battle-tested **scheduling / state / fault-tolerance** mechanisms from the big-data world (Spark/Flink/Kafka) into the inference engine layer.

> [!WARNING]
> **Early work-in-progress project.** Only the checked items below are implemented. Everything else is planned and **not usable yet**. APIs and structure will change.

## Status

Implemented (inherited from the nano-vllm base, working):

- [x] **PagedAttention**: fixed-size KV blocks, prefix caching via rolling hash
- [x] **Two-phase scheduling**: prefill + decode continuous batching with preemption
- [x] **Tensor parallelism**: column/row-parallel linear, fused QKV / gate-up projections
- [x] **CUDA graph**: capture for decode, Torch compilation for fused ops
- [x] **Qwen3 Dense** forward: GQA + QK-norm + RoPE (θ=1e6)

In progress:

- [ ] **Qwen3-MoE** forward: router top-k of N + SwiGLU experts with re-norm.

Planned (not started):

- [ ] KV access-order LRU + CPU offload, tiered KV backend
- [ ] PD (prefill/decode) disaggregation + KV transfer
- [ ] Multi-GPU expert parallelism (EP)
- [ ] MTP speculative decoding


## Installation

Requires an NVIDIA CUDA GPU (`flash-attn` / `triton`). Windows users: use WSL2.

```bash
git clone https://github.com/SparkSnail/prism-infer.git
cd prism-infer
pip install -e .
```

## Model Download

| Model | Type | Use |
|-------|------|-----|
| **Qwen3-0.6B** | Dense | smoke test / Dense path / `example.py` / `bench.py` |
| **Qwen3-30B-A3B** | MoE | exercises the MoE code path (needs large GPU memory) |

```bash
pip install modelscope   # or use huggingface-cli

# Dense (default)
modelscope download --model Qwen/Qwen3-0.6B --local_dir ~/huggingface/Qwen3-0.6B

# MoE (large, ~60GB BF16)
modelscope download --model Qwen/Qwen3-30B-A3B --local_dir ~/models/Qwen3-30B-A3B
```

## Quick Start

```python
from prism_infer import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, prism-infer."], sampling_params)
print(outputs[0]["text"])
```

`example.py` and `bench.py` read the model path from the `PRISM_MODEL` env var (default: Dense 0.6B):

```bash
python example.py
python bench.py
PRISM_MODEL=~/models/Qwen3-30B-A3B python example.py   # MoE
```

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests/
```

Tests need a CUDA GPU (the package imports triton/flash-attn at import time). `test_parity.py` needs real weights via the `PRISM_TEST_MODEL` env var.


## Acknowledgements & License

Built on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by GeeeekExplorer (Xingkai Yu). MIT licensed, see [LICENSE](LICENSE) for the dual copyright notice.