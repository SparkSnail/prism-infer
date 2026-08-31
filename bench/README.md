# Runtime benchmarks

`bench.py` is the public benchmark entry point for the prism-infer runtime. It
drives the engine's public loop (`add_request`, `step`, `is_finished`) and is
intended for repeatable runtime comparisons. HTTP serving and multi-node
request-load measurements belong to [prism-serve](https://github.com/SparkSnail/prism-serve).

## Metrics

| Metric | Definition |
| --- | --- |
| TTFT | Time from enqueue to the first output token for one request |
| Prefill throughput | Prompt tokens processed per second during prefill |
| Decode TPS | Generated tokens per second during decode steps |
| End-to-end throughput | Completed output tokens divided by total wall time |
| Batch TTFT | Time until the first decode step for a batch; useful for comparing prefill latency |

All timings are host wall-clock measurements with CUDA synchronized around each
engine step. `idle_steps` counts no-progress steps in the result; a run stops
with an error after `--max-idle-steps` consecutive idle steps.

## Requirements

- A CUDA-capable NVIDIA GPU and the runtime dependencies installed from this repository.
- A local model directory compatible with prism-infer. Pass `--model PATH` or set `PRISM_MODEL`.
- The same model, software revision, GPU topology, environment, and workload when comparing runs.

The benchmark generates synthetic, deterministic token-id prompts inside the
model's embedding/tokenizer vocabulary. Each repetition and sweep cell receives
a distinct seed, so the benchmark does not intentionally reuse prefixes from an
earlier cell. Sampling is `ignore_eos=True`; set `--temperature 0` for greedy
decoding.

## Quick start

Run commands from the repository root:

```bash
# Dense model (CUDA graphs are enabled by default)
python bench/bench.py ttft --model ~/models/Qwen3-0.6B
python bench/bench.py throughput --model ~/models/Qwen3-0.6B
python bench/bench.py sweep --model ~/models/Qwen3-0.6B --batch-sizes 1,2,4,8,16,32,64

# MoE model (the current MoE path requires eager execution)
python bench/bench.py throughput --model ~/models/Qwen3-30B-A3B --eager
python bench/bench.py sweep --model ~/models/Qwen3-30B-A3B --eager --batch-sizes 1,4,16,32
```

The most frequently tuned workload options are `--input-len`, `--output-len`,
`--num-seqs`, and `--max-model-len`. `--tp N` enables tensor parallelism and
`--ep N` enables expert parallelism; TP and EP cannot both be greater than one.
Use `--eager` to make the execution mode explicit for a comparison. The
engine-capacity options `--max-num-batched-tokens`, `--max-num-seqs`,
`--num-kvcache-blocks`, and `--gpu-memory-utilization` are also recorded in
structured output.

## Reproducible output

Use several repetitions and write the raw samples as JSON:

```bash
python bench/bench.py throughput \
  --model ~/models/Qwen3-0.6B \
  --input-len 512 --output-len 256 --num-seqs 64 \
  --seed 0 --repetitions 3 --warmup-seqs 1 \
  --output results/qwen3-0.6b-throughput.json
```

The JSON envelope contains `schema_version`, the resolved model path, workload
and engine settings, sampling and prompt-generation policy, warm-up settings,
runtime metadata, every measured sample (including its prompt digest), and a
median `summary`. Throughput
fields use explicit units such as `prefill_tokens_per_second`,
`decode_tokens_per_second`, `output_tokens_per_second`, and `total_time_s`.
`sweep` stores one result object per batch size. Use `--format json` when
another tool consumes the report on stdout; the default `text` format remains
convenient for interactive runs.

For a meaningful comparison, keep `--seed`, `--repetitions`, warm-up settings,
and all engine-capacity options fixed. The automatically sized KV cache depends
on available GPU memory; specify `--num-kvcache-blocks` when exact cache
capacity is part of the experiment. Repetitions reuse one initialized engine,
so these are warm-process measurements; use separate invocations when a
cold-start measurement is required.

## Multi-GPU runtime measurements

Compare the same workload at different parallelism settings:

```bash
python bench/bench.py throughput --model ~/models/Qwen3-0.6B --tp 1
python bench/bench.py throughput --model ~/models/Qwen3-0.6B --tp 2

python bench/bench.py throughput --model ~/models/Qwen3-30B-A3B \
  --eager --ep 2 --num-seqs 32 --output-len 64
```

If the host does not provide GPU peer-to-peer access, set
`NCCL_P2P_DISABLE=1` and record that environment in the JSON output. Parallel
results are topology-specific; they are not a claim that adding GPUs always
improves latency.

## TP parity diagnostic

[`tp_parity.py`](tp_parity.py) is a correctness diagnostic, separate from the
runtime performance commands. It compares token streams from TP=1 and TP=2
under a near-greedy setting. A parity result is evidence for that particular
model, prompt, precision, and execution path; it is not a throughput number or
a proof of all possible sampled sequences.

```bash
python bench/tp_parity.py --model ~/models/Qwen3-0.6B --max-tokens 32
```

Do not combine parity output with benchmark JSON when publishing performance
results. For serving-level latency, throughput, and multi-node comparisons, use
the benchmark and evidence workflow in prism-serve.

## Scope

These scripts measure the local inference runtime under a controlled workload.
They are useful for regression checks and engineering comparisons, but they do
not establish production SLOs, capacity guarantees, or end-to-end serving
reliability.
