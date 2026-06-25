"""prism-infer benchmark suite.

Measures the metrics that matter for an inference engine, for both Dense and
MoE models:

  - TTFT  (time to first token) -- single-request latency
  - Prefill throughput (tok/s)  -- prompt processing speed
  - Decode TPS (tok/s)          -- steady-state generation speed
  - End-to-end throughput       -- output tokens / wall time
  - Throughput vs batch size    -- a sweep curve
Usage:
    # Dense (CUDA graph): full throughput + latency report
    PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py throughput

    # MoE: must run eager (the MoE path is not CUDA-graph compatible yet)
    PRISM_MODEL=~/models/Qwen3-30B-A3B python bench/bench.py throughput --eager

    # Single-request TTFT
    PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py ttft

    # Throughput vs batch size curve
    PRISM_MODEL=~/models/Qwen3-0.6B python bench/bench.py sweep
"""
import argparse
import os
import time
from random import randint, seed

from prism_infer import LLM, SamplingParams


def _make_requests(num_seqs: int, input_len: int, output_len: int):
    # Random token ids as prompts; ignore_eos so every request runs exactly
    # output_len decode steps (makes throughput numbers comparable / stable).
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_seqs)]
    params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
        for _ in range(num_seqs)
    ]
    return prompts, params


def _run(llm: LLM, prompts, params) -> dict:
    # Drive the engine step-by-step and split timing into prefill vs decode.
    #   step() returns (outputs, num_tokens):
    #     num_tokens > 0  -> prefill step, value = input tokens processed
    #     num_tokens < 0  -> decode step,  -value = batch size = tokens produced
    for p, sp in zip(prompts, params):
        llm.add_request(p, sp)

    prefill_time = decode_time = 0.0
    prefill_tokens = decode_tokens = 0
    first_decode_t = None

    t0 = time.perf_counter()
    while not llm.is_finished():
        t = time.perf_counter()
        _, num_tokens = llm.step()
        dt = time.perf_counter() - t
        if num_tokens > 0:
            prefill_time += dt
            prefill_tokens += num_tokens
        else:
            if first_decode_t is None:
                # First decode step begins only after the initial batch finished
                # prefill, i.e. every initial request already has its first token.
                first_decode_t = time.perf_counter() - t0
            decode_time += dt
            decode_tokens += -num_tokens
    total_time = time.perf_counter() - t0

    # Each sequence's first token is produced by its prefill step; decode steps
    # produce the rest. So total output = decode tokens + one first-token per seq.
    total_out = decode_tokens + len(prompts)

    return {
        "prefill_tput": prefill_tokens / prefill_time if prefill_time else 0.0,
        "decode_tps": decode_tokens / decode_time if decode_time else 0.0,
        "e2e_tput": total_out / total_time if total_time else 0.0,
        "prefill_phase_s": first_decode_t if first_decode_t is not None else total_time,
        "total_time_s": total_time,
        "out_tokens": total_out,
    }


def _build_llm(path: str, eager: bool, max_model_len: int, tp: int = 1) -> LLM:
    # MoE must run eager (CUDA graph not supported on the MoE path yet).
    llm = LLM(path, enforce_eager=eager, tensor_parallel_size=tp, max_model_len=max_model_len)
    # Warm up so the first timed run is not penalized by lazy init / graph capture.
    llm.generate([[1, 2, 3]], SamplingParams(ignore_eos=True, max_tokens=4), use_tqdm=False)
    return llm


def cmd_throughput(args):
    path = os.path.expanduser(os.environ["PRISM_MODEL"])
    llm = _build_llm(path, args.eager, args.max_model_len, args.tp)
    prompts, params = _make_requests(args.num_seqs, args.input_len, args.output_len)
    r = _run(llm, prompts, params)
    print(f"\n=== throughput ({'eager/MoE' if args.eager else 'cuda-graph'}, TP={args.tp}) ===")
    print(f"config        : num_seqs={args.num_seqs} input_len={args.input_len} output_len={args.output_len}")
    print(f"prefill tput  : {r['prefill_tput']:.1f} tok/s")
    print(f"decode TPS    : {r['decode_tps']:.1f} tok/s")
    print(f"e2e tput      : {r['e2e_tput']:.1f} tok/s  ({r['out_tokens']} out tok / {r['total_time_s']:.2f}s)")
    print(f"batch TTFT*   : {r['prefill_phase_s'] * 1000:.1f} ms   (*prefill-phase end; all reqs have 1st token)")


def cmd_ttft(args):
    # True single-request TTFT: time to produce the first token for one prompt.
    path = os.path.expanduser(os.environ["PRISM_MODEL"])
    llm = _build_llm(path, args.eager, args.max_model_len, args.tp)
    prompts, _ = _make_requests(1, args.input_len, 1)
    sp = [SamplingParams(ignore_eos=True, max_tokens=1)]
    llm.add_request(prompts[0], sp[0])
    t0 = time.perf_counter()
    while not llm.is_finished():
        llm.step()
    ttft = (time.perf_counter() - t0) * 1000
    print(f"\n=== ttft ({'eager/MoE' if args.eager else 'cuda-graph'}) ===")
    print(f"config        : num_seqs=1 input_len={args.input_len}")
    print(f"TTFT          : {ttft:.1f} ms")


def cmd_sweep(args):
    path = os.path.expanduser(os.environ["PRISM_MODEL"])
    llm = _build_llm(path, args.eager, args.max_model_len, args.tp)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    print(f"\n=== sweep ({'eager/MoE' if args.eager else 'cuda-graph'}, TP={args.tp}) input_len={args.input_len} output_len={args.output_len} ===")
    print(f"{'batch':>6} | {'decode TPS':>11} | {'e2e tput':>10}")
    print("-" * 34)
    for bs in batch_sizes:
        prompts, params = _make_requests(bs, args.input_len, args.output_len)
        r = _run(llm, prompts, params)
        print(f"{bs:>6} | {r['decode_tps']:>9.1f}   | {r['e2e_tput']:>8.1f}")


def main():
    # Shared options live on a parent parser so they may appear either before
    # or after the subcommand (e.g. `bench.py throughput --num-seqs 8`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--eager", action="store_true", help="disable CUDA graph (required for MoE models)")
    common.add_argument("--input-len", type=int, default=512, help="prompt length (tokens)")
    common.add_argument("--output-len", type=int, default=256, help="generated tokens per request")
    common.add_argument("--num-seqs", type=int, default=64, help="concurrent requests")
    common.add_argument("--max-model-len", type=int, default=4096)
    common.add_argument("--tp", type=int, default=1, help="tensor parallel size (>=2 needs that many GPUs)")

    parser = argparse.ArgumentParser(description="prism-infer benchmark suite", parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("throughput", parents=[common])
    sub.add_parser("ttft", parents=[common])
    sweep = sub.add_parser("sweep", parents=[common])
    sweep.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")

    args = parser.parse_args()
    if "PRISM_MODEL" not in os.environ:
        parser.error("set PRISM_MODEL to a local model directory")
    {"throughput": cmd_throughput, "ttft": cmd_ttft, "sweep": cmd_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
