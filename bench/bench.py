"""Public runtime benchmark CLI for prism-infer."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZES = "1,2,4,8,16,32,64"
MetricSample = dict[str, Any]


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Add options shared by every benchmark command.

    Subparsers use suppressed defaults so an option supplied before the command
    is not overwritten by a child parser's default value.
    """
    parser.add_argument(
        "--model",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="local model directory (falls back to PRISM_MODEL)",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="disable CUDA graphs (required for MoE models)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 512,
        help="prompt length in tokens (default: 512)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 256,
        help="generated tokens per request (default: 256)",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 64,
        help="concurrent requests for throughput (default: 64)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 4096,
        help="engine context limit in tokens (default: 4096)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 16384,
        help="maximum tokens scheduled in one engine step",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 512,
        help="engine request capacity",
    )
    parser.add_argument(
        "--num-kvcache-blocks",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else -1,
        help="fixed KV-cache blocks; -1 lets the engine size them",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=argparse.SUPPRESS if suppress_defaults else 0.9,
        help="fraction of GPU memory available to the KV cache",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 1,
        help="tensor parallel size",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 1,
        help="expert parallel size for MoE models",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS if suppress_defaults else 0.6,
        help="sampling temperature (0 selects greedy decoding)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 0,
        help="base seed for prompt generation and sampling",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 1,
        help="measured repetitions; summaries report medians",
    )
    parser.add_argument(
        "--warmup-seqs",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 1,
        help="requests used for warm-up before measurement",
    )
    parser.add_argument(
        "--warmup-output-len",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 4,
        help="generated tokens per warm-up request",
    )
    parser.add_argument(
        "--max-idle-steps",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 1000,
        help="fail after this many no-progress engine steps",
    )
    parser.add_argument(
        "--batch-sizes",
        default=argparse.SUPPRESS if suppress_defaults else DEFAULT_BATCH_SIZES,
        help="comma-separated sizes for sweep (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default=argparse.SUPPRESS if suppress_defaults else "text",
        help="report format (default: text)",
    )
    parser.add_argument(
        "--output",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="write the complete JSON result to this path",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark parser without importing CUDA-dependent modules."""
    parser = argparse.ArgumentParser(description="prism-infer runtime benchmarks")
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("throughput", "measure prefill, decode, and end-to-end throughput"),
        ("ttft", "measure single-request time to first token"),
        ("sweep", "measure throughput across batch sizes"),
    ):
        child = subparsers.add_parser(name, help=help_text, description=help_text)
        _add_common_arguments(child, suppress_defaults=True)
    return parser


def _resolve_model(model: str | None) -> str:
    """Resolve and validate a local model directory."""
    value = model or os.environ.get("PRISM_MODEL")
    if not value:
        raise ValueError("set --model or PRISM_MODEL to a local model directory")
    path = Path(os.path.expanduser(value))
    if not path.is_dir():
        raise ValueError(f"model directory does not exist: {path}")
    return str(path.resolve())


def _parse_batch_sizes(raw: str) -> list[int]:
    """Parse a non-empty, strictly positive, duplicate-free size list."""
    if not isinstance(raw, str):
        raise ValueError("batch sizes must be provided as a comma-separated string")
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            raise ValueError("batch sizes must be comma-separated positive integers")
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(
                "batch sizes must be comma-separated positive integers"
            ) from exc
        if value <= 0:
            raise ValueError("batch sizes must be positive")
        if value in values:
            raise ValueError("batch sizes must not contain duplicates")
        values.append(value)
    if not values:
        raise ValueError("at least one batch size is required")
    return values


def _validate_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    """Validate cross-option constraints and resolve the model path."""
    try:
        args.model = _resolve_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    for name in (
        "input_len",
        "output_len",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "tp",
        "ep",
        "repetitions",
        "warmup_output_len",
        "max_idle_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_seqs <= 0:
        parser.error("--num-seqs must be positive")
    if args.warmup_seqs < 0:
        parser.error("--warmup-seqs must be non-negative")
    if args.num_kvcache_blocks == 0 or args.num_kvcache_blocks < -1:
        parser.error("--num-kvcache-blocks must be -1 or positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.tp > 8:
        parser.error("--tp must be no greater than 8")
    if args.ep > 8:
        parser.error("--ep must be no greater than 8")
    if args.tp > 1 and args.ep > 1:
        parser.error("--tp and --ep cannot both be greater than 1")
    effective_output_len = 1 if args.cmd == "ttft" else args.output_len
    if args.input_len + effective_output_len > args.max_model_len:
        parser.error("input plus output length exceeds --max-model-len")
    warmup_input_len = min(args.input_len, 3)
    if warmup_input_len + args.warmup_output_len > args.max_model_len:
        parser.error("warm-up length exceeds --max-model-len")
    try:
        args.batch_sizes = _parse_batch_sizes(args.batch_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _load_torch() -> Any | None:
    """Load torch only when a benchmark is actually being run."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_synchronize(torch_module: Any | None) -> None:
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _seed_runtime(seed_value: int) -> None:
    """Seed host and device RNGs used by prompt generation and sampling."""
    random.seed(seed_value)
    torch_module = _load_torch()
    if torch_module is None:
        return
    torch_module.manual_seed(seed_value)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed_value)


def _tokenizer_vocab_size(llm: Any) -> int:
    config = getattr(getattr(llm, "model_runner", None), "config", None)
    hf_config = getattr(config, "hf_config", None)
    model_vocab_size = getattr(hf_config, "vocab_size", None)
    if isinstance(model_vocab_size, int) and model_vocab_size > 0:
        return model_vocab_size
    tokenizer = getattr(llm, "tokenizer", None)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(tokenizer_vocab_size, int) and tokenizer_vocab_size > 0:
        return tokenizer_vocab_size
    try:
        size = len(tokenizer)
    except TypeError:
        size = getattr(tokenizer, "vocab_size", 0)
    if not isinstance(size, int) or size < 1:
        raise RuntimeError("the model tokenizer does not expose a usable vocabulary size")
    return size


def _make_requests(
    num_seqs: int,
    input_len: int,
    output_len: int,
    seed_value: int,
    temperature: float,
    vocab_size: int,
) -> tuple[list[list[int]], list[Any]]:
    """Build deterministic token-id prompts within the model vocabulary."""
    from prism_infer.sampling_params import SamplingParams

    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    rng = random.Random(seed_value)
    prompts: list[list[int]] = []
    for sequence_index in range(num_seqs):
        prompt = [rng.randrange(vocab_size) for _ in range(input_len)]
        if prompt:
            # Make each measured case distinct even when one engine retains
            # prefix-cache blocks from an earlier repetition or batch size.
            prompt[0] = (seed_value + sequence_index) % vocab_size
        prompts.append(prompt)
    params = [
        SamplingParams(
            temperature=temperature,
            ignore_eos=True,
            max_tokens=output_len,
        )
        for _ in range(num_seqs)
    ]
    return prompts, params


def _prompt_digest(prompts: list[list[int]]) -> str:
    encoded = json.dumps(prompts, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run(
    llm: Any,
    prompts: list[list[int]],
    params: list[Any],
    max_idle_steps: int = 1000,
) -> MetricSample:
    """Run a batch and return phase timings plus completion counts."""
    if not prompts:
        raise ValueError("at least one prompt is required")
    if len(prompts) != len(params):
        raise ValueError("prompt and sampling-parameter counts must match")
    torch_module = _load_torch()
    _cuda_synchronize(torch_module)
    t0 = time.perf_counter()
    for prompt, sampling_params in zip(prompts, params):
        llm.add_request(prompt, sampling_params)

    prefill_time = decode_time = 0.0
    prefill_tokens = decode_tokens = 0
    completed_requests = completed_output_tokens = 0
    prefill_steps = decode_steps = idle_steps = idle_step_count = 0
    first_decode_t: float | None = None

    while not llm.is_finished():
        _cuda_synchronize(torch_module)
        step_start = time.perf_counter()
        outputs, num_tokens = llm.step()
        _cuda_synchronize(torch_module)
        step_time = time.perf_counter() - step_start
        if num_tokens > 0:
            idle_steps = 0
            prefill_steps += 1
            prefill_time += step_time
            prefill_tokens += num_tokens
        elif num_tokens < 0:
            idle_steps = 0
            decode_steps += 1
            if first_decode_t is None:
                # The first decode step starts after the initial prefill and
                # therefore marks the end of batch prefill/TTFT.
                first_decode_t = step_start - t0
            decode_time += step_time
            decode_tokens += -num_tokens
        else:
            idle_steps += 1
            idle_step_count += 1
            if idle_steps > max_idle_steps:
                raise RuntimeError(
                    "benchmark made no progress for "
                    f"{max_idle_steps} engine steps"
                )

        for _, token_ids in outputs or []:
            completed_requests += 1
            completed_output_tokens += len(token_ids)

    _cuda_synchronize(torch_module)
    total_time = time.perf_counter() - t0
    if completed_requests != len(prompts):
        raise RuntimeError(
            f"engine finished with {completed_requests}/{len(prompts)} completed requests"
        )
    requested_output_tokens = sum(
        int(getattr(sampling_params, "max_tokens", 0))
        for sampling_params in params
    )
    batch_ttft = first_decode_t if first_decode_t is not None else total_time
    return {
        "prefill_tokens_per_second": (
            prefill_tokens / prefill_time if prefill_time else 0.0
        ),
        "decode_tokens_per_second": (
            decode_tokens / decode_time if decode_time else 0.0
        ),
        "output_tokens_per_second": (
            completed_output_tokens / total_time if total_time else 0.0
        ),
        "batch_ttft_s": batch_ttft,
        "total_time_s": total_time,
        "output_tokens": completed_output_tokens,
        "completed_requests": completed_requests,
        "requested_output_tokens": requested_output_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "idle_steps": idle_step_count,
    }


def _measure_ttft(
    llm: Any,
    prompt: list[int],
    sampling_params: Any,
    max_idle_steps: int = 1000,
) -> MetricSample:
    """Measure until the first output token is materialized."""
    torch_module = _load_torch()
    _cuda_synchronize(torch_module)
    t0 = time.perf_counter()
    llm.add_request(prompt, sampling_params)
    idle_steps = idle_step_count = 0
    steps = 0
    while not llm.is_finished():
        _cuda_synchronize(torch_module)
        llm_outputs, num_tokens = llm.step()
        _cuda_synchronize(torch_module)
        steps += 1
        if llm_outputs:
            return {
                "ttft_ms": (time.perf_counter() - t0) * 1000.0,
                "completed_requests": len(llm_outputs),
                "output_tokens": sum(
                    len(token_ids) for _, token_ids in llm_outputs
                ),
                "idle_steps": idle_step_count,
                "steps": steps,
            }
        if num_tokens == 0:
            idle_steps += 1
            idle_step_count += 1
            if idle_steps > max_idle_steps:
                raise RuntimeError(
                    "TTFT benchmark made no progress for "
                    f"{max_idle_steps} engine steps"
                )
        else:
            idle_steps = 0
    raise RuntimeError("engine finished before producing a first token")


def _build_llm(args: argparse.Namespace) -> tuple[Any, int]:
    """Construct the engine and run an optional, separately reported warm-up."""
    from prism_infer import LLM

    _seed_runtime(args.seed)
    llm = LLM(
        args.model,
        enforce_eager=args.eager,
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.ep,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks=args.num_kvcache_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    config = getattr(getattr(llm, "model_runner", None), "config", None)
    args.effective_max_model_len = getattr(
        config, "max_model_len", args.max_model_len
    )
    args.effective_num_kvcache_blocks = getattr(
        config, "num_kvcache_blocks", args.num_kvcache_blocks
    )
    vocab_size = _tokenizer_vocab_size(llm)
    if args.warmup_seqs:
        warmup_input_len = min(args.input_len, 3)
        prompts, params = _make_requests(
            args.warmup_seqs,
            warmup_input_len,
            args.warmup_output_len,
            args.seed - 1,
            args.temperature,
            vocab_size,
        )
        llm.generate(prompts, params, use_tqdm=False)
        # Warm-up can consume torch's sampling RNG; measured repetitions start
        # from the requested seed rather than from an implementation detail.
        _seed_runtime(args.seed)
    return llm, vocab_size


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"python": platform.python_version()}
    try:
        metadata["prism_infer"] = importlib.metadata.version("prism-infer")
    except importlib.metadata.PackageNotFoundError:
        metadata["prism_infer"] = None
    repo_root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["git_revision"] = revision
        metadata["git_dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        metadata["git_revision"] = os.environ.get("PRISM_BENCH_GIT_REVISION")
        metadata["git_dirty"] = None
    torch_module = _load_torch()
    if torch_module is not None:
        metadata["torch"] = getattr(torch_module, "__version__", None)
        metadata["cuda_runtime"] = getattr(
            getattr(torch_module, "version", None), "cuda", None
        )
        if torch_module.cuda.is_available():
            metadata["gpu_count"] = torch_module.cuda.device_count()
            metadata["gpus"] = [
                torch_module.cuda.get_device_name(index)
                for index in range(metadata["gpu_count"])
            ]
    metadata["environment"] = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "NCCL_P2P_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "PRISM_BENCH_GIT_REVISION",
        )
        if name in os.environ
    }
    return metadata


def _base_payload(
    args: argparse.Namespace,
    benchmark: str,
    vocab_size: int,
    *,
    num_seqs: int | None = None,
    output_len: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "command": args.cmd,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "execution": {
            "mode": "eager" if args.eager else "cuda_graph",
            "enforce_eager": args.eager,
            "tp": args.tp,
            "ep": args.ep,
            "max_model_len": args.max_model_len,
            "effective_max_model_len": getattr(
                args, "effective_max_model_len", args.max_model_len
            ),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "effective_num_kvcache_blocks": getattr(
                args, "effective_num_kvcache_blocks", args.num_kvcache_blocks
            ),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_idle_steps": args.max_idle_steps,
        },
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len if output_len is None else output_len,
            "num_seqs": args.num_seqs if num_seqs is None else num_seqs,
            "temperature": args.temperature,
            "ignore_eos": True,
        },
        "repetitions": args.repetitions,
        "warmup": {
            "requests": args.warmup_seqs,
            "input_len": min(args.input_len, 3),
            "output_len": args.warmup_output_len,
        },
        "prompt_generation": {
            "scheme": "deterministic_random_token_ids",
            "base_seed": args.seed,
            "vocab_size": vocab_size,
            "case_seeds_are_disjoint": True,
        },
        "timing": {
            "clock": "time.perf_counter",
            "cuda_synchronized": True,
            "batch_ttft_boundary": "before_first_decode_step",
        },
        "runtime": _runtime_metadata(),
    }


def _median_metrics(
    samples: list[MetricSample],
    keys: tuple[str, ...],
) -> dict[str, float]:
    return {
        key: float(statistics.median(float(sample[key]) for sample in samples))
        for key in keys
        if samples and key in samples[0]
    }


def _write_output(payload: dict[str, Any], output: str | None) -> None:
    if not output:
        return
    path = Path(os.path.expanduser(output))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"JSON results written to {path}", file=sys.stderr)


def cmd_throughput(args: argparse.Namespace) -> dict[str, Any]:
    llm, vocab_size = _build_llm(args)
    samples: list[MetricSample] = []
    for repetition in range(args.repetitions):
        run_seed = args.seed + repetition
        _seed_runtime(run_seed)
        prompts, params = _make_requests(
            args.num_seqs,
            args.input_len,
            args.output_len,
            run_seed,
            args.temperature,
            vocab_size,
        )
        sample = _run(llm, prompts, params, args.max_idle_steps)
        sample["seed"] = run_seed
        sample["prompt_digest"] = _prompt_digest(prompts)
        samples.append(sample)
    payload = _base_payload(args, "throughput", vocab_size)
    payload["samples"] = samples
    payload["summary"] = _median_metrics(
        samples,
        (
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "output_tokens_per_second",
            "batch_ttft_s",
            "total_time_s",
            "output_tokens",
        ),
    )
    return payload


def cmd_ttft(args: argparse.Namespace) -> dict[str, Any]:
    llm, vocab_size = _build_llm(args)
    samples: list[MetricSample] = []
    for repetition in range(args.repetitions):
        run_seed = args.seed + repetition
        _seed_runtime(run_seed)
        prompts, params = _make_requests(
            1,
            args.input_len,
            1,
            run_seed,
            args.temperature,
            vocab_size,
        )
        sample = _measure_ttft(llm, prompts[0], params[0], args.max_idle_steps)
        sample["seed"] = run_seed
        sample["prompt_digest"] = _prompt_digest(prompts)
        samples.append(sample)
    payload = _base_payload(
        args,
        "ttft",
        vocab_size,
        num_seqs=1,
        output_len=1,
    )
    payload["samples"] = samples
    payload["summary"] = _median_metrics(samples, ("ttft_ms", "output_tokens"))
    return payload


def cmd_sweep(args: argparse.Namespace) -> dict[str, Any]:
    llm, vocab_size = _build_llm(args)
    results: list[dict[str, Any]] = []
    case_count = len(args.batch_sizes)
    for batch_index, batch_size in enumerate(args.batch_sizes):
        samples: list[MetricSample] = []
        for repetition in range(args.repetitions):
            run_seed = args.seed + repetition * case_count + batch_index
            _seed_runtime(run_seed)
            prompts, params = _make_requests(
                batch_size,
                args.input_len,
                args.output_len,
                run_seed,
                args.temperature,
                vocab_size,
            )
            sample = _run(llm, prompts, params, args.max_idle_steps)
            sample["seed"] = run_seed
            sample["prompt_digest"] = _prompt_digest(prompts)
            samples.append(sample)
        results.append(
            {
                "batch_size": batch_size,
                "samples": samples,
                "summary": _median_metrics(
                    samples,
                    (
                        "decode_tokens_per_second",
                        "output_tokens_per_second",
                        "prefill_tokens_per_second",
                        "batch_ttft_s",
                    ),
                ),
            }
        )
    payload = _base_payload(args, "sweep", vocab_size)
    payload["workload"]["num_seqs"] = None
    payload["batch_sizes"] = args.batch_sizes
    payload["results"] = results
    return payload


def _print_text(payload: dict[str, Any]) -> None:
    execution = payload["execution"]
    mode = execution["mode"]
    print(
        f"\n=== {payload['benchmark']} ({mode}, "
        f"TP={execution['tp']}, EP={execution['ep']}) ==="
    )
    workload = payload["workload"]
    if payload["benchmark"] == "throughput":
        summary = payload["summary"]
        print(
            "config        : "
            f"num_seqs={workload['num_seqs']} "
            f"input_len={workload['input_len']} "
            f"output_len={workload['output_len']}"
        )
        print(f"repetitions   : {payload['repetitions']} (median summary)")
        print(
            f"prefill tput  : {summary['prefill_tokens_per_second']:.1f} tok/s"
        )
        print(
            f"decode TPS    : {summary['decode_tokens_per_second']:.1f} tok/s"
        )
        print(
            f"e2e tput      : {summary['output_tokens_per_second']:.1f} tok/s"
        )
        print(f"batch TTFT    : {summary['batch_ttft_s'] * 1000:.1f} ms")
    elif payload["benchmark"] == "ttft":
        print(f"config        : input_len={workload['input_len']}")
        print(f"repetitions   : {payload['repetitions']} (median summary)")
        print(f"TTFT          : {payload['summary']['ttft_ms']:.1f} ms")
    else:
        print(
            f"config        : input_len={workload['input_len']} "
            f"output_len={workload['output_len']}"
        )
        print(f"repetitions   : {payload['repetitions']} (median summary)")
        print(f"{'batch':>6} | {'decode TPS':>11} | {'e2e tput':>10}")
        print("-" * 34)
        for result in payload["results"]:
            summary = result["summary"]
            print(
                f"{result['batch_size']:>6} | "
                f"{summary['decode_tokens_per_second']:>9.1f}   | "
                f"{summary['output_tokens_per_second']:>8.1f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = _validate_args(parser.parse_args(argv), parser)
    handler = {
        "throughput": cmd_throughput,
        "ttft": cmd_ttft,
        "sweep": cmd_sweep,
    }[args.cmd]
    payload = handler(args)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(payload)
    _write_output(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
