import argparse
import json
import os
import subprocess
import sys


DEFAULT_PROMPT = "The capital of France is"


def _resolve_model(cli_model: str | None) -> str:
    path = cli_model or os.environ.get("PRISM_TEST_MODEL") or os.environ.get("PRISM_MODEL")
    if not path:
        sys.exit("set --model or PRISM_TEST_MODEL / PRISM_MODEL to a local model dir")
    return os.path.expanduser(path)


# ----------------------------------------------------------------------------
# Worker: build one LLM at a given TP degree, generate, print result as JSON.
# Runs as its own process so the process group is created and torn down cleanly.
# ----------------------------------------------------------------------------
def run_worker(args) -> None:
    import torch
    from prism_infer import LLM, SamplingParams

    model = _resolve_model(args.model)

    # Deterministic per-process RNG so the only variable is the TP degree.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Fixed KV capacity: do NOT rely on per-rank mem_get_info auto-estimation,
    # which is unstable under concurrent TP init (see allocate_kv_cache).
    llm = LLM(
        model,
        enforce_eager=True,                 # eager: rule out CUDA-graph as a variable
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        num_kvcache_blocks=args.num_kvcache_blocks,
        gpu_memory_utilization=0.4,
    )

    # A very low temperature makes Gumbel-max near-greedy while retaining the
    # same sampling path in both workers. This is a smoke test, not a proof of
    # parity for every prompt or sampling configuration.
    sp = SamplingParams(temperature=1e-6, ignore_eos=True, max_tokens=args.max_tokens)
    out = llm.generate([args.prompt], sp, use_tqdm=False)[0]
    print("PARITY_JSON " + json.dumps({"token_ids": out["token_ids"], "text": out["text"]}))


# ----------------------------------------------------------------------------
# Driver: launch one worker per TP degree, parse JSON, compare token streams.
# ----------------------------------------------------------------------------
def _launch(tp: int, args) -> dict:
    # Defensive: clear any stale fixed-name shm from an older crashed run.
    try:
        os.remove("/dev/shm/prism_infer")
    except OSError:
        pass

    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--tp", str(tp),
        "--model", _resolve_model(args.model),
        "--prompt", args.prompt,
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--num-kvcache-blocks", str(args.num_kvcache_blocks),
    ]
    print(f"\n--- launching TP={tp} worker ---")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.exit(f"TP={tp} worker failed (exit {proc.returncode})")

    line = next((l for l in proc.stdout.splitlines() if l.startswith("PARITY_JSON ")), None)
    if line is None:
        sys.stdout.write(proc.stdout)
        sys.exit(f"TP={tp} worker produced no PARITY_JSON line")
    return json.loads(line[len("PARITY_JSON "):])


def _first_divergence(a: list[int], b: list[int]) -> int:
    # Index of the first differing token, or -1 if one is a prefix of the other
    # and they match up to the shorter length.
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def run_driver(args) -> None:
    tps = [int(t) for t in args.tps.split(",")]
    base = tps[0]
    results = {tp: _launch(tp, args) for tp in tps}

    base_ids = results[base]["token_ids"]
    print("\n================ TP parity report ================")
    print(f"model      : {_resolve_model(args.model)}")
    print(f"prompt     : {args.prompt!r}")
    print(f"max_tokens : {args.max_tokens}  (temperature=1e-6, eager)")
    print(f"baseline   : TP={base}")
    print(f"\nTP={base} text: {results[base]['text']!r}")

    all_first_ok = True
    all_full_ok = True
    for tp in tps[1:]:
        ids = results[tp]["token_ids"]
        first_ok = bool(base_ids) and bool(ids) and base_ids[0] == ids[0]
        div = _first_divergence(base_ids, ids)
        full_ok = (div == -1) and (len(base_ids) == len(ids))
        all_first_ok &= first_ok
        all_full_ok &= full_ok

        print(f"\nTP={tp} text: {results[tp]['text']!r}")
        print(f"TP={tp} vs TP={base}:")
        print(f"  first-token match : {'YES' if first_ok else 'NO'}")
        if full_ok:
            print(f"  full-sequence match: YES ({len(ids)} tokens)")
        else:
            print(f"  full-sequence match: NO  (first divergence at index {div})")

    print("\n---------------- verdict ----------------")
    if all_full_ok:
        print("PASS: token streams matched for this prompt and configuration.")
        print("      This does not prove parity for all prompts or workloads.")
    elif all_first_ok:
        print("INCONCLUSIVE: first token matched for this prompt and TP set.")
        print("              Later divergence may come from sampling noise or BF16")
        print("              multi-step error amplification; inspect logits before")
        print("              drawing a general correctness conclusion.")
    else:
        print("MISMATCH: first token differs for this prompt and TP set.")
        print("          Investigate all_reduce (o_proj/down_proj), head split,")
        print("          vocab gather, and single-step logits before assigning a cause.")
    print("=========================================")


def main() -> None:
    p = argparse.ArgumentParser(description="prism-infer TP=1 vs TP=2 parity diagnostic")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)  # internal
    p.add_argument("--tp", type=int, default=1, help="worker mode: TP degree to run")
    p.add_argument("--tps", default="1,2", help="driver mode: comma-separated TP degrees")
    p.add_argument("--model", default=None, help="local model dir (else PRISM_TEST_MODEL/PRISM_MODEL)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--num-kvcache-blocks", type=int, default=64,
                   help="fixed KV capacity; avoids unstable per-rank auto-estimation")
    args = p.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
