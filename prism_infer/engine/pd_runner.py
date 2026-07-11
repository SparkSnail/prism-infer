"""PD disaggregation launcher.

Each process starts LLMEngine in prefill-only or decode-only mode.
KV cache is transferred between processes via NCCL P2P (cross-host) or
CUDA IPC (same-host).

Usage (single-node, two GPUs):
  CUDA_VISIBLE_DEVICES=0 python -m prism_infer.engine.pd_runner \\
      --mode prefill-only --model /path/Qwen3-0.6B \\
      --master-addr localhost --master-port 29500 --world-size 2 --rank 0

  CUDA_VISIBLE_DEVICES=1 python -m prism_infer.engine.pd_runner \\
      --mode decode-only --model /path/Qwen3-0.6B \\
      --master-addr localhost --master-port 29500 --world-size 2 --rank 1

Usage (multi-node):
  NCCL_SOCKET_IFNAME=eth0 python -m prism_infer.engine.pd_runner \\
      --mode prefill-only --model /shared/Qwen3-7B \\
      --master-addr 192.168.1.10 --master-port 29500 --world-size 2 --rank 0

  NCCL_SOCKET_IFNAME=eth0 python -m prism_infer.engine.pd_runner \\
      --mode decode-only --model /shared/Qwen3-7B \\
      --master-addr 192.168.1.10 --master-port 29500 --world-size 2 --rank 1
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.distributed as dist

from prism_infer.config import Config
from prism_infer.engine.llm_engine import LLMEngine
from prism_infer.engine.sequence import SequenceStatus
from prism_infer.sampling_params import SamplingParams


def init_pd_process_group(
    master_addr: str,
    master_port: int,
    world_size: int,
    rank: int,
) -> dist.ProcessGroup:
    """Initialize the NCCL process group for the P/D pair.

    Both processes must call this simultaneously (NCCL rendezvous).
    The group is reused for the full process lifetime.
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
        )
        print(f"[rank={rank}] NCCL init done "
              f"(world_size={world_size}, master={master_addr}:{master_port})")
    return dist.group.WORLD


def run_prefill(args):
    """Prefill process: accept requests, run prefill, push KV to decode."""
    rank = args.rank
    pd_group = init_pd_process_group(
        args.master_addr, args.master_port, args.world_size, rank
    )
    config = Config(
        model=args.model,
        engine_mode="prefill-only",
        kv_transfer_backend=args.kv_transfer_backend,
        pd_decode_addr=f"rank:{(rank + 1) % args.world_size}",
        pd_master_port=args.master_port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        _use_shm_worker_loop=False,
    )
    config._pd_group = pd_group
    config._pd_rank = 1   # decode is rank 1 within the pd group

    print(f"[rank={rank}] starting prefill engine, model={args.model}")
    engine = LLMEngine(config)

    prompts = args.prompts or ["What is the weather like today?", "Give a brief introduction to quantum computing."]
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Capture the first generated token from each prefill so decode can start from it.
    first_tokens: dict[int, int] = {}
    original_postprocess = engine.scheduler.postprocess

    def capture_first_tokens(seqs, token_ids, is_prefill):
        if is_prefill:
            for seq, token_id in zip(seqs, token_ids):
                first_tokens[seq.seq_id] = token_id
        return original_postprocess(seqs, token_ids, is_prefill)

    engine.scheduler.postprocess = capture_first_tokens

    for p in prompts:
        engine.add_request(p, sp)

    # Send block counts so the decode side can pre-allocate before KV arrives.
    # D side must issue irecv before P side issues isend (NCCL P2P ordering).
    block_counts = [
        (len(engine.tokenizer.encode(prompt)) + config.kvcache_block_size - 1)
        // config.kvcache_block_size
        for prompt in prompts
    ]
    dist.send(torch.tensor(block_counts, dtype=torch.int64, device="cuda"), dst=1)

    print(f"[rank={rank}] prefill start, {len(prompts)} requests")
    t0 = time.perf_counter()
    step = 0
    while not engine.is_finished():
        outputs, num_tokens = engine.step()
        step += 1
        if step % 10 == 0:
            print(f"[rank={rank}] step={step}, num_tokens={num_tokens}")

    # Wait for any in-flight NCCL sends to complete before forwarding first tokens.
    pusher = getattr(engine.kv_connector, "pusher", None)
    assert pusher is not None, "prefill-only connector must provide a KV pusher"
    transport = pusher.transport
    for reqs, _chunks, _slices, _callback in list(
        getattr(transport, "_pending", [])
    ):
        for request in reqs:
            request.wait()
    if hasattr(transport, "poll_completions"):
        transport.poll_completions()

    # Send the first token of each sequence (ordered by seq_id) so decode can
    # start its first decode step without an extra prefill round.
    ordered_first_tokens = [first_tokens[seq_id] for seq_id in sorted(first_tokens)]
    dist.send(
        torch.tensor(ordered_first_tokens, dtype=torch.int64, device="cuda"),
        dst=1,
    )

    elapsed = time.perf_counter() - t0
    print(f"[rank={rank}] prefill done in {elapsed:.2f}s ({step} steps)")
    print(f"[rank={rank}] waiting for decode to finish...")
    dist.barrier()
    print(f"[rank={rank}] prefill process exit")


def run_decode(args):
    """Decode process: wait for KV transfer, run decode, emit tokens."""
    rank = args.rank
    pd_group = init_pd_process_group(
        args.master_addr, args.master_port, args.world_size, rank
    )
    config = Config(
        model=args.model,
        engine_mode="decode-only",
        kv_transfer_backend=args.kv_transfer_backend,
        pd_master_port=args.master_port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        _use_shm_worker_loop=False,
    )
    config._pd_group = pd_group
    config._pd_rank = 0   # prefill is rank 0 within the pd group

    print(f"[rank={rank}] starting decode engine, model={args.model}")
    engine = LLMEngine(config)

    prompts = args.prompts or ["What is the weather like today?", "Give a brief introduction to quantum computing."]
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    # Step 1: receive per-request block counts; pre-allocate and issue irecv so the
    # D side is ready to receive before P side starts sending.
    block_counts = torch.zeros(len(prompts), dtype=torch.int64, device="cuda")
    dist.recv(block_counts, src=0)
    received: list[tuple[str, list[int]]] = []
    block_manager = engine.scheduler.block_manager
    transport = engine.kv_connector.transport
    for prompt, block_count in zip(prompts, block_counts.tolist()):
        dst_blocks = [block_manager._allocate_block() for _ in range(int(block_count))]
        transport.recv_kv(src_rank=0, block_ids=dst_blocks)
        received.append((prompt, dst_blocks))

    # Step 2: receive first token from P side (produced by prefill logits).
    # D side starts decode from this token, avoiding a redundant prefill round.
    first_tokens = torch.zeros(len(prompts), dtype=torch.int64, device="cuda")
    dist.recv(first_tokens, src=0)
    for (prompt, dst_blocks), first_token in zip(received, first_tokens.tolist()):
        engine.add_request(prompt, sp)
        seq = engine.scheduler.waiting[-1]
        _activate_received_sequence(engine, seq, dst_blocks, int(first_token))

    print(f"[rank={rank}] decode waiting for KV transfer...")
    t0 = time.perf_counter()
    outputs_all = {}
    step = 0
    while not engine.is_finished():
        outputs, num_tokens = engine.step()
        step += 1
        for seq_id, token_ids in outputs:
            outputs_all[seq_id] = token_ids
        if step % 20 == 0:
            print(f"[rank={rank}] step={step}, done={len(outputs_all)}/{len(prompts)}")

    elapsed = time.perf_counter() - t0
    print(f"\n[rank={rank}] decode done in {elapsed:.2f}s ({step} steps)")
    print("=" * 60)
    print("outputs:")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    for seq_id in sorted(outputs_all.keys()):
        token_ids = outputs_all[seq_id]
        text = tokenizer.decode(token_ids)
        prompt_idx = seq_id % len(prompts)
        print(f"  [{seq_id}] prompt: {prompts[prompt_idx]!r}")
        print(f"       output: {text!r}")
        print()

    dist.barrier()
    print(f"[rank={rank}] decode process exit")
    return outputs_all


def _activate_received_sequence(
    engine: LLMEngine,
    seq,
    dst_blocks: list[int],
    first_token: int,
) -> None:
    """Move a received-KV sequence directly into decode, skipping local prefill.

    The P side has already run prefill and transferred KV into dst_blocks.
    We set block_table, mark cached tokens, append the first token that the
    P side produced, and move the sequence from waiting to running.
    """
    seq.block_table = dst_blocks
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.append_token(first_token)
    seq.is_prefill = False
    seq.status = SequenceStatus.RUNNING
    engine.scheduler.waiting.remove(seq)
    engine.scheduler.running.append(seq)


def run_unified(args):
    """Unified mode: prefill+decode in one process, used as baseline for parity check."""
    config = Config(
        model=args.model,
        engine_mode="unified",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )
    engine = LLMEngine(config)

    prompts = args.prompts or ["What is the weather like today?", "Give a brief introduction to quantum computing."]
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    for p in prompts:
        engine.add_request(p, sp)

    outputs_all = {}
    while not engine.is_finished():
        outputs, _ = engine.step()
        for seq_id, token_ids in outputs:
            outputs_all[seq_id] = token_ids

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    print("=" * 60)
    print("[unified] outputs:")
    for seq_id in sorted(outputs_all.keys()):
        token_ids = outputs_all[seq_id]
        text = tokenizer.decode(token_ids)
        prompt_idx = seq_id % len(prompts)
        print(f"  [{seq_id}] prompt: {prompts[prompt_idx]!r}")
        print(f"       output: {text!r}")
    return outputs_all


def parse_args():
    parser = argparse.ArgumentParser(description="prism-infer PD disaggregation launcher")
    parser.add_argument("--mode", required=True,
                        choices=["prefill-only", "decode-only", "unified"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--kv-transfer-backend", default="auto",
                        choices=["auto", "nccl", "ipc"])
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--prompts", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prefill-only":
        run_prefill(args)
    elif args.mode == "decode-only":
        run_decode(args)
    elif args.mode == "unified":
        run_unified(args)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
