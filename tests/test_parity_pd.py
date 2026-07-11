import atexit
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch


MODEL_PATH = os.environ.get("PRISM_TEST_MODEL")
PROMPT = "The capital of France is a question about geography. " * 40
N_TOKENS = 20


# torchrun re-enters this module with --_worker; collection-time skips must not
# terminate workers before they reach the worker entry point.
if "--_worker" not in sys.argv:
    if not torch.cuda.is_available():
        pytest.skip("PD parity requires CUDA", allow_module_level=True)
    if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
        pytest.skip(
            "set PRISM_TEST_MODEL to a local Dense model directory",
            allow_module_level=True,
        )


def _worker_main() -> None:
    import torch.distributed as dist

    from prism_infer.config import Config
    from prism_infer.engine.kv_transfer import NCCLTransport
    from prism_infer.engine.llm_engine import LLMEngine
    from prism_infer.engine.pd_runner import _activate_received_sequence
    from prism_infer.engine.sequence import Sequence
    from prism_infer.sampling_params import SamplingParams

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    pd_port = int(os.environ["MASTER_PORT"]) + 1
    model = os.environ["PRISM_MODEL"]
    prompt = os.environ["_PD_PROMPT"]
    out_path = os.environ["_PD_OUT"]

    torch.cuda.set_device(rank % torch.cuda.device_count())

    mode = "prefill-only" if rank == 0 else "decode-only"
    config = Config(
        model=model,
        engine_mode=mode,
        kv_transfer_backend="nccl",
        pd_decode_addr="rank:1",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        num_kvcache_blocks=64,
        _use_shm_worker_loop=False,
    )
    engine = LLMEngine(config)

    dist.destroy_process_group()
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{pd_port}",
        world_size=world_size,
        rank=rank,
    )
    pd_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")

    if rank == 0:
        engine.kv_connector.pusher.transport.pd_group = pd_group
        decode_transport = None
    else:
        decode_transport = NCCLTransport(
            pd_group,
            decode_rank=0,
            kv_cache=engine.model_runner.kv_cache,
        )

    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=True,
        max_tokens=N_TOKENS,
    )

    if rank == 0:
        prompt_ids = engine.tokenizer.encode(prompt)
        assert 395 <= len(prompt_ids) <= 405, (
            f"expected a prompt near 401 tokens, got {len(prompt_ids)}"
        )
        num_blocks = (len(prompt_ids) + Sequence.block_size - 1) // Sequence.block_size
        assert num_blocks >= 2, f"expected at least 2 KV blocks, got {num_blocks}"

        # Publish the receive count before prefill so rank 1 posts every irecv
        # before the connector issues its per-block NCCL isends.
        dist.send(
            torch.tensor([num_blocks], dtype=torch.int64, device="cuda"),
            dst=1,
            group=pd_group,
        )

        captured: dict[str, int] = {}
        original_postprocess = engine.scheduler.postprocess

        def capture_first_token(seqs, token_ids, is_prefill):
            if is_prefill and token_ids and "first" not in captured:
                captured["first"] = token_ids[0]
            return original_postprocess(seqs, token_ids, is_prefill)

        engine.scheduler.postprocess = capture_first_token
        engine.add_request(prompt, sampling_params)
        while not engine.is_finished():
            engine.step()

        transport = engine.kv_connector.pusher.transport
        for requests, _chunks, _slices, _callback in list(transport._pending):
            for request in requests:
                request.wait()
        transport.poll_completions()

        dist.send(
            torch.tensor([captured["first"]], dtype=torch.int64, device="cuda"),
            dst=1,
            group=pd_group,
        )
    else:
        num_blocks_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(num_blocks_tensor, src=0, group=pd_group)
        num_blocks = int(num_blocks_tensor.item())
        assert num_blocks >= 2, f"expected at least 2 KV blocks, got {num_blocks}"

        block_manager = engine.scheduler.block_manager
        dst_blocks = [block_manager._allocate_block() for _ in range(num_blocks)]
        assert decode_transport is not None
        decode_transport.recv_kv(src_rank=0, block_ids=dst_blocks)

        first_token_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
        dist.recv(first_token_tensor, src=0, group=pd_group)

        engine.add_request(prompt, sampling_params)
        seq = engine.scheduler.waiting[-1]
        _activate_received_sequence(
            engine,
            seq,
            dst_blocks,
            int(first_token_tensor.item()),
        )
        engine.scheduler._kv_ready_fn = None

        token_ids: list[int] = []
        while not engine.is_finished():
            outputs, _ = engine.step()
            for _, completed_ids in outputs:
                token_ids = list(completed_ids)

        np.save(out_path, np.array(token_ids, dtype=np.int64))

    dist.barrier(group=pd_group)
    dist.destroy_process_group()


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def _spawn_pd_pair(prompt: str, out_path: str) -> None:
    env = os.environ.copy()
    env["PRISM_MODEL"] = MODEL_PATH
    env["_PD_PROMPT"] = prompt
    env["_PD_OUT"] = out_path

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_port",
        _free_port(),
        __file__,
        "--_worker",
    ]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"PD worker pair failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_unified(prompt: str) -> list[int]:
    import torch.distributed as dist

    from prism_infer import LLM, SamplingParams

    if dist.is_initialized():
        dist.destroy_process_group()

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        num_kvcache_blocks=128,
    )
    output = llm.generate(
        [prompt],
        SamplingParams(temperature=0, ignore_eos=True, max_tokens=N_TOKENS),
        use_tqdm=False,
    )
    token_ids = list(output[0]["token_ids"])
    atexit.unregister(llm.exit)
    llm.exit()
    torch.cuda.empty_cache()

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return token_ids


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="PD parity requires at least 2 visible CUDA GPUs",
)
def test_pd_greedy_tokens_match_unified() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        pd_output_path = os.path.join(temp_dir, "pd_tokens.npy")
        unified_ids = _run_unified(PROMPT)
        _spawn_pd_pair(PROMPT, pd_output_path)
        pd_ids = np.load(pd_output_path).tolist()

    assert len(unified_ids) == N_TOKENS
    assert len(pd_ids) == N_TOKENS
    assert pd_ids == unified_ids, (
        "PD and unified greedy token sequences differ:\n"
        f"unified: {unified_ids}\n"
        f"PD:      {pd_ids}"
    )


if __name__ == "__main__" and "--_worker" in sys.argv:
    _worker_main()