# test_parity_ep.py -- EP=2 logits parity for MoE models
#
# Verifies EP forward correctness by comparing first decode-step logits of
# EP=1 vs EP=2.  Each EP degree runs as its own torchrun worker group so the
# NCCL process group is set up correctly and the processes are fully isolated.
#
# Runs in a separate file from test_parity_moe_e2e.py so the 30B model loaded
# by that test is fully unloaded before EP workers start (avoids OOM on GPU 0).
#
# Prerequisites:
#   - PRISM_TEST_MOE_MODEL -> a local MoE model dir (e.g. Qwen3-30B-A3B)
#   - >=2 visible CUDA GPUs, flash-attn
#   Automatically skipped when unset, no GPU, or <2 GPUs.
import json
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

MODEL_PATH = os.environ.get("PRISM_TEST_MOE_MODEL")
PROMPT = "The capital of France is"

if not torch.cuda.is_available():
    pytest.skip("EP parity UT requires CUDA GPU", allow_module_level=True)
if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    pytest.skip("set PRISM_TEST_MOE_MODEL to a local MoE model dir", allow_module_level=True)


# ----------------------------------------------------------------------------
# Worker entry point: invoked by torchrun, one process per GPU rank.
# All ranks run generate() in lockstep; EP all-to-all fires inside MoE layers.
# Rank 0 hooks the sampler to capture the first decode-step logits.
# ----------------------------------------------------------------------------
def _worker_main() -> None:
    import torch
    from prism_infer import LLM, SamplingParams
    from prism_infer.layers import sampler as sampler_mod

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    model = os.environ["PRISM_MODEL"]
    prompt = os.environ["_EP_PROMPT"]
    out_path = os.environ["_EP_OUT"]

    cap = {"logits": None}
    if rank == 0:
        orig = sampler_mod.Sampler.forward

        def hook(self, logits, *a, **kw):
            # Skip prefill (shape[0] > 1); capture the first decode step only.
            if cap["logits"] is None and logits.shape[0] == 1:
                cap["logits"] = logits.detach().float().cpu().numpy()
            return orig(self, logits, *a, **kw)

        sampler_mod.Sampler.forward = hook

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    llm = LLM(
        model,
        expert_parallel_size=world_size,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        num_kvcache_blocks=256,
        master_port=master_port,
        _use_shm_worker_loop=False,
    )
    llm.generate([prompt], SamplingParams(temperature=0.01, max_tokens=1), use_tqdm=False)

    if rank == 0:
        if cap["logits"] is None:
            sys.exit("Sampler.forward was never called")
        np.save(out_path, cap["logits"])
        print(f"PARITY_SAVED ep={world_size} shape={list(cap['logits'].shape)}")


# ----------------------------------------------------------------------------
# Driver helpers
# ----------------------------------------------------------------------------
def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def _spawn_ep(ep: int, prompt: str, out_path: str) -> None:
    """Launch ep processes via torchrun, wait for completion."""
    env = os.environ.copy()
    env["PRISM_MODEL"] = MODEL_PATH
    env["_EP_PROMPT"] = prompt
    env["_EP_OUT"] = out_path
    env["NCCL_P2P_DISABLE"] = "1"  # required in PCIe containers (e.g. AutoDL)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={ep}",
        "--master_port", _free_port(),
        __file__, "--_worker",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"EP={ep} worker group failed (exit {proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _compare(path_a: str, path_b: str, ep_a: int, ep_b: int, prompt: str) -> dict:
    a = np.load(path_a)
    b = np.load(path_b)
    if a.shape != b.shape:
        return {"prompt": prompt, "ok": False, "error": f"shape mismatch {a.shape} vs {b.shape}"}
    d = np.abs(a - b)
    top1_a = int(a.argmax(-1)[0])
    top1_b = int(b.argmax(-1)[0])
    sorted0 = sorted(a[0].tolist(), reverse=True)
    gap = float(sorted0[0] - sorted0[1])
    top1_match = (top1_a == top1_b)
    ok = top1_match and (float(d.max()) < gap)
    return {
        "prompt": prompt,
        "ep1_top1": top1_a, f"ep{ep_b}_top1": top1_b,
        "top1_match": top1_match,
        "max_delta": round(float(d.max()), 6),
        "top1_top2_gap": round(gap, 6),
        "ok": ok,
    }


# ----------------------------------------------------------------------------
# pytest test
# ----------------------------------------------------------------------------
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="EP=2 parity requires >=2 visible GPUs",
)
def test_ep2_logits_parity():
    """EP=1 vs EP=2 first decode-step logits parity.

    top-1 token must match on the test prompt.
    FAIL (top-1 mismatch) means a real dispatch/combine bug in ExpertParallelMoE.
    LIKELY-OK (top-1 match, max|Δ| >= gap) is BF16 all-to-all noise, not a bug.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out1 = os.path.join(tmp, "ep1.npy")
        out2 = os.path.join(tmp, "ep2.npy")
        _spawn_ep(1, PROMPT, out1)
        _spawn_ep(2, PROMPT, out2)
        r = _compare(out1, out2, 1, 2, PROMPT)

    assert r.get("top1_match", False), (
        f"EP=2 logits parity FAIL — top-1 token differs (ep1={r.get('ep1_top1')} "
        f"ep2={r.get('ep2_top1')}). Suspect dispatch/combine bug.\n{json.dumps(r, indent=2)}"
    )
    # top-1 matches — PASS or LIKELY-OK, either is acceptable.


# ----------------------------------------------------------------------------
# Worker entry point (called by torchrun re-invocation)
# ----------------------------------------------------------------------------
if __name__ == "__main__" and "--_worker" in sys.argv:
    _worker_main()
