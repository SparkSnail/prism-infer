# Shared pytest fixtures.
#
# Notes:
#   1. Initialize a single-process distributed group (gloo). layers/linear.py's
#      _tp_rank()/_tp_size() default to (0, 1) when _TP_GROUP is None, so no
#      process group is strictly required for TP=1. However dist.init_process_group
#      is still needed for tests that call dist APIs directly (e.g. EP all-to-all
#      with ep_size=1 falls back to x.clone() but the group must exist). gloo lets
#      it run on CPU without NCCL/GPU.
#   2. torch.compile fallback: several ops (RMSNorm/RoPE/SiluAndMul) use
#      @torch.compile and may fail to compile on pure CPU or some environments;
#      suppress_errors=True makes them fall back to eager so test logic is not
#      blocked by compilation issues.
import os
import socket
import pytest
import torch
import torch.distributed as dist

# Fall back to eager when torch.compile fails (skip silently if dynamo is absent).
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


def _free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session", autouse=True)
def _init_distributed():
    # Initialize a single-process group once per session (rank=0, world_size=1).
    # Use a dynamically assigned free port to avoid EADDRINUSE on rapid re-runs.
    if dist.is_available() and not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    # Do not destroy explicitly: process exit reclaims it; destroying here can
    # conflict with the ordering of other fixtures.


@pytest.fixture(autouse=True)
def _seed():
    # Fix the random seed before each test for reproducibility.
    torch.manual_seed(0)
