# Shared pytest fixtures.
#
# Notes:
#   1. Initialize a single-process distributed group (gloo). layers/linear.py's
#      LinearBase calls dist.get_rank() / dist.get_world_size() at construction
#      time (for tensor-parallel sharding). Even with TP=1 the process group must
#      be initialized first, otherwise constructing any linear layer raises
#      "Default process group has not been initialized". gloo lets it run on CPU
#      without NCCL/GPU.
#   2. torch.compile fallback: several ops (RMSNorm/RoPE/SiluAndMul) use
#      @torch.compile and may fail to compile on pure CPU or some environments;
#      suppress_errors=True makes them fall back to eager so test logic is not
#      blocked by compilation issues.
import os
import pytest
import torch
import torch.distributed as dist

# Fall back to eager when torch.compile fails (skip silently if dynamo is absent).
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


@pytest.fixture(scope="session", autouse=True)
def _init_distributed():
    # Initialize a single-process group once per session (rank=0, world_size=1).
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    # Do not destroy explicitly: process exit reclaims it; destroying here can
    # conflict with the ordering of other fixtures.


@pytest.fixture(autouse=True)
def _seed():
    # Fix the random seed before each test for reproducibility.
    torch.manual_seed(0)
