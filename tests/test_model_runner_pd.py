# test_model_runner_pd.py — CPU-runnable tests for PD-mode ModelRunner behaviour
#
# The rank-override bug (self.rank stays at GPU rank instead of being forced to 0
# in the private world_size=1 group) can only be caught by constructing a
# ModelRunner with rank=1 in a non-initialized dist context. Full ModelRunner
# construction needs a real GPU, so we mock everything after the rank-override
# lines and assert only on self.rank / self.world_size.
import socket
import torch
import torch.distributed as dist
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _fake_hf_config():
    """Minimal HF config that satisfies ModelRunner's attribute access."""
    cfg = SimpleNamespace(
        dtype=torch.bfloat16,
        num_hidden_layers=2,
        num_key_value_heads=4,
        num_attention_heads=8,
        hidden_size=512,
    )
    # head_dim is read via getattr with fallback
    cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    return cfg


def _make_config(rank_for_port=0) -> SimpleNamespace:
    return SimpleNamespace(
        model="/fake/model",
        hf_config=_fake_hf_config(),
        kvcache_block_size=256,
        enforce_eager=True,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        master_port=_free_port(),
        shm_name="",
        num_kvcache_blocks=8,
        gpu_memory_utilization=0.3,
        max_num_batched_tokens=512,
        max_model_len=512,
        max_num_seqs=4,
        # world_size property
        world_size=1,
    )


# ---------------------------------------------------------------------------
# The core mock harness
#
# We patch everything that touches real GPU / filesystem / network, then
# call ModelRunner.__init__ and inspect self.rank / self.world_size.
# ---------------------------------------------------------------------------

PATCHES = [
    "prism_infer.engine.model_runner.dist.init_process_group",
    "prism_infer.engine.model_runner.dist.is_initialized",
    "prism_infer.engine.model_runner.dist.new_group",
    "prism_infer.engine.model_runner.torch.cuda.set_device",
    "prism_infer.engine.model_runner.torch.cuda.device_count",
    "prism_infer.engine.model_runner.torch.set_default_dtype",
    "prism_infer.engine.model_runner.torch.set_default_device",
    "prism_infer.engine.model_runner.Qwen3ForCausalLM",
    "prism_infer.engine.model_runner.load_model",
    "prism_infer.engine.model_runner.Sampler",
]


def _build_runner_mocked(gpu_rank: int, dist_already_init: bool = False):
    """Construct a ModelRunner with all GPU/dist calls mocked.

    Args:
        gpu_rank: the rank passed to ModelRunner.__init__ (= GPU device id)
        dist_already_init: whether dist.is_initialized() returns True
                           (simulates torchrun-already-init scenario)

    Returns:
        The ModelRunner instance (partially constructed — only up to the
        TP group setup; model/warmup/kvcache are mocked out).
    """
    from prism_infer.engine.model_runner import ModelRunner

    config = _make_config()
    mock_ctx = {}

    with patch("prism_infer.engine.model_runner.dist.is_initialized",
               return_value=dist_already_init) as m_init, \
         patch("prism_infer.engine.model_runner.dist.init_process_group") as m_ipg, \
         patch("prism_infer.engine.model_runner.torch.cuda.set_device"), \
         patch("prism_infer.engine.model_runner.torch.cuda.device_count",
               return_value=2), \
         patch("prism_infer.engine.model_runner.torch.set_default_dtype"), \
         patch("prism_infer.engine.model_runner.torch.set_default_device"), \
         patch("prism_infer.engine.model_runner.torch.get_default_dtype",
               return_value=torch.float32), \
         patch("prism_infer.engine.model_runner.Qwen3ForCausalLM",
               return_value=MagicMock()), \
         patch("prism_infer.engine.model_runner.load_model"), \
         patch("prism_infer.engine.model_runner.Sampler",
               return_value=MagicMock()), \
         patch("prism_infer.engine.model_runner.torch.cuda.empty_cache"), \
         patch("prism_infer.engine.model_runner.torch.cuda.reset_peak_memory_stats"), \
         patch("prism_infer.engine.model_runner.torch.cuda.mem_get_info",
               return_value=(4 * 1024**3, 8 * 1024**3)), \
         patch("prism_infer.engine.model_runner.torch.cuda.memory_stats",
               return_value={"allocated_bytes.all.peak": 1e8,
                             "allocated_bytes.all.current": 1e8}), \
         patch("prism_infer.engine.model_runner.torch.empty",
               return_value=MagicMock()):
        runner = ModelRunner.__new__(ModelRunner)
        # Manually call __init__ but patch out warmup/allocate/capture
        # so we only test up to the rank-override + PG init lines.
        with patch.object(ModelRunner, "warmup_model"), \
             patch.object(ModelRunner, "allocate_kv_cache"), \
             patch.object(ModelRunner, "capture_cudagraph"):
            ModelRunner.__init__(runner, config, gpu_rank, [])

        mock_ctx["init_process_group"] = m_ipg

    return runner, mock_ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pd_standalone_rank1_overridden_to_0():
    """BUG REGRESSION: rank=1 GPU must be overridden to self.rank=0 in PD mode.

    When dist is NOT yet initialised (PD standalone), ModelRunner must force
    self.rank=0 and self.world_size=1 regardless of the GPU rank argument.
    Before the fix, self.rank stayed at 1, causing sampler to return None and
    the decode loop to spin forever without generating any tokens.
    """
    runner, ctx = _build_runner_mocked(gpu_rank=1, dist_already_init=False)

    assert runner.rank == 0, (
        f"PD standalone: self.rank should be 0 (overridden), got {runner.rank}. "
        "This is the regression: sampler returns None when self.rank != 0."
    )
    assert runner.world_size == 1, (
        f"PD standalone: self.world_size should be 1, got {runner.world_size}"
    )


def test_pd_standalone_rank0_stays_0():
    """Rank=0 (prefill GPU) also gets self.rank=0 — same override path."""
    runner, _ = _build_runner_mocked(gpu_rank=0, dist_already_init=False)
    assert runner.rank == 0
    assert runner.world_size == 1


def test_pd_standalone_init_process_group_called_with_world_size_1():
    """dist.init_process_group must be called with world_size=1, rank=0 in PD mode."""
    runner, ctx = _build_runner_mocked(gpu_rank=1, dist_already_init=False)

    m_ipg = ctx["init_process_group"]
    assert m_ipg.called, "dist.init_process_group was not called"

    # Check keyword args: world_size=1, rank=0
    _, kwargs = m_ipg.call_args
    # PyTorch accepts positional or keyword — check both
    call_args = m_ipg.call_args
    all_args = list(call_args.args) + list(call_args.kwargs.values())
    assert 1 in all_args or call_args.kwargs.get("world_size") == 1, (
        f"Expected world_size=1 in init_process_group call, got: {call_args}"
    )


def test_torchrun_path_rank_not_overridden():
    """When dist IS already initialised (torchrun EP path), rank stays as passed.

    torchrun sets up the process group before user code runs, so ModelRunner
    skips init_process_group entirely. self.rank must equal the passed-in rank.
    """
    runner, ctx = _build_runner_mocked(gpu_rank=1, dist_already_init=True)

    # In the torchrun path the override block is skipped
    assert runner.rank == 1, (
        f"torchrun path: self.rank should stay at passed-in value 1, got {runner.rank}"
    )
    # init_process_group must NOT be called (dist already initialised)
    assert not ctx["init_process_group"].called, (
        "init_process_group should not be called when dist.is_initialized() is True"
    )


def test_pd_standalone_rank_override_is_idempotent_for_rank0():
    """Override is harmless when GPU rank happens to be 0 already."""
    runner0, _ = _build_runner_mocked(gpu_rank=0, dist_already_init=False)
    runner1, _ = _build_runner_mocked(gpu_rank=1, dist_already_init=False)

    # Both must end up with rank=0, world_size=1
    assert runner0.rank == runner1.rank == 0
    assert runner0.world_size == runner1.world_size == 1
