# test_parity_moe_e2e.py -- end-to-end greedy token parity for MoE models
#
# Coverage: same scheduler + paged BlockManager + KV cache path as test_parity_e2e.py,
# but exercises the MoE forward (router + expert dispatch + combine) and confirms the
# full generation loop produces the same tokens as HuggingFace greedy on Qwen3-MoE.
#
# Both prism (temperature=0 → argmax) and HF (do_sample=False → argmax) take the exact
# same greedy path.  Flash-attn vs SDPA BF16 differences can cause near-tie divergence
# after a few tokens, so MIN_COMMON_PREFIX is intentionally small.
#
# Prerequisites:
#   - PRISM_TEST_MOE_MODEL -> a local MoE model dir (e.g. Qwen3-30B-A3B)
#       export PRISM_TEST_MOE_MODEL=~/autodl-tmp/Qwen3-30B-A3B
#   - A CUDA GPU with enough VRAM (>=40 GB for Qwen3-30B-A3B BF16) + flash-attn
#   Automatically skipped when unset, no GPU, or model is not MoE.
import os
import atexit

import pytest
import torch
import torch.distributed as dist

MODEL_PATH = os.environ.get("PRISM_TEST_MOE_MODEL")
PROMPT = "The capital of France is"
N_TOKENS = 32
MIN_COMMON_PREFIX = 5  # flash-attn vs SDPA BF16 near-tie divergence is expected

pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("MoE e2e parity UT requires CUDA GPU", allow_module_level=True)
if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    pytest.skip("set PRISM_TEST_MOE_MODEL to a local MoE model dir", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from prism_infer import LLM, SamplingParams  # noqa: E402

# Verify the model is actually MoE; skip silently if it's a dense model.
_hf_cfg = AutoConfig.from_pretrained(MODEL_PATH)
if not hasattr(_hf_cfg, "num_experts"):
    pytest.skip(
        f"PRISM_TEST_MOE_MODEL ({MODEL_PATH}) does not look like a MoE model "
        "(no num_experts in config); set it to a Qwen3-MoE checkpoint",
        allow_module_level=True,
    )


@pytest.fixture(scope="module", autouse=True)
def _release_distributed():
    # conftest's session fixture holds a single-process gloo group; the engine's
    # ModelRunner unconditionally inits its own (nccl) group, which would fail with
    # "init the default process group twice". Release the gloo group for this module,
    # then restore one for any modules that run afterwards.
    if dist.is_initialized():
        dist.destroy_process_group()
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


@pytest.fixture(scope="module")
def hf_greedy_ids():
    # HuggingFace greedy reference: force exactly N_TOKENS new tokens (ignore EOS).
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=cfg.dtype, trust_remote_code=True
        )
        .to("cuda")
        .eval()
    )
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=N_TOKENS,
            min_new_tokens=N_TOKENS,
            do_sample=False,
            num_beams=1,
        )
    gen = out[0, input_ids.shape[1]:].tolist()
    del model
    torch.cuda.empty_cache()
    return gen


@pytest.fixture(scope="module")
def prism_greedy_ids(hf_greedy_ids):
    # Depend on hf_greedy_ids so the HF model is built and freed first.
    prev_device = torch.tensor(0.0).device
    # MoE model requires enforce_eager (no CUDA graph support yet).
    # Use high gpu_memory_utilization since the 30B model needs most of the VRAM;
    # num_kvcache_blocks is set explicitly to avoid auto-estimation instability.
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        num_kvcache_blocks=256,
    )
    sp = SamplingParams(temperature=0, ignore_eos=True, max_tokens=N_TOKENS)
    out = llm.generate([PROMPT], sp, use_tqdm=False)
    ids = out[0]["token_ids"]
    atexit.unregister(llm.exit)
    llm.exit()
    torch.set_default_device(prev_device)
    torch.cuda.empty_cache()
    return ids


def _common_prefix_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_generates_requested_length(prism_greedy_ids):
    assert len(prism_greedy_ids) == N_TOKENS


def test_first_token_matches_hf(hf_greedy_ids, prism_greedy_ids):
    assert prism_greedy_ids[0] == hf_greedy_ids[0]


def test_greedy_prefix_matches_hf(hf_greedy_ids, prism_greedy_ids):
    k = _common_prefix_len(prism_greedy_ids, hf_greedy_ids)
    assert k >= MIN_COMMON_PREFIX, (
        f"greedy token sequences diverge too early: common prefix {k} < {MIN_COMMON_PREFIX}\n"
        f"  hf:    {hf_greedy_ids}\n"
        f"  prism: {prism_greedy_ids}"
    )
