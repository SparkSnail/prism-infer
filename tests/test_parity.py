import os

import pytest
import torch

MODEL_PATH = os.environ.get("PRISM_TEST_MODEL")

pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("parity UT requires CUDA GPU", allow_module_level=True)
if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
    pytest.skip("set PRISM_TEST_MODEL to a local model dir", allow_module_level=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from prism_infer.models.qwen3 import Qwen3ForCausalLM  # noqa: E402
from prism_infer.utils.loader import load_model  # noqa: E402
from prism_infer.utils.context import set_context, reset_context  # noqa: E402


@pytest.fixture(scope="module")
def prompt_ids():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
    return ids.to("cuda")


@pytest.fixture(scope="module")
def hf_last_logits(prompt_ids):
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=cfg.dtype, trust_remote_code=True
    ).to("cuda").eval()
    with torch.no_grad():
        logits = model(prompt_ids.unsqueeze(0)).logits[0, -1].float()
    del model
    torch.cuda.empty_cache()
    return logits


@pytest.fixture(scope="module")
def prism_last_logits(prompt_ids):
    hf_config = AutoConfig.from_pretrained(MODEL_PATH)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")
    try:
        model = Qwen3ForCausalLM(hf_config)
        load_model(model, MODEL_PATH)
        model.eval()

        T = prompt_ids.numel()
        positions = torch.arange(T, dtype=torch.long, device="cuda")
        cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=T,
            max_seqlen_k=T,
            slot_mapping=None,
            block_tables=None,
        )
        with torch.no_grad():
            hidden = model.model(prompt_ids, positions)
            logits = model.compute_logits(hidden)
        reset_context()
        out = logits[0].float()
        del model
        torch.cuda.empty_cache()
        return out
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(prev_dtype)


def test_top1_token_matches(hf_last_logits, prism_last_logits):
    assert int(hf_last_logits.argmax()) == int(prism_last_logits.argmax())


def test_logits_l2_within_tolerance(hf_last_logits, prism_last_logits):
    diff = (hf_last_logits - prism_last_logits).norm()
    base = hf_last_logits.norm().clamp_min(1e-6)
    rel_l2 = (diff / base).item()
    assert rel_l2 < 1e-2, f"relative L2 = {rel_l2}"
