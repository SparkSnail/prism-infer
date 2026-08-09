import argparse
from dataclasses import replace
import hashlib
import json

import pytest

from prism_infer.server.model_profile import FIXED_QWEN3_0_6B_PROFILE
from prism_infer.server.unified_baseline import (
    OUTPUT_MARKER,
    SCHEMA_VERSION,
    format_result_line,
    load_input_ids,
    run,
)


def _snapshot(tmp_path):
    profile = FIXED_QWEN3_0_6B_PROFILE
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "head_dim": profile.head_dim,
        "max_position_embeddings": profile.max_model_len,
        "model_type": "qwen3",
        "num_hidden_layers": profile.num_hidden_layers,
        "num_key_value_heads": profile.num_key_value_heads,
        "rope_theta": profile.rope_theta,
        "torch_dtype": profile.dtype,
    }
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "config.json").write_bytes(raw)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"test-only")
    return replace(profile, config_sha256=hashlib.sha256(raw).hexdigest())


def _environment(model, profile):
    return {
        "PRISM_MODEL_PROFILE": profile.profile_id,
        "PRISM_MODEL": str(model),
        "PRISM_MODEL_ID": profile.model_id,
        "PRISM_MODEL_REVISION": profile.model_revision,
        "PRISM_TOKENIZER_REVISION": profile.tokenizer_revision,
        "PRISM_MODEL_CONFIG_SHA256": profile.config_sha256,
        "PRISM_DTYPE": profile.dtype,
        "PRISM_TP_SIZE": str(profile.tensor_parallel_size),
        "PRISM_TOKENS_PER_BLOCK": str(profile.tokens_per_block),
        "PRISM_KV_BLOCK_BYTES": str(profile.kv_block_bytes),
        "PRISM_KV_LAYOUT": profile.kv_layout,
        "PRISM_KV_COMPATIBILITY_ID": profile.kv_compatibility_id,
        "PRISM_MODEL_NUM_HIDDEN_LAYERS": str(profile.num_hidden_layers),
        "PRISM_MODEL_NUM_KEY_VALUE_HEADS": str(profile.num_key_value_heads),
        "PRISM_MODEL_HEAD_DIM": str(profile.head_dim),
        "PRISM_MODEL_ROPE_THETA": str(profile.rope_theta),
        "PRISM_MAX_MODEL_LEN": str(profile.max_model_len),
        "PRISM_MAX_NUM_BATCHED_TOKENS": str(profile.max_num_batched_tokens),
        "PRISM_MAX_NUM_SEQS": str(profile.max_num_seqs),
        "PRISM_GPU_MEMORY_UTILIZATION": str(profile.gpu_memory_utilization),
        "PRISM_ENFORCE_EAGER": "false",
    }


def _args(input_path, model, profile, **overrides):
    values = {
        "input": str(input_path),
        "expected_input_tokens": 3,
        "expected_output_tokens": 2,
        "profile_id": profile.profile_id,
        "model": str(model),
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "tokenizer_revision": profile.tokenizer_revision,
        "config_sha256": profile.config_sha256,
        "kv_compatibility_id": profile.kv_compatibility_id,
        "temperature": 0.0,
        "ignore_eos": True,
        **overrides,
    }
    return argparse.Namespace(**values)


class _FakeLLM:
    def __init__(self, output_ids):
        self.output_ids = output_ids
        self.calls = []

    def generate(self, prompts, sampling_params, *, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [{"token_ids": self.output_ids}]


def test_unified_baseline_runs_profile_bound_greedy_reference(tmp_path):
    profile = _snapshot(tmp_path)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"input_ids": [11, 12, 13]}), encoding="utf-8")
    fake = _FakeLLM([21, 22])
    created = {}

    def llm_factory(model, **kwargs):
        created.update(model=model, kwargs=kwargs)
        return fake

    result = run(
        _args(input_path, tmp_path, profile),
        environ=_environment(tmp_path, profile),
        profile=profile,
        llm_factory=llm_factory,
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert set(result) == {
        "schema_version", "model_profile", "sampling", "engine",
        "input_ids", "output_ids",
    }
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["model_profile"] == profile.as_resource_report()
    assert result["sampling"] == {
        "temperature": 0.0,
        "max_tokens": 2,
        "ignore_eos": True,
    }
    assert result["engine"] == profile.engine_kwargs(enforce_eager=True)
    assert result["input_ids"] == [11, 12, 13]
    assert result["output_ids"] == [21, 22]
    assert created == {
        "model": str(tmp_path),
        "kwargs": profile.engine_kwargs(enforce_eager=True),
    }
    assert fake.calls == [([[11, 12, 13]], {
        "temperature": 0.0, "max_tokens": 2, "ignore_eos": True,
    }, False)]


@pytest.mark.parametrize(
    "value",
    [[], [1, True], [1, -1], [1, 2**64], {"input_ids": [1], "extra": 2}],
)
def test_unified_baseline_rejects_invalid_uint64_input(tmp_path, value):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="uint64 token IDs|exactly|input object"):
        load_input_ids(path, expected_count=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"profile_id": "wrong"}, "--profile-id"),
        ({"model_revision": "main"}, "--model-revision"),
        ({"temperature": 0.1}, "--temperature"),
        ({"ignore_eos": False}, "--ignore-eos"),
    ],
)
def test_unified_baseline_rejects_cli_or_profile_drift(
    tmp_path, overrides, message
):
    profile = _snapshot(tmp_path)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([11, 12, 13]), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run(
            _args(input_path, tmp_path, profile, **overrides),
            environ=_environment(tmp_path, profile),
            profile=profile,
            llm_factory=lambda *args, **kwargs: _FakeLLM([21, 22]),
            sampling_params_factory=lambda **kwargs: kwargs,
        )


def test_unified_baseline_rejects_short_output(tmp_path):
    profile = _snapshot(tmp_path)
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([11, 12, 13]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly 2 uint64 token IDs"):
        run(
            _args(input_path, tmp_path, profile),
            environ=_environment(tmp_path, profile),
            profile=profile,
            llm_factory=lambda *args, **kwargs: _FakeLLM([21]),
            sampling_params_factory=lambda **kwargs: kwargs,
        )


def test_unified_baseline_marker_is_single_canonical_json_line():
    line = format_result_line({"schema_version": SCHEMA_VERSION})

    assert line.startswith(OUTPUT_MARKER)
    assert line.count(OUTPUT_MARKER) == 1
    assert line.endswith("}")
