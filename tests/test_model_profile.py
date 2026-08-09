from dataclasses import replace
import hashlib
import json

import pytest

from prism_infer.server.model_profile import (
    FIXED_MODEL_PROFILES,
    FIXED_QWEN3_0_6B_PROFILE,
    FIXED_QWEN3_8B_BF16_TP1_PROFILE,
    ModelProfileError,
    calculate_kv_block_bytes,
    preflight_model_profile,
)


def _model_snapshot(tmp_path, profile, **overrides):
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "head_dim": profile.head_dim,
        "model_type": "qwen3",
        "num_hidden_layers": profile.num_hidden_layers,
        "num_key_value_heads": profile.num_key_value_heads,
        "rope_theta": profile.rope_theta,
        "torch_dtype": profile.dtype,
        **overrides,
    }
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "config.json").write_bytes(raw)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"test-only")
    return raw


def _environment(tmp_path, profile):
    return {
        "PRISM_MODEL_PROFILE": profile.profile_id,
        "PRISM_MODEL": str(tmp_path),
        "PRISM_MODEL_ID": profile.model_id,
        "PRISM_MODEL_REVISION": profile.model_revision,
        "PRISM_TOKENIZER_REVISION": profile.tokenizer_revision,
        "PRISM_MODEL_CONFIG_SHA256": profile.config_sha256,
        "PRISM_DTYPE": profile.dtype,
        "PRISM_TP_SIZE": str(profile.tensor_parallel_size),
        "PRISM_TOKENS_PER_BLOCK": str(profile.tokens_per_block),
        "PRISM_KV_LAYOUT": profile.kv_layout,
        "PRISM_KV_BLOCK_BYTES": str(profile.kv_block_bytes),
        "PRISM_KV_COMPATIBILITY_ID": profile.kv_compatibility_id,
        "PRISM_MODEL_NUM_HIDDEN_LAYERS": str(profile.num_hidden_layers),
        "PRISM_MODEL_NUM_KEY_VALUE_HEADS": str(profile.num_key_value_heads),
        "PRISM_MODEL_HEAD_DIM": str(profile.head_dim),
        "PRISM_MODEL_ROPE_THETA": str(profile.rope_theta),
        "PRISM_MAX_MODEL_LEN": str(profile.max_model_len),
        "PRISM_MAX_NUM_BATCHED_TOKENS": str(profile.max_num_batched_tokens),
        "PRISM_MAX_NUM_SEQS": str(profile.max_num_seqs),
        "PRISM_GPU_MEMORY_UTILIZATION": str(profile.gpu_memory_utilization),
        "PRISM_ENFORCE_EAGER": str(profile.enforce_eager).lower(),
    }


@pytest.mark.parametrize(
    ("profile", "expected_bytes", "expected_compatibility_id"),
    [
        (
            FIXED_QWEN3_0_6B_PROFILE,
            29_360_128,
            "a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19",
        ),
        (
            FIXED_QWEN3_8B_BF16_TP1_PROFILE,
            37_748_736,
            "2647222531d143800a56551dbe8d030c535a1bb2e0e47ff2a12f0781964f4c6c",
        ),
    ],
)
def test_fixed_profiles_derive_exact_block_and_compatibility(
    profile, expected_bytes, expected_compatibility_id
):
    assert calculate_kv_block_bytes(
        num_hidden_layers=profile.num_hidden_layers,
        tokens_per_block=profile.tokens_per_block,
        num_key_value_heads=profile.num_key_value_heads,
        head_dim=profile.head_dim,
        dtype_bytes=profile.dtype_bytes,
    ) == expected_bytes
    assert profile.kv_block_bytes == expected_bytes
    assert profile.kv_compatibility_id == expected_compatibility_id
    assert replace(profile, model_revision="different").kv_compatibility_id \
        != profile.kv_compatibility_id


def test_profile_registry_contains_only_the_two_fixed_bundles():
    assert set(FIXED_MODEL_PROFILES) == {
        "week12-qwen3-0.6b",
        "qwen3-8b-bf16-tp1",
    }
    assert FIXED_QWEN3_0_6B_PROFILE.engine_kwargs() == {
        "tensor_parallel_size": 1,
        "kvcache_block_size": 256,
        "max_model_len": 4096,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 512,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": False,
    }
    assert FIXED_QWEN3_8B_BF16_TP1_PROFILE.engine_kwargs()["max_num_seqs"] == 128


def test_model_profile_preflight_is_opt_in():
    assert preflight_model_profile({}) is None


@pytest.mark.parametrize(
    "fixed_profile",
    [FIXED_QWEN3_0_6B_PROFILE, FIXED_QWEN3_8B_BF16_TP1_PROFILE],
)
def test_model_profile_preflight_selects_one_coherent_snapshot(
    tmp_path, fixed_profile
):
    raw = _model_snapshot(tmp_path, fixed_profile)
    profile = replace(
        fixed_profile,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )

    selected = preflight_model_profile(
        _environment(tmp_path, profile), profile=profile
    )

    assert selected is profile
    assert selected.as_resource_report()["kv_compatibility_id"] \
        == profile.kv_compatibility_id
    assert selected.as_resource_report()["max_num_seqs"] == profile.max_num_seqs


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("PRISM_MODEL_REVISION", "main"),
        ("PRISM_TOKENIZER_REVISION", "main"),
        ("PRISM_DTYPE", "float16"),
        ("PRISM_TP_SIZE", "2"),
        ("PRISM_TOKENS_PER_BLOCK", "128"),
        ("PRISM_KV_BLOCK_BYTES", "117440512"),
        ("PRISM_KV_COMPATIBILITY_ID", "template-not-a-real-fingerprint"),
        ("PRISM_MODEL_NUM_HIDDEN_LAYERS", "28"),
        ("PRISM_MODEL_NUM_KEY_VALUE_HEADS", "4"),
        ("PRISM_MODEL_HEAD_DIM", "64"),
        ("PRISM_MODEL_ROPE_THETA", "10000"),
        ("PRISM_MAX_MODEL_LEN", "8192"),
        ("PRISM_MAX_NUM_BATCHED_TOKENS", "32768"),
        ("PRISM_MAX_NUM_SEQS", "256"),
        ("PRISM_GPU_MEMORY_UTILIZATION", "0.8"),
        ("PRISM_ENFORCE_EAGER", "true"),
    ],
)
def test_model_profile_preflight_rejects_environment_mismatch(
    tmp_path, name, bad_value
):
    fixed_profile = FIXED_QWEN3_8B_BF16_TP1_PROFILE
    raw = _model_snapshot(tmp_path, fixed_profile)
    profile = replace(
        fixed_profile,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    environment = _environment(tmp_path, profile)
    environment[name] = bad_value

    with pytest.raises(ModelProfileError, match=name):
        preflight_model_profile(environment, profile=profile)


def test_model_profile_preflight_rejects_hybrid_profile(tmp_path):
    fixed_profile = FIXED_QWEN3_8B_BF16_TP1_PROFILE
    raw = _model_snapshot(tmp_path, fixed_profile)
    profile = replace(
        fixed_profile,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    environment = _environment(tmp_path, profile)
    environment["PRISM_MODEL_REVISION"] = FIXED_QWEN3_0_6B_PROFILE.model_revision

    with pytest.raises(ModelProfileError, match="PRISM_MODEL_REVISION"):
        preflight_model_profile(environment, profile=profile)


def test_model_profile_preflight_rejects_unknown_profile():
    with pytest.raises(ModelProfileError, match="unsupported PRISM_MODEL_PROFILE"):
        preflight_model_profile({"PRISM_MODEL_PROFILE": "qwen3-main"})


def test_model_profile_preflight_rejects_raw_config_digest_mismatch(tmp_path):
    _model_snapshot(tmp_path, FIXED_QWEN3_0_6B_PROFILE)
    environment = _environment(tmp_path, FIXED_QWEN3_0_6B_PROFILE)

    with pytest.raises(ModelProfileError, match="SHA-256 mismatch"):
        preflight_model_profile(environment)


def test_model_profile_preflight_rejects_structural_config_mismatch(tmp_path):
    fixed_profile = FIXED_QWEN3_0_6B_PROFILE
    raw = _model_snapshot(tmp_path, fixed_profile, num_hidden_layers=27)
    profile = replace(
        fixed_profile,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )

    with pytest.raises(ModelProfileError, match="num_hidden_layers mismatch"):
        preflight_model_profile(_environment(tmp_path, profile), profile=profile)
