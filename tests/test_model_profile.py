from dataclasses import replace
import hashlib
import json

import pytest

from prism_infer.server.model_profile import (
    ModelProfileError,
    FIXED_QWEN3_0_6B_PROFILE,
    calculate_kv_block_bytes,
    preflight_model_profile,
)


def _model_snapshot(tmp_path, *, num_hidden_layers: int = 28):
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "head_dim": 128,
        "model_type": "qwen3",
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": 8,
        "rope_theta": 1_000_000.0,
        "torch_dtype": "bfloat16",
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
    }


def test_fixed_profile_derives_28_mib_block_and_stable_compatibility():
    profile = FIXED_QWEN3_0_6B_PROFILE

    assert calculate_kv_block_bytes(
        num_hidden_layers=28,
        tokens_per_block=256,
        num_key_value_heads=8,
        head_dim=128,
        dtype_bytes=2,
    ) == 29_360_128
    assert profile.kv_block_bytes == 28 * 1024 * 1024
    assert profile.config_sha256 == (
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
    )
    assert len(profile.kv_compatibility_id) == 64
    assert replace(profile, model_revision="different").kv_compatibility_id \
        != profile.kv_compatibility_id


def test_model_profile_preflight_is_opt_in():
    assert preflight_model_profile({}) is None


def test_model_profile_preflight_accepts_one_coherent_snapshot(tmp_path):
    raw = _model_snapshot(tmp_path)
    profile = replace(
        FIXED_QWEN3_0_6B_PROFILE,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )

    result = preflight_model_profile(
        _environment(tmp_path, profile), profile=profile
    )

    assert result is profile
    assert result.as_resource_report()["kv_compatibility_id"] \
        == profile.kv_compatibility_id


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
    ],
)
def test_model_profile_preflight_rejects_environment_mismatch(
    tmp_path, name, bad_value
):
    raw = _model_snapshot(tmp_path)
    profile = replace(
        FIXED_QWEN3_0_6B_PROFILE,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )
    environment = _environment(tmp_path, profile)
    environment[name] = bad_value

    with pytest.raises(ModelProfileError, match=name):
        preflight_model_profile(environment, profile=profile)


def test_model_profile_preflight_rejects_raw_config_digest_mismatch(tmp_path):
    _model_snapshot(tmp_path)
    environment = _environment(tmp_path, FIXED_QWEN3_0_6B_PROFILE)

    with pytest.raises(ModelProfileError, match="SHA-256 mismatch"):
        preflight_model_profile(environment)


def test_model_profile_preflight_rejects_structural_config_mismatch(tmp_path):
    raw = _model_snapshot(tmp_path, num_hidden_layers=27)
    profile = replace(
        FIXED_QWEN3_0_6B_PROFILE,
        config_sha256=hashlib.sha256(raw).hexdigest(),
    )

    with pytest.raises(ModelProfileError, match="num_hidden_layers mismatch"):
        preflight_model_profile(_environment(tmp_path, profile), profile=profile)
