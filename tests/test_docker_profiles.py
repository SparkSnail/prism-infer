from pathlib import Path
import re

import pytest

from prism_infer.server.model_profile import (
    FIXED_QWEN3_0_6B_PROFILE,
    FIXED_QWEN3_8B_BF16_TP1_PROFILE,
)


DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"


def _instructions():
    return [
        line.strip()
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _named_stages(instructions):
    stages = set()
    for line in instructions:
        match = re.match(r"FROM\s+\S+\s+AS\s+(\S+)$", line, re.IGNORECASE)
        if match:
            stages.add(match.group(1))
    return stages


def _resolve_profile_stage(variant):
    instructions = _instructions()
    stages = _named_stages(instructions)
    dynamic = next(
        line for line in instructions
        if line.startswith("FROM profile-${PRISM_IMAGE_VARIANT}")
    )
    source = dynamic.split()[1].replace("${PRISM_IMAGE_VARIANT}", variant)
    if source not in stages:
        raise ValueError(f"unknown image variant: {variant}")
    return source


def _stage_text(name, next_marker):
    text = DOCKERFILE.read_text(encoding="utf-8")
    return text.split(f"FROM common AS {name}", 1)[1].split(next_marker, 1)[0]


def _stage_environment(block):
    values = {}
    for line in block.splitlines():
        value = line.strip()
        if value.startswith("ENV "):
            value = value.removeprefix("ENV ")
        if not re.match(r"^[A-Z][A-Z0-9_]*=", value):
            continue
        name, raw = value.rstrip(" \\").split("=", 1)
        values[name] = raw
    return values


def _expected_environment(profile, variant):
    return {
        "PRISM_IMAGE_VARIANT": variant,
        "PRISM_MODEL_PROFILE": profile.profile_id,
        "PRISM_MODEL": f"/opt/models/{profile.model_id.rsplit('/', 1)[1]}",
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
        "PRISM_ENFORCE_EAGER": str(profile.enforce_eager).lower(),
    }


def test_image_variant_arg_is_global_and_resolves_both_named_stages():
    instructions = _instructions()
    first_from = next(i for i, line in enumerate(instructions) if line.startswith("FROM "))

    assert "ARG PRISM_IMAGE_VARIANT=correctness" in instructions[:first_from]
    assert _resolve_profile_stage("correctness") == "profile-correctness"
    assert _resolve_profile_stage("performance") == "profile-performance"
    with pytest.raises(ValueError, match="unknown image variant"):
        _resolve_profile_stage("main")


@pytest.mark.parametrize(
    ("stage", "next_marker", "profile", "variant"),
    [
        (
            "profile-correctness",
            "FROM common AS profile-performance",
            FIXED_QWEN3_0_6B_PROFILE,
            "correctness",
        ),
        (
            "profile-performance",
            "FROM profile-${PRISM_IMAGE_VARIANT} AS selected-profile",
            FIXED_QWEN3_8B_BF16_TP1_PROFILE,
            "performance",
        ),
    ],
)
def test_named_stage_contains_one_coherent_profile_bundle(
    stage, next_marker, profile, variant
):
    block = _stage_text(stage, next_marker)

    assert _stage_environment(block) == _expected_environment(profile, variant)


def test_runtime_image_exposes_profile_provenance_labels():
    text = DOCKERFILE.read_text(encoding="utf-8")

    for key in (
        "ai.sparksnail.prism.image.variant",
        "ai.sparksnail.prism.model.profile",
        "ai.sparksnail.prism.model.revision",
        "ai.sparksnail.prism.model.config-sha256",
        "ai.sparksnail.prism.kv.compatibility-id",
    ):
        assert key in text
