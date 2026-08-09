"""Validate immutable model profiles before CUDA or NCCL initialization."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType


class ModelProfileError(RuntimeError):
    """The configured profile is missing or mismatches the local snapshot."""


def calculate_kv_block_bytes(
    *,
    num_hidden_layers: int,
    tokens_per_block: int,
    num_key_value_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """Return the physical byte size of one full TP=1 K/V block."""
    values = {
        "num_hidden_layers": num_hidden_layers,
        "tokens_per_block": tokens_per_block,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"KV block dimensions must be positive: {values!r}")
    return (
        2
        * num_hidden_layers
        * tokens_per_block
        * num_key_value_heads
        * head_dim
        * dtype_bytes
    )


@dataclass(frozen=True, slots=True)
class FixedModelProfile:
    profile_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    config_sha256: str
    dtype: str
    dtype_bytes: int
    tensor_parallel_size: int
    tokens_per_block: int
    kv_layout: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float
    enforce_eager: bool

    @property
    def kv_block_bytes(self) -> int:
        return calculate_kv_block_bytes(
            num_hidden_layers=self.num_hidden_layers,
            tokens_per_block=self.tokens_per_block,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            dtype_bytes=self.dtype_bytes,
        )

    def compatibility_fields(self) -> dict[str, object]:
        """Return the canonical fields used for KV compatibility."""
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "config_sha256": self.config_sha256,
            "dtype": self.dtype,
            "kv_layout": self.kv_layout,
            "tokens_per_block": self.tokens_per_block,
            "kv_block_bytes": self.kv_block_bytes,
            "tensor_parallel_size": self.tensor_parallel_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
        }

    @property
    def kv_compatibility_id(self) -> str:
        encoded = json.dumps(
            self.compatibility_fields(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def runtime_fields(self) -> dict[str, object]:
        """Return runtime limits fixed by this profile."""
        return {
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
        }

    def engine_kwargs(self, *, enforce_eager: bool | None = None) -> dict[str, object]:
        """Return shared Config/LLM keyword arguments for this profile."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "kvcache_block_size": self.tokens_per_block,
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": (
                self.enforce_eager if enforce_eager is None else enforce_eager
            ),
        }

    def as_resource_report(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            **self.compatibility_fields(),
            "kv_compatibility_id": self.kv_compatibility_id,
            **self.runtime_fields(),
        }


FIXED_QWEN3_0_6B_PROFILE = FixedModelProfile(
    profile_id="week12-qwen3-0.6b",
    model_id="Qwen/Qwen3-0.6B",
    model_revision="9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439",
    tokenizer_revision="9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439",
    config_sha256="660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    dtype="bfloat16",
    dtype_bytes=2,
    tensor_parallel_size=1,
    tokens_per_block=256,
    kv_layout="NHDC",
    num_hidden_layers=28,
    num_key_value_heads=8,
    head_dim=128,
    rope_theta=1_000_000.0,
    max_model_len=4096,
    max_num_batched_tokens=16384,
    max_num_seqs=512,
    gpu_memory_utilization=0.9,
    enforce_eager=False,
)

FIXED_QWEN3_8B_BF16_TP1_PROFILE = FixedModelProfile(
    profile_id="qwen3-8b-bf16-tp1",
    model_id="Qwen/Qwen3-8B",
    model_revision="b968826d9c46dd6066d109eabc6255188de91218",
    tokenizer_revision="b968826d9c46dd6066d109eabc6255188de91218",
    config_sha256="f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    dtype="bfloat16",
    dtype_bytes=2,
    tensor_parallel_size=1,
    tokens_per_block=256,
    kv_layout="NHDC",
    num_hidden_layers=36,
    num_key_value_heads=8,
    head_dim=128,
    rope_theta=1_000_000.0,
    max_model_len=4096,
    max_num_batched_tokens=16384,
    max_num_seqs=128,
    gpu_memory_utilization=0.9,
    enforce_eager=False,
)

FIXED_MODEL_PROFILES: Mapping[str, FixedModelProfile] = MappingProxyType(
    {
        FIXED_QWEN3_0_6B_PROFILE.profile_id: FIXED_QWEN3_0_6B_PROFILE,
        FIXED_QWEN3_8B_BF16_TP1_PROFILE.profile_id:
            FIXED_QWEN3_8B_BF16_TP1_PROFILE,
    }
)


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise ModelProfileError(f"missing required model profile environment: {name}")
    return value.strip()


def _require_exact(
    environ: Mapping[str, str], name: str, expected: str
) -> None:
    actual = _required(environ, name)
    if actual != expected:
        raise ModelProfileError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_int(
    environ: Mapping[str, str], name: str, expected: int
) -> None:
    raw = _required(environ, name)
    try:
        actual = int(raw)
    except ValueError as exc:
        raise ModelProfileError(f"{name} must be an integer, got {raw!r}") from exc
    if actual != expected:
        raise ModelProfileError(
            f"{name} mismatch: expected {expected}, got {actual}"
        )


def _require_float(
    environ: Mapping[str, str], name: str, expected: float
) -> None:
    raw = _required(environ, name)
    try:
        actual = float(raw)
    except ValueError as exc:
        raise ModelProfileError(f"{name} must be a float, got {raw!r}") from exc
    if not math.isfinite(actual) or actual != expected:
        raise ModelProfileError(
            f"{name} mismatch: expected {expected}, got {raw!r}"
        )


def _require_bool(
    environ: Mapping[str, str], name: str, expected: bool
) -> None:
    raw = _required(environ, name)
    expected_raw = str(expected).lower()
    if raw not in {"true", "false"} or raw != expected_raw:
        raise ModelProfileError(
            f"{name} mismatch: expected {expected_raw!r}, got {raw!r}"
        )


def _load_and_verify_config(
    model_dir: Path, profile: FixedModelProfile
) -> dict[str, object]:
    config_path = model_dir / "config.json"
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ModelProfileError(f"cannot read pinned model config: {config_path}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != profile.config_sha256:
        raise ModelProfileError(
            "config.json SHA-256 mismatch: "
            f"expected {profile.config_sha256}, got {digest}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelProfileError("pinned model config.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ModelProfileError("pinned model config.json must be an object")
    expected_fields: dict[str, object] = {
        "model_type": "qwen3",
        "num_hidden_layers": profile.num_hidden_layers,
        "num_key_value_heads": profile.num_key_value_heads,
        "head_dim": profile.head_dim,
        "rope_theta": profile.rope_theta,
        "torch_dtype": profile.dtype,
    }
    for key, expected in expected_fields.items():
        actual = value.get(key)
        if actual != expected:
            raise ModelProfileError(
                f"config.json {key} mismatch: expected {expected!r}, got {actual!r}"
            )
    if value.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ModelProfileError(
            "config.json architectures must be exactly ['Qwen3ForCausalLM']"
        )
    return value


def preflight_model_profile(
    environ: Mapping[str, str] | None = None,
    *,
    profile: FixedModelProfile | None = None,
) -> FixedModelProfile | None:
    """Validate an opt-in profile, or return None for generic model mode."""
    values = os.environ if environ is None else environ
    enabled_profile = values.get("PRISM_MODEL_PROFILE")
    if enabled_profile is None or not enabled_profile.strip():
        return None
    profile_id = enabled_profile.strip()
    selected = profile
    if selected is None:
        selected = FIXED_MODEL_PROFILES.get(profile_id)
        if selected is None:
            raise ModelProfileError(
                f"unsupported PRISM_MODEL_PROFILE: {profile_id!r}"
            )
    elif profile_id != selected.profile_id:
        raise ModelProfileError(
            "PRISM_MODEL_PROFILE mismatch: "
            f"expected {selected.profile_id!r}, got {profile_id!r}"
        )

    _require_exact(values, "PRISM_MODEL_ID", selected.model_id)
    _require_exact(values, "PRISM_MODEL_REVISION", selected.model_revision)
    _require_exact(values, "PRISM_TOKENIZER_REVISION", selected.tokenizer_revision)
    _require_exact(values, "PRISM_MODEL_CONFIG_SHA256", selected.config_sha256)
    _require_exact(values, "PRISM_DTYPE", selected.dtype)
    _require_exact(values, "PRISM_KV_LAYOUT", selected.kv_layout)
    _require_exact(
        values, "PRISM_KV_COMPATIBILITY_ID", selected.kv_compatibility_id
    )
    _require_int(values, "PRISM_TP_SIZE", selected.tensor_parallel_size)
    _require_int(values, "PRISM_TOKENS_PER_BLOCK", selected.tokens_per_block)
    _require_int(values, "PRISM_KV_BLOCK_BYTES", selected.kv_block_bytes)
    _require_int(
        values, "PRISM_MODEL_NUM_HIDDEN_LAYERS", selected.num_hidden_layers
    )
    _require_int(
        values, "PRISM_MODEL_NUM_KEY_VALUE_HEADS", selected.num_key_value_heads
    )
    _require_int(values, "PRISM_MODEL_HEAD_DIM", selected.head_dim)
    _require_float(values, "PRISM_MODEL_ROPE_THETA", selected.rope_theta)
    _require_int(values, "PRISM_MAX_MODEL_LEN", selected.max_model_len)
    _require_int(
        values, "PRISM_MAX_NUM_BATCHED_TOKENS", selected.max_num_batched_tokens
    )
    _require_int(values, "PRISM_MAX_NUM_SEQS", selected.max_num_seqs)
    _require_float(
        values, "PRISM_GPU_MEMORY_UTILIZATION", selected.gpu_memory_utilization
    )
    _require_bool(values, "PRISM_ENFORCE_EAGER", selected.enforce_eager)

    model_dir = Path(_required(values, "PRISM_MODEL")).expanduser()
    if not model_dir.is_dir():
        raise ModelProfileError(f"PRISM_MODEL is not a directory: {model_dir}")
    _load_and_verify_config(model_dir, selected)
    if not (model_dir / "tokenizer.json").is_file() \
            or not (model_dir / "tokenizer_config.json").is_file():
        raise ModelProfileError(
            "pinned model snapshot requires tokenizer.json and tokenizer_config.json"
        )
    if not (model_dir / "model.safetensors").is_file() \
            and not (model_dir / "model.safetensors.index.json").is_file():
        raise ModelProfileError("pinned model snapshot has no safetensors weights")
    return selected
