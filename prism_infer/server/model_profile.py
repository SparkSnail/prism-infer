"""Validate a fixed model profile before CUDA or NCCL initialization."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


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
    """Return the physical byte size of one full K/V block."""
    values = {
        "num_hidden_layers": num_hidden_layers,
        "tokens_per_block": tokens_per_block,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"KV block dimensions must be positive: {values!r}")
    # The leading factor accounts for separate K and V tensors.
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
        """Return all canonical fields used for KV compatibility."""
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

    def as_resource_report(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            **self.compatibility_fields(),
            "kv_compatibility_id": self.kv_compatibility_id,
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
)


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise ModelProfileError(f"missing required 2P2D model environment: {name}")
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
    architectures = value.get("architectures")
    if architectures != ["Qwen3ForCausalLM"]:
        raise ModelProfileError(
            "config.json architectures must be exactly ['Qwen3ForCausalLM']"
        )
    return value


def preflight_model_profile(
    environ: Mapping[str, str] | None = None,
    *,
    profile: FixedModelProfile = FIXED_QWEN3_0_6B_PROFILE,
) -> FixedModelProfile | None:
    """Validate an opt-in profile, or return None for generic model mode."""
    values = os.environ if environ is None else environ
    enabled_profile = values.get("PRISM_MODEL_PROFILE")
    if enabled_profile is None or not enabled_profile.strip():
        return None
    if enabled_profile.strip() != profile.profile_id:
        raise ModelProfileError(
            "unsupported PRISM_MODEL_PROFILE: "
            f"expected {profile.profile_id!r}, got {enabled_profile.strip()!r}"
        )

    _require_exact(values, "PRISM_MODEL_ID", profile.model_id)
    _require_exact(values, "PRISM_MODEL_REVISION", profile.model_revision)
    _require_exact(values, "PRISM_TOKENIZER_REVISION", profile.tokenizer_revision)
    _require_exact(values, "PRISM_MODEL_CONFIG_SHA256", profile.config_sha256)
    _require_exact(values, "PRISM_DTYPE", profile.dtype)
    _require_exact(values, "PRISM_KV_LAYOUT", profile.kv_layout)
    _require_exact(
        values, "PRISM_KV_COMPATIBILITY_ID", profile.kv_compatibility_id
    )
    _require_int(values, "PRISM_TP_SIZE", profile.tensor_parallel_size)
    _require_int(values, "PRISM_TOKENS_PER_BLOCK", profile.tokens_per_block)
    _require_int(values, "PRISM_KV_BLOCK_BYTES", profile.kv_block_bytes)

    model_dir = Path(_required(values, "PRISM_MODEL")).expanduser()
    if not model_dir.is_dir():
        raise ModelProfileError(f"PRISM_MODEL is not a directory: {model_dir}")
    _load_and_verify_config(model_dir, profile)
    if not (model_dir / "tokenizer.json").is_file() \
            or not (model_dir / "tokenizer_config.json").is_file():
        raise ModelProfileError(
            "pinned model snapshot requires tokenizer.json and tokenizer_config.json"
        )
    if not (model_dir / "model.safetensors").is_file() \
            and not (model_dir / "model.safetensors.index.json").is_file():
        raise ModelProfileError("pinned model snapshot has no safetensors weights")
    return profile
