"""Run a fixed-profile unified reference inside a one-off GPU job."""

import argparse
from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path

from prism_infer.server.model_profile import (
    FixedModelProfile,
    ModelProfileError,
    preflight_model_profile,
)


SCHEMA_VERSION = "prism.unified_baseline/v1"
OUTPUT_MARKER = "PRISM_UNIFIED_BASELINE="


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-input-tokens", required=True, type=int)
    parser.add_argument("--expected-output-tokens", required=True, type=int)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--kv-compatibility-id", required=True)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--ignore-eos", action="store_true", required=True)
    return parser


def load_input_ids(path: str | Path, *, expected_count: int) -> list[int]:
    """Load an exact-count uint64 token fixture."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if set(value) != {"input_ids"}:
            raise ValueError("unified baseline input object must contain only input_ids")
        value = value["input_ids"]
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(
            f"unified baseline requires exactly {expected_count} uint64 token IDs"
        )
    if not all(type(token) is int and 0 <= token < 2**64 for token in value):
        raise ValueError(
            f"unified baseline requires exactly {expected_count} uint64 token IDs"
        )
    return value


def _create_llm(model: str, **kwargs: object) -> object:
    from prism_infer import LLM

    return LLM(model, **kwargs)


def _create_sampling_params(**kwargs: object) -> object:
    from prism_infer import SamplingParams

    return SamplingParams(**kwargs)


def _require_cli_metadata(
    args: argparse.Namespace,
    profile: FixedModelProfile,
    environ: Mapping[str, str],
) -> None:
    fields = {
        "--profile-id": (args.profile_id, profile.profile_id),
        "--model-id": (args.model_id, profile.model_id),
        "--model-revision": (args.model_revision, profile.model_revision),
        "--tokenizer-revision": (
            args.tokenizer_revision,
            profile.tokenizer_revision,
        ),
        "--config-sha256": (args.config_sha256, profile.config_sha256),
        "--kv-compatibility-id": (
            args.kv_compatibility_id,
            profile.kv_compatibility_id,
        ),
    }
    for flag, (actual, expected) in fields.items():
        if actual != expected:
            raise ValueError(
                f"{flag} mismatch: expected {expected!r}, got {actual!r}"
            )
    configured_model = environ.get("PRISM_MODEL")
    if configured_model is None or not configured_model.strip():
        raise ModelProfileError("missing required model profile environment: PRISM_MODEL")
    if Path(args.model).expanduser().resolve() \
            != Path(configured_model).expanduser().resolve():
        raise ValueError("--model does not match the preflighted PRISM_MODEL path")


def _validate_run_args(args: argparse.Namespace, profile: FixedModelProfile) -> None:
    counts = {
        "--expected-input-tokens": args.expected_input_tokens,
        "--expected-output-tokens": args.expected_output_tokens,
    }
    for flag, value in counts.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{flag} must be a positive integer")
    if args.expected_input_tokens + args.expected_output_tokens > profile.max_model_len:
        raise ValueError("baseline input and output exceed the fixed max_model_len")
    if not math.isfinite(args.temperature) or args.temperature != 0.0:
        raise ValueError("--temperature must be exactly 0")
    if args.ignore_eos is not True:
        raise ValueError("--ignore-eos is required")


def run(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    profile: FixedModelProfile | None = None,
    llm_factory: Callable[..., object] | None = None,
    sampling_params_factory: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Run a profile-bound greedy reference and return its token evidence."""
    values = os.environ if environ is None else environ
    selected = preflight_model_profile(values, profile=profile)
    if selected is None:
        raise ModelProfileError("PRISM_MODEL_PROFILE is required for unified baseline")
    _require_cli_metadata(args, selected, values)
    _validate_run_args(args, selected)
    input_ids = load_input_ids(
        args.input, expected_count=args.expected_input_tokens
    )

    engine = selected.engine_kwargs(enforce_eager=True)
    create_llm = _create_llm if llm_factory is None else llm_factory
    create_sampling = (
        _create_sampling_params
        if sampling_params_factory is None else sampling_params_factory
    )
    llm = create_llm(args.model, **engine)
    sampling = {
        "temperature": 0.0,
        "max_tokens": args.expected_output_tokens,
        "ignore_eos": True,
    }
    outputs = getattr(llm, "generate")(
        [input_ids], create_sampling(**sampling), use_tqdm=False
    )
    if not isinstance(outputs, list) or len(outputs) != 1 \
            or not isinstance(outputs[0], dict):
        raise RuntimeError("unified baseline engine returned an invalid result envelope")
    output_ids = outputs[0].get("token_ids")
    if not isinstance(output_ids, list) \
            or len(output_ids) != args.expected_output_tokens \
            or not all(
                type(token) is int and 0 <= token < 2**64 for token in output_ids
            ):
        raise RuntimeError(
            "unified baseline did not produce exactly "
            f"{args.expected_output_tokens} uint64 token IDs"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_profile": selected.as_resource_report(),
        "sampling": sampling,
        "engine": engine,
        "input_ids": input_ids,
        "output_ids": output_ids,
    }


def format_result_line(result: Mapping[str, object]) -> str:
    return OUTPUT_MARKER + json.dumps(
        result, sort_keys=True, separators=(",", ":")
    )


def main() -> None:
    result = run(build_parser().parse_args())
    print(format_result_line(result))


if __name__ == "__main__":
    main()
