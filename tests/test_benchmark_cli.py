import importlib.util
import json
from pathlib import Path

import pytest


BENCH_PATH = Path(__file__).parents[1] / "bench" / "bench.py"
SPEC = importlib.util.spec_from_file_location("prism_infer_bench", BENCH_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_common_options_work_before_and_after_subcommand() -> None:
    before = bench.build_parser().parse_args(
        ["--model", "/models/a", "--input-len", "128", "throughput"]
    )
    after = bench.build_parser().parse_args(
        ["throughput", "--model", "/models/b", "--input-len", "256"]
    )

    assert before.model == "/models/a"
    assert before.input_len == 128
    assert after.model == "/models/b"
    assert after.input_len == 256


def test_sweep_batch_sizes_are_validated_as_integers(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    args = bench.build_parser().parse_args(
        [
            "sweep",
            "--model",
            str(model_dir),
            "--batch-sizes",
            "1, 4,16",
        ]
    )

    bench._validate_args(args, bench.build_parser())

    assert args.batch_sizes == [1, 4, 16]


def test_validation_rejects_context_overflow_and_parallel_modes(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    parser = bench.build_parser()

    overflow = parser.parse_args(
        [
            "throughput",
            "--model",
            str(model_dir),
            "--input-len",
            "8",
            "--output-len",
            "4",
            "--max-model-len",
            "10",
        ]
    )
    with pytest.raises(SystemExit):
        bench._validate_args(overflow, parser)

    parallel = bench.build_parser().parse_args(
        ["throughput", "--model", str(model_dir), "--tp", "2", "--ep", "2"]
    )
    with pytest.raises(SystemExit):
        bench._validate_args(parallel, bench.build_parser())


def test_prompt_generation_is_seeded_and_vocab_bounded() -> None:
    first, _ = bench._make_requests(2, 6, 2, 17, 0.0, 32)
    second, _ = bench._make_requests(2, 6, 2, 17, 0.0, 32)
    other, _ = bench._make_requests(2, 6, 2, 18, 0.0, 32)

    assert first == second
    assert first != other
    assert all(0 <= token < 32 for prompt in first for token in prompt)
    assert bench._prompt_digest(first) != bench._prompt_digest(other)


def test_vocab_bound_prefers_model_embedding_size() -> None:
    class Config:
        class HF:
            vocab_size = 24

        hf_config = HF()

    class Tokenizer:
        vocab_size = 32

        def __len__(self):
            return 40

    class LLM:
        model_runner = type("Runner", (), {"config": Config()})()
        tokenizer = Tokenizer()

    assert bench._tokenizer_vocab_size(LLM()) == 24


def test_run_counts_idle_steps_without_changing_the_guard() -> None:
    class Sampling:
        max_tokens = 2

    class FakeLLM:
        def __init__(self) -> None:
            self.index = 0
            self.steps = [
                ([], 0),
                ([], 2),
                ([], -1),
                ([(1, [7, 8])], -1),
            ]

        def add_request(self, prompt, sampling_params) -> None:
            return None

        def is_finished(self) -> bool:
            return self.index == len(self.steps)

        def step(self):
            result = self.steps[self.index]
            self.index += 1
            return result

    original_load_torch = bench._load_torch
    bench._load_torch = lambda: None
    try:
        result = bench._run(FakeLLM(), [[1]], [Sampling()], max_idle_steps=1)
    finally:
        bench._load_torch = original_load_torch

    assert result["idle_steps"] == 1
    assert result["completed_requests"] == 1


def test_median_metrics_are_stable() -> None:
    samples = [
        {"decode_tokens_per_second": 10},
        {"decode_tokens_per_second": 30},
        {"decode_tokens_per_second": 20},
    ]

    assert bench._median_metrics(
        samples, ("decode_tokens_per_second",)
    ) == {"decode_tokens_per_second": 20.0}


def test_json_output_preserves_result_and_configuration(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "summary": {"decode_tokens_per_second": 42.5},
    }
    output = tmp_path / "nested" / "result.json"

    bench._write_output(payload, str(output))

    assert json.loads(output.read_text(encoding="utf-8")) == payload
