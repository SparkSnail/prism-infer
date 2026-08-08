from types import SimpleNamespace

from prism_infer.config import Config
from prism_infer.engine.pd_runner import _activate_received_sequence
from prism_infer.engine.sequence import Sequence, SequenceStatus
from prism_infer.sampling_params import SamplingParams


def test_activate_received_sequence_skips_decode_side_prefill():
    """After receiving prompt KV from P side, seq must enter decode directly
    with P-side first token -- no local prefill should be scheduled."""
    seq = Sequence([10, 20, 30], SamplingParams(max_tokens=4))
    scheduler = SimpleNamespace(waiting=[seq], running=[])
    engine = SimpleNamespace(scheduler=scheduler)

    _activate_received_sequence(engine, seq, [7], first_token=99)

    assert seq.status == SequenceStatus.RUNNING
    assert seq.is_prefill is False
    assert seq.block_table == [7]
    assert seq.num_cached_tokens == seq.num_prompt_tokens == 3
    assert seq.completion_token_ids == [99]
    assert scheduler.waiting == []
    assert scheduler.running == [seq]


def test_config_allows_pd_group_injection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "prism_infer.config.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(max_position_embeddings=4096),
    )
    config = Config(model=str(tmp_path))
    group = object()

    config._pd_group = group
    config._pd_rank = 0

    assert config._pd_group is group
    assert config._pd_rank == 0
