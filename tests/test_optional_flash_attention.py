import pytest

from prism_infer.layers import attention


def test_cpu_environment_can_import_prism_infer_without_flash_attention():
    import prism_infer

    assert prism_infer is not None


def test_attention_execution_requires_flash_attention(monkeypatch):
    monkeypatch.setattr(attention, "flash_attn_varlen_func", None)
    monkeypatch.setattr(attention, "flash_attn_with_kvcache", None)

    with pytest.raises(RuntimeError, match="requires flash-attn"):
        attention._require_flash_attention()
