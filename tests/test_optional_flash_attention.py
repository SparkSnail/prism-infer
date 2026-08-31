import pytest

from prism_infer.layers import attention


def test_cpu_environment_can_import_prism_infer_without_flash_attention():
    import prism_infer

    assert prism_infer is not None


def test_cpu_environment_can_import_attention_without_triton():
    from prism_infer.layers import attention

    assert attention.triton is not None or attention._TRITON_IMPORT_ERROR is not None


def test_triton_requirement_is_deferred_until_attention_execution(monkeypatch):
    from prism_infer.layers import attention

    monkeypatch.setattr(attention, "triton", None)
    monkeypatch.setattr(attention, "_TRITON_IMPORT_ERROR", ImportError("missing triton"))

    with pytest.raises(RuntimeError, match="requires Triton"):
        attention._require_triton()


def test_attention_execution_requires_flash_attention(monkeypatch):
    monkeypatch.setattr(attention, "flash_attn_varlen_func", None)
    monkeypatch.setattr(attention, "flash_attn_with_kvcache", None)

    with pytest.raises(RuntimeError, match="requires flash-attn"):
        attention._require_flash_attention()
