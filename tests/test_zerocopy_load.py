import pytest
import torch
import torch.nn as nn


def test_matches_standard_load(tmp_path):
    try:
        import safetensors.torch as st
    except ImportError:
        pytest.skip("safetensors not installed")

    from prism_infer.utils.loader import load_weights_zerocopy

    w = torch.randn(64, 64)
    path = str(tmp_path / "w.safetensors")
    st.save_file({"w": w}, path)

    param = nn.Parameter(torch.empty(64, 64))
    load_weights_zerocopy(path, "w", param)
    if param.data.is_cuda:
        torch.cuda.synchronize()

    assert torch.allclose(st.load_file(path)["w"], param.data.cpu(), atol=1e-6)


def test_missing_key_raises(tmp_path):
    try:
        import safetensors.torch as st
    except ImportError:
        pytest.skip("safetensors not installed")

    from prism_infer.utils.loader import load_weights_zerocopy

    st.save_file({"w": torch.randn(8, 8)}, str(tmp_path / "w.safetensors"))
    param = nn.Parameter(torch.empty(8, 8))
    with pytest.raises(KeyError):
        load_weights_zerocopy(str(tmp_path / "w.safetensors"), "missing", param)
