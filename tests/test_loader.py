from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file

from prism_infer.layers.moe import MoEExpert
from prism_infer.layers.linear import ReplicatedLinear
from prism_infer.utils.loader import load_model


class TinyMoEModel(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, hidden: int, inter: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [MoEExpert(hidden, inter) for _ in range(num_experts)]
        )
        self.gate = ReplicatedLinear(hidden, num_experts, bias=False)


def test_load_moe_weights(tmp_path):
    hidden, inter, num_experts = 8, 4, 2
    model = TinyMoEModel(hidden, inter, num_experts)

    weights = {}
    for e in range(num_experts):
        weights[f"experts.{e}.gate_proj.weight"] = torch.randn(inter, hidden)
        weights[f"experts.{e}.up_proj.weight"] = torch.randn(inter, hidden)
        weights[f"experts.{e}.down_proj.weight"] = torch.randn(hidden, inter)
    weights["gate.weight"] = torch.randn(num_experts, hidden)

    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))

    load_model(model, str(tmp_path))

    for e in range(num_experts):
        fused = model.experts[e].gate_up_proj.weight.data
        assert torch.allclose(fused[:inter], weights[f"experts.{e}.gate_proj.weight"])
        assert torch.allclose(fused[inter:], weights[f"experts.{e}.up_proj.weight"])
        assert torch.allclose(
            model.experts[e].down_proj.weight.data,
            weights[f"experts.{e}.down_proj.weight"],
        )
    assert torch.allclose(model.gate.weight.data, weights["gate.weight"])


def test_load_covers_all_params(tmp_path):
    hidden, inter, num_experts = 8, 4, 3
    model = TinyMoEModel(hidden, inter, num_experts)
    for p in model.parameters():
        p.data.fill_(float("nan"))

    weights = {}
    for e in range(num_experts):
        weights[f"experts.{e}.gate_proj.weight"] = torch.randn(inter, hidden)
        weights[f"experts.{e}.up_proj.weight"] = torch.randn(inter, hidden)
        weights[f"experts.{e}.down_proj.weight"] = torch.randn(hidden, inter)
    weights["gate.weight"] = torch.randn(num_experts, hidden)
    save_file(weights, str(tmp_path / "model.safetensors"))

    load_model(model, str(tmp_path))

    for name, p in model.named_parameters():
        assert torch.isfinite(p.data).all(), f"param {name} not fully loaded"
