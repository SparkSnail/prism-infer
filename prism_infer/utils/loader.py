import re
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _plain_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, *args):
    """Like default_weight_loader but accepts and ignores extra args (e.g. shard_id).

    Used as fallback in the packed path where weight_loader is called with a
    shard_id argument. Without this, parameters that lack a custom weight_loader
    would raise TypeError when called with three arguments.
    """
    param.data.copy_(loaded_weight)


# Matches expert weight names: "...experts.{global_id}...."
_EXPERT_RE = re.compile(r"(.+\.experts\.)(\d+)(\..*)")


def _remap_expert_weight(model: nn.Module, weight_name: str) -> str | None:
    """Remap experts.{E}.xxx to ep_moe.local_experts.{local_E}.xxx for EP layers.

    Returns the remapped name if the weight belongs to this rank's local experts,
    None if it belongs to another rank (caller should skip), or the original name
    if this layer is not in EP mode.
    """
    m = _EXPERT_RE.match(weight_name)
    if m is None:
        return weight_name  # not an expert weight

    prefix, expert_id_str, suffix = m.group(1), m.group(2), m.group(3)
    global_eid = int(expert_id_str)

    # prefix ends with ".experts." so the MoE module lives at prefix[:-len(".experts.")]
    # e.g. "model.layers.1.mlp.experts." -> "model.layers.1.mlp"
    moe_path = prefix[:-len(".experts.")]
    try:
        moe_mod = model.get_submodule(moe_path)
    except AttributeError:
        return weight_name

    if not hasattr(moe_mod, "ep_moe") or moe_mod.ep_moe is None:
        return weight_name  # single-GPU path, load normally into self.experts[E]

    ep_moe = moe_mod.ep_moe
    base = ep_moe.local_expert_base  # ep_rank * num_local_experts
    num_local = ep_moe.num_local_experts

    if not (base <= global_eid < base + num_local):
        return None  # belongs to another rank, skip

    local_eid = global_eid - base
    return f"{moe_path}.ep_moe.local_experts.{local_eid}{suffix}"


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        # EP: remap experts.{E}.xxx -> ep_moe.local_experts.{le}.xxx,
                        # or skip if this expert belongs to another rank.
                        param_name = _remap_expert_weight(model, param_name)
                        if param_name is None:
                            break
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", _plain_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # EP: remap or skip expert weights on the non-packed path too.
                    param_name = _remap_expert_weight(model, weight_name)
                    if param_name is None:
                        continue
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_weights_zerocopy(safetensors_path: str, weight_name: str,
                          param: "nn.Parameter") -> None:
    """Load one weight tensor via mmap + pinned memory + async H2D DMA."""
    try:
        from safetensors import safe_open as _safe_open
    except ImportError:
        raise ImportError("safetensors required for load_weights_zerocopy")

    with _safe_open(safetensors_path, framework="pt", device="cpu") as f:
        if weight_name not in f.keys():
            raise KeyError(f"{weight_name!r} not found in {safetensors_path}")
        host = f.get_tensor(weight_name)
        copy_to_cuda = param.device.type == "cuda"
        if copy_to_cuda and not host.is_pinned():
            host = host.pin_memory()
        param.data.copy_(host.view(param.data.dtype).reshape(param.data.shape),
                         non_blocking=copy_to_cuda)
