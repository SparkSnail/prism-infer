import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from prism_infer.layers.activation import SiluAndMul
from prism_infer.layers.linear import (
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from prism_infer.engine.parallel.expert_parallel import (
    ExpertParallelMoE,
    get_expert_parallel_size,
)


class MoEExpert(nn.Module):
    """Single SwiGLU expert. Gate and up projections are fused into one matmul."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class MoE(nn.Module):
    """Token-choice MoE following Qwen/Mixtral routing: softmax -> top-k -> re-norm.

    When expert_parallel_size > 1 (set via set_expert_parallel_size before model
    construction), routing decisions are computed locally on the replicated gate and
    then forwarded to ExpertParallelMoE for dispatch -> local experts -> combine.
    Otherwise falls back to a single-GPU per-expert loop.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

        # Gate is replicated: small enough ([hidden, num_experts]) that sharding
        # is not worthwhile, and every rank needs the full routing decision.
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        self.ep_size = get_expert_parallel_size()
        if self.ep_size > 1:
            ep_rank = dist.get_rank() if dist.is_initialized() else 0
            self.ep_moe = ExpertParallelMoE(
                hidden_size, intermediate_size, num_experts, self.ep_size, ep_rank)
            self.experts = None
        else:
            self.experts = nn.ModuleList([
                MoEExpert(hidden_size, intermediate_size)
                for _ in range(num_experts)
            ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        num_tokens, hidden_dim = hidden_states.shape

        router_logits = self.gate(hidden_states)

        # Qwen/Mixtral style: softmax over all experts before top-k, then re-norm.
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.ep_size > 1:
            out = self.ep_moe(hidden_states, selected_experts, routing_weights)
            return out.view(orig_shape)

        # Single-GPU path: iterate over experts, gather routed tokens, accumulate.
        final_hidden_states = torch.zeros(
            num_tokens, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            expert_output = self.experts[expert_idx](hidden_states[token_idx])
            weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, expert_output * weights)

        return final_hidden_states.view(orig_shape)
