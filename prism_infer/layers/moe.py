import torch
from torch import nn
import torch.nn.functional as F

from prism_infer.layers.activation import SiluAndMul
from prism_infer.layers.linear import (
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


class MoEExpert(nn.Module):
    # A single expert: a SwiGLU FFN.

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()  # silu(gate) * up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class MoE(nn.Module):
    # Generic token-choice MoE: softmax -> top-k -> (re-norm) -> dispatch -> combine.
    # Routing follows Qwen/Mixtral-style MoE; model files map their config to these args.

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

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        router_logits = self.gate(hidden_states)
        # softmax over ALL experts BEFORE top-k (Qwen/Mixtral style); 
        # this is more expensive but leads to better routing and is necessary for proper re-norm if top-k < num_experts
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            # re-norm top-k weights to sum to 1
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # [T, top_k, num_experts] -> [num_experts, top_k, T] for easier indexing; 
        # we will iterate over experts and gather tokens for each expert.
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            # Get the tokens assigned to this expert;
            # token_idx are the indices of tokens assigned to this expert;
            # slot_idx are the corresponding top-k slot indices (0 to top_k-1) for these tokens.
            slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            expert_output = self.experts[expert_idx](current_state)
            weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, expert_output * weights)

        return final_hidden_states.view(orig_shape)