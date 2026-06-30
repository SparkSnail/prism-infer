import torch
from torch import nn
import torch.distributed as dist

from prism_infer.layers.activation import SiluAndMul

_EXPERT_PARALLEL_SIZE = 1


def set_expert_parallel_size(n: int) -> None:
    global _EXPERT_PARALLEL_SIZE
    _EXPERT_PARALLEL_SIZE = n


def get_expert_parallel_size() -> int:
    return _EXPERT_PARALLEL_SIZE


def expert_rank(expert_id: int, num_local_experts: int) -> int:
    return expert_id // num_local_experts


def build_dispatch_plan(topk_ids: torch.Tensor, num_local_experts: int, ep_size: int):
    """Build permutation and split sizes for the dispatch all-to-all.

    Returns:
        perm:             [T*k] indices that reorder routing entries from
                          token-interleaved to dst-rank-grouped order.
                          Apply as sorted_x = flat_x[perm] on dispatch and
                          unperm[perm] = recv_out on combine.
        input_splits:     [ep_size] rows sent to each rank.
        sorted_local_eid: [T*k] per-entry local expert index (0..num_local-1),
                          reordered by perm to stay aligned with sent hidden vectors.
    """
    flat_expert = topk_ids.reshape(-1)
    dst_rank = torch.div(flat_expert, num_local_experts, rounding_mode="floor")
    # stable=True preserves relative order within a rank, making un-permute straightforward
    perm = torch.argsort(dst_rank, stable=True)
    input_splits = torch.bincount(dst_rank, minlength=ep_size)
    local_eid = flat_expert % num_local_experts
    sorted_local_eid = local_eid[perm].to(torch.int64)
    return perm, input_splits, sorted_local_eid


def combine_weighted_sum(unpermuted: torch.Tensor, topk_weights: torch.Tensor,
                         num_tokens: int, top_k: int) -> torch.Tensor:
    """Weighted sum over k expert outputs per token.

    Args:
        unpermuted:   [T*k, H] expert outputs in routing order.
        topk_weights: [T, k]   router weights (pre-normalised).
    Returns:
        [T, H]  out[t] = sum_k( w[t,k] * expert_out[t,k] )
    """
    hidden = unpermuted.shape[-1]
    grouped = unpermuted.view(num_tokens, top_k, hidden)                 # [T, k, H]
    weights = topk_weights.to(unpermuted.dtype).unsqueeze(-1)            # [T, k, 1]
    return (grouped * weights).sum(dim=1)                                # [T, H]


class _PlainExpert(nn.Module):
    """Full SwiGLU expert for the EP path, without tensor-parallel sharding.

    Uses plain nn.Linear instead of the TP-aware variant because under EP
    world_size equals ep_size, which would otherwise incorrectly shard each
    expert across GPUs.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

        # The checkpoint stores gate_proj and up_proj as separate [I, H] tensors
        # (shard_id 0 and 1). Write each into the corresponding half of the fused
        # [2I, H] parameter. Without this, the up_proj load (shard_id=1) would
        # overwrite the entire parameter, silently corrupting gate_proj.
        def _gate_up_weight_loader(param, loaded_weight, shard_id):
            half = param.data.shape[0] // 2
            if shard_id == 0:
                param.data[:half].copy_(loaded_weight)
            else:
                param.data[half:].copy_(loaded_weight)

        self.gate_up_proj.weight.weight_loader = _gate_up_weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class ExpertParallelMoE(nn.Module):
    """MoE layer with expert parallelism: experts sharded by rank, tokens routed
    via all-to-all (dispatch -> local experts -> combine).

    Routing decisions (topk_ids / topk_weights) are computed by the replicated
    router in layers/moe.MoE and passed in; only the local experts for this rank
    are instantiated.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int,
                 ep_size: int, ep_rank: int):
        super().__init__()
        assert num_experts % ep_size == 0, "num_experts must be divisible by ep_size"
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_local_experts = num_experts // ep_size
        self.local_expert_base = ep_rank * self.num_local_experts
        self.local_experts = nn.ModuleList([
            _PlainExpert(hidden_size, intermediate_size)
            for _ in range(self.num_local_experts)
        ])

    def _exchange_counts(self, input_splits: list[int], device) -> list[int]:
        """Transpose the send-count table to learn how many rows to receive from each rank.

        Variable-size all_to_all requires output_splits to be known before the data
        transfer. A single fixed-size all_to_all over the [ep_size] count vectors
        achieves this: recv[i] = rank_i.input_splits[this_rank].
        """
        if self.ep_size == 1:
            return list(input_splits)
        send = torch.tensor(input_splits, dtype=torch.int64, device=device)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        return recv.tolist()

    def _all_to_all(self, x: torch.Tensor, out_splits: list[int], in_splits: list[int]) -> torch.Tensor:
        """Variable-size all-to-all along dim0."""
        if self.ep_size == 1:
            return x.clone()
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        # contiguous() required: perm-indexed tensors may not be contiguous
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits)
        return out

    def forward(self, x: torch.Tensor, topk_ids: torch.Tensor,
                topk_weights: torch.Tensor) -> torch.Tensor:
        """Run one EP MoE layer.

        Args:
            x:            [T, H] token hidden states on this GPU.
            topk_ids:     [T, k] global expert ids per token.
            topk_weights: [T, k] router weights.
        Returns:
            [T, H]
        """
        num_tokens, top_k = topk_ids.shape
        device = x.device

        # Expand each token to k routing entries, one per selected expert
        flat_x = x.repeat_interleave(top_k, dim=0)                      # [T*k, H]
        perm, input_splits, sorted_local_eid = build_dispatch_plan(
            topk_ids, self.num_local_experts, self.ep_size)
        sorted_x = flat_x[perm]
        input_splits_list  = input_splits.tolist()
        output_splits_list = self._exchange_counts(input_splits_list, device)

        # Dispatch: send hidden vectors and local expert ids to owner GPUs
        recv_x   = self._all_to_all(sorted_x,         output_splits_list, input_splits_list)  # [R, H]
        recv_eid = self._all_to_all(sorted_local_eid, output_splits_list, input_splits_list)  # [R]

        # Local expert compute: batch tokens by expert for efficiency
        out = torch.zeros_like(recv_x)
        for le in range(self.num_local_experts):
            sel = recv_eid == le
            if torch.any(sel):
                out[sel] = self.local_experts[le](recv_x[sel])

        # Combine: return results to home ranks (swap splits = exact reverse of dispatch)
        recv_out = self._all_to_all(out, input_splits_list, output_splits_list)
        unperm = torch.empty_like(recv_out)
        unperm[perm] = recv_out                                          # scatter-assign inverts perm

        return combine_weighted_sum(unperm, topk_weights, num_tokens, top_k)
