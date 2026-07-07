from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    """Manages all NCCL process group lifetimes for a prism-infer instance.

    Three group types (example: TP=4, 4P + 4D):
      TP groups x2:  [P0,P1,P2,P3], [D0,D1,D2,D3]
      PD groups x4:  [P0,D0], [P1,D1], [P2,D2], [P3,D3]

    All groups are created once at service startup and reused for the full
    lifetime -- NCCL group init is expensive (ring/tree topology negotiation,
    hundreds of ms) and must not be repeated per-request.

    pd_groups is a list rather than a single field because each TP shard has
    its own dedicated PD group. TP=1 degenerates to a list of length 1.
    """

    world_size: int
    rank: int
    tp_size: int = 1
    ep_size: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500

    tp_group: Optional[dist.ProcessGroup] = field(default=None, repr=False)
    # pd_groups[i]: P2P group between prefill rank i and decode rank i
    pd_groups: list = field(default_factory=list, repr=False)
    pd_peer_rank: int = -1  # -1 = unified mode (no PD peer)

    def init(self) -> None:
        """Initialize the global process group and subgroups. Call once at startup."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                world_size=self.world_size,
                rank=self.rank,
            )
        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        if self.tp_size > 1:
            tp_ranks = list(range(self.tp_size))
            self.tp_group = dist.new_group(ranks=tp_ranks)

    def create_pd_groups(
        self,
        prefill_ranks: list[int],
        decode_ranks: list[int],
    ) -> None:
        """Create one PD P2P process group per TP shard.

        Must be called by ALL ranks (including non-members) -- NCCL new_group
        requires every rank in the parent group to participate in the call.
        Each rank passes the same arguments; NCCL decides internally whether
        the rank is a member.

        Args:
            prefill_ranks: global ranks of prefill GPUs, in TP-shard order
            decode_ranks:  global ranks of decode GPUs, same order as prefill_ranks

        After the call:
            pd_groups[i] is the P2P group for prefill_ranks[i] <-> decode_ranks[i]
            pd_peer_rank is set to this rank's peer global rank
        """
        assert len(prefill_ranks) == len(decode_ranks), (
            f"prefill_ranks and decode_ranks must have equal length, "
            f"got {len(prefill_ranks)} vs {len(decode_ranks)}"
        )
        self.pd_groups = []
        for p_rank, d_rank in zip(prefill_ranks, decode_ranks):
            group = dist.new_group(ranks=[p_rank, d_rank], backend="nccl")
            self.pd_groups.append(group)
            if self.rank == p_rank:
                self.pd_peer_rank = d_rank
            elif self.rank == d_rank:
                self.pd_peer_rank = p_rank

    def destroy(self) -> None:
        """Destroy all process groups. Call on service shutdown."""
        if dist.is_initialized():
            dist.destroy_process_group()


_CONTEXT: Optional[DistributedContext] = None


def init_distributed(
    world_size: int,
    rank: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
    tp_size: int = 1,
    ep_size: int = 1,
) -> DistributedContext:
    """Initialize the distributed context. Call once at engine startup."""
    global _CONTEXT
    _CONTEXT = DistributedContext(
        world_size=world_size,
        rank=rank,
        tp_size=tp_size,
        ep_size=ep_size,
        master_addr=master_addr,
        master_port=master_port,
    )
    _CONTEXT.init()
    return _CONTEXT


def get_context() -> DistributedContext:
    """Return the global DistributedContext singleton. Requires init_distributed() first."""
    assert _CONTEXT is not None, "call init_distributed() before get_context()"
    return _CONTEXT
