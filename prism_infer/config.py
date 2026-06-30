import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    cpu_offload_blocks: int = 0
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # multi-GPU: generated once by LLMEngine and passed to all ranks via Config.
    # 0 / "" mean "auto": a free TCP port and a unique shm name (PID/UUID).
    # Replaces hardcoded port 2333 / shm name "prism_infer" which caused
    # EADDRINUSE / FileExistsError on multi-GPU re-runs.
    master_port: int = 0
    shm_name: str = ""
    # Internal flag: TP workers use the shm+Event loop for rank>0 coordination.
    # EP workers (launched via torchrun) do not — each rank runs its own forward
    # and exits normally. Set to False when constructing Config for EP parity scripts.
    _use_shm_worker_loop: bool = True

    @property
    def world_size(self) -> int:
        # EP and TP are mutually exclusive; world_size is whichever is active.
        return max(self.tensor_parallel_size, self.expert_parallel_size)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert not (self.tensor_parallel_size > 1 and self.expert_parallel_size > 1), \
            "TP and EP are mutually exclusive"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
