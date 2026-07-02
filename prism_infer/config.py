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
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    cpu_offload_blocks: int = 0     # CPU offload pool size; 0 = LRU-only (no offload)
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    # Generated once by LLMEngine before spawning workers so all ranks share
    # the same port / shm name (0 / "" = auto-assign).
    master_port: int = 0
    shm_name: str = ""
    # False for EP standalone (torchrun); True for TP shm+Event worker loop.
    _use_shm_worker_loop: bool = True

    engine_mode: str = "unified"
    # "unified"       default, prefill+decode in one engine (zero regression)
    # "prefill-only"  run prefill, push KV via KVBlockPusher, no decode
    # "decode-only"   wait for KV transfer then run decode
    kv_transfer_backend: str = "auto"
    # "nccl" | "ipc" | "auto" (auto: same-host-> ipc, cross-host -> nccl)
    pd_decode_addr: str = ""           # decode instance address for prefill-only mode
    pd_master_port: int = 29500        # process-group init port for PD pair
    max_bytes_inflight: int = 256 * 1024 * 1024   # global in-flight cap
    max_blocks_per_peer: int = 64      # per-dst in-flight block cap
    zerocopy_weight_load: bool = False # use mmap + async H2D for weight loading
    instance_id: str = ""              # unique instance id (auto UUID if empty)

    @property
    def world_size(self) -> int:
        return max(self.tensor_parallel_size, self.expert_parallel_size)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert not (self.tensor_parallel_size > 1 and self.expert_parallel_size > 1), \
            "TP and EP are mutually exclusive"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
