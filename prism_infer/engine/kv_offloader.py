from collections import deque
import torch

# This class is used to offload KV cache blocks from GPU to CPU when GPU memory is insufficient.
class KVOffloader:

    def __init__(self, kv_cache: torch.Tensor, num_cpu_blocks: int):
        # kv_cache: ModelRunner.kv_cache, shpae = [2, L, num_blocks, block_size, kv_heads, head_dim]
        self.kv_cache = kv_cache
        # block shpae = [2, L, block_size, kv_heads, head_dim]
        block_shape = (kv_cache.shape[0], kv_cache.shape[1], *kv_cache.shape[3:])
        
        pin = kv_cache.is_cuda
        # CPU pool shape：[num_cpu_blocks, 2, L, block_size, kv_heads, head_dim]
        self.cpu_pool = torch.empty(
            (num_cpu_blocks, *block_shape), dtype=kv_cache.dtype, pin_memory=pin
        )
        self.free_slots: deque[int] = deque(range(num_cpu_blocks))
    
    def has_room(self) -> bool:
        return bool(self.free_slots)
    
    def copy_gpu_to_cpu(self, gpu_block_id: int) -> int:
        # offload the block from GPU to CPU, return the slot index in CPU pool
        slot = self.free_slots.popleft()
        self.cpu_pool[slot].copy_(self.kv_cache[:, :, gpu_block_id])
        return slot
    
    def copy_cpu_to_gpu(self, slot: int, gpu_block_id: int):
        # recall the block from CPU pool to GPU, copy to kv_cache[gpu_block_id]
        self.kv_cache[:, :, gpu_block_id].copy_(self.cpu_pool[slot])
        self.free_slots.append(slot)