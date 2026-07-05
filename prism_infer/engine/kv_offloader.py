from collections import deque
import torch


class KVOffloader:
    """LRU CPU offload pool for KV cache blocks.

    Evicted GPU blocks are copied to pinned CPU memory and recalled on
    prefix-cache hit, saving a full prefill re-computation.

    kv_cache shape: [2, num_layers, num_blocks, block_size, kv_heads, head_dim]
    cpu_pool shape: [num_cpu_blocks, 2, num_layers, block_size, kv_heads, head_dim]
    """

    def __init__(self, kv_cache: torch.Tensor, num_cpu_blocks: int):
        self.kv_cache = kv_cache
        block_shape = (kv_cache.shape[0], kv_cache.shape[1], *kv_cache.shape[3:])
        pin = kv_cache.is_cuda
        self.cpu_pool = torch.empty(
            (num_cpu_blocks, *block_shape), dtype=kv_cache.dtype, pin_memory=pin
        )
        self.free_slots: deque[int] = deque(range(num_cpu_blocks))

    def has_room(self) -> bool:
        return bool(self.free_slots)

    def copy_gpu_to_cpu(self, gpu_block_id: int) -> int:
        """Copy one block to the CPU pool. Returns the CPU slot index."""
        slot = self.free_slots.popleft()
        self.cpu_pool[slot].copy_(self.kv_cache[:, :, gpu_block_id])
        return slot

    def copy_cpu_to_gpu(self, slot: int, gpu_block_id: int):
        """Copy one CPU slot back to a GPU block and free the slot."""
        self.kv_cache[:, :, gpu_block_id].copy_(self.cpu_pool[slot])
        self.free_slots.append(slot)
