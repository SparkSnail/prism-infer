import atexit
import os
import socket
import uuid
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from prism_infer.config import Config
from prism_infer.sampling_params import SamplingParams
from prism_infer.engine.sequence import Sequence
from prism_infer.engine.scheduler import Scheduler
from prism_infer.engine.model_runner import ModelRunner


def _find_free_port() -> int:
    # Bind to port 0 so the OS picks a free port; release it and reuse the number.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class LLMEngine:

    def __init__(self, model, **kwargs):
        # model may be a path string or a pre-built Config object.
        # pd_runner and test_parity_pd pass a Config directly so the process group
        # they created before calling LLMEngine is preserved.
        if isinstance(model, Config):
            config = model
        else:
            config_fields = {field.name for field in fields(Config)}
            config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
            config = Config(model, **config_kwargs)

        Sequence.block_size = config.kvcache_block_size

        # Generate a unique NCCL port + shm name once, before spawning workers,
        # so rank 0 and all workers (which receive Config via pickle) agree.
        # The port is always needed: even TP=1 calls init_process_group(world_size=1),
        # and port 0 makes torch raise "port number missing". The shm name is only
        # used for TP>1 worker communication.
        if config.master_port == 0:
            config.master_port = _find_free_port()

        if not config._use_shm_worker_loop:
            # EP standalone mode: launched by torchrun; all ranks are peers, no spawn.
            ep_rank = int(os.environ.get("RANK", "0"))
            self.ps = []
            self.events = []
            self.model_runner = ModelRunner(config, ep_rank, [])
        else:
            if config.world_size > 1 and not config.shm_name:
                config.shm_name = f"prism_infer_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            self.ps = []
            self.events = []
            ctx = mp.get_context("spawn")
            for i in range(1, config.world_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        instance_id = config.instance_id or f"infer-{os.getpid()}"
        self.scheduler.block_manager.instance_id = instance_id
        self.scheduler.block_manager.instance_epoch = uuid.uuid4().hex
        from prism_infer.engine.prefix_cache import PrefixCacheService
        self.prefix_cache = PrefixCacheService(
            self.scheduler.block_manager,
            getattr(self.model_runner, "kv_cache", None),
        )

        # CPU offload is only safe on single-GPU (TP=1, EP=1): multi-GPU EP has no
        # mechanism to synchronise per-rank offload state across ranks.
        if (config.cpu_offload_blocks > 0
                and config.tensor_parallel_size == 1
                and config.expert_parallel_size == 1):
            from prism_infer.engine.kv_offloader import KVOffloader
            self.scheduler.block_manager.offloader = KVOffloader(
                self.model_runner.kv_cache, config.cpu_offload_blocks)

        atexit.register(self._ep_exit if not config._use_shm_worker_loop else self.exit)

        # KVConnector: single choke-point for PD-role behaviour.
        # unified: no-op; prefill-only: fires KVBlockPusher; decode-only: polls KVReceiver.
        from prism_infer.engine.kv_connector import _build_connector
        kv_cache = getattr(self.model_runner, "kv_cache", None)
        self.kv_connector = _build_connector(
            config,
            kv_cache=kv_cache,
            block_manager=self.scheduler.block_manager,
        )
        self.scheduler._kv_ready_fn = self.kv_connector.on_before_decode

        # Remote migration fencing and supervisor-owned reclamation are not
        # provided by this local helper state.
        from prism_infer.engine.kv_snapshot import MigrationWatchdog
        self._migration_watchdog = MigrationWatchdog(self.scheduler.block_manager)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def _ep_exit(self):
        self.model_runner.exit()
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # Pending sends outlive their source Sequence and still need progress.
        self.kv_connector.poll()
        seqs, is_prefill = self.scheduler.schedule()
        # The sole running seq may have self-preempted under extreme memory pressure;
        # skip the model call and return empty for this step.
        if not seqs:
            return [], 0

        # Positive = prefill token count; negative = decode batch size.
        # Callers use the sign to distinguish the two phases.
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        token_ids = (
            self.model_runner.run(seqs, is_prefill)
            if not self.model_runner.config._use_shm_worker_loop
            else self.model_runner.call("run", seqs, is_prefill)
        )
        # EP standalone: run() returns None on rank > 0; skip scheduler update.
        if token_ids is None:
            return [], num_tokens

        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        outputs = []
        for seq in seqs:
            if is_prefill and seq.num_cached_tokens == seq.num_prompt_tokens:
                self.kv_connector.on_prefill_done(seq)
            if seq.is_finished:
                outputs.append((seq.seq_id, seq.completion_token_ids))

        self.kv_connector.poll()

        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished() and not self.kv_connector.has_pending()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        by_seq_id: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                by_seq_id[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        outputs = [by_seq_id[seq_id] for seq_id in sorted(by_seq_id)]
        return [{"text": self.tokenizer.decode(ids), "token_ids": ids} for ids in outputs]

    def handle_migrate_req(self, req) -> dict:
        """Create a local migration snapshot.

        This helper does not provide remote RPC, DATA_READY fencing, retries,
        or supervisor ownership.
        """
        from prism_infer.engine.kv_snapshot import (
            snapshot_sequence, incremental_snapshot,
        )
        from prism_infer.engine.sequence import SequenceStatus

        seq_id = int(req.seq_id)
        seq = None
        for s in list(self.scheduler.running) + list(self.scheduler.waiting):
            if s.seq_id == seq_id:
                seq = s
                break
        if seq is None:
            return {"success": False, "error": f"seq_id={seq_id} not found"}
        if seq.is_finished:
            return {"success": False, "error": f"seq_id={seq_id} already finished (ABORTED_SRC)"}

        seq.status = SequenceStatus.MIGRATING_OUT
        try:
            allow_unaligned = (req.mode == "unaligned")
            if req.incremental and req.base_blocks:
                handle = incremental_snapshot(seq, req.base_blocks)
            else:
                handle = snapshot_sequence(seq, allow_unaligned=allow_unaligned)
            return {"success": True, "handle": handle}
        except Exception as e:
            seq.status = SequenceStatus.RUNNING
            return {"success": False, "error": str(e)}

    def pre_alloc_blocks_for_migration(
        self,
        seq_id: str,
        block_num: int,
        token_ids: list,
    ) -> dict:
        """Pre-allocate blocks for the local migration helper."""
        from prism_infer.engine.kv_snapshot import pre_alloc_blocks
        dst_blocks = pre_alloc_blocks(
            seq_id, block_num, token_ids,
            self.scheduler.block_manager,
        )
        if dst_blocks is None:
            return {"success": False, "error": "dst OOM (ABORTED_DST)"}
        self._migration_watchdog.register(seq_id, dst_blocks)
        return {"success": True, "dst_blocks": dst_blocks}

    def commit_migration_for_seq(self, seq_id: str, handle, sampling_params) -> dict:
        """Activate a locally restored sequence after the caller proves KV readiness.

        This helper does not establish a remote receive-completion fence.
        """
        from prism_infer.engine.kv_snapshot import apply_snapshot, commit_migration
        from prism_infer.engine.sequence import SequenceStatus

        try:
            dst_blocks = self._migration_watchdog.pending_blocks(seq_id)
            if dst_blocks is None:
                return {
                    "success": False,
                    "error": f"seq_id={seq_id} has no pre-allocated blocks",
                }
            seq = apply_snapshot(
                handle,
                self.scheduler.block_manager,
                sampling_params,
                dst_blocks=dst_blocks,
            )
            commit_migration(seq_id, handle, seq)
            self._migration_watchdog.commit(seq_id)
            if seq.status == SequenceStatus.RUNNING:
                self.scheduler.running.append(seq)
            else:
                self.scheduler.waiting.append(seq)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reset_to_waiting_for_seq(self, seq_id: str) -> dict:
        """Revert a KV_TRANSFERRING sequence to WAITING for recompute fallback.

        Called by serve when KV transfer times out and on_fail="recompute".
        """
        from prism_infer.engine.kv_snapshot import reset_to_waiting
        seq_id_int = int(seq_id)
        for s in list(self.scheduler.running) + list(self.scheduler.waiting):
            if s.seq_id == seq_id_int:
                reset_to_waiting(s)
                return {"success": True}
        return {"success": False, "error": f"seq_id={seq_id} not found"}
