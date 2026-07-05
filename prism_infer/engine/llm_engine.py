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
        if config.tensor_parallel_size > 1 and not config.shm_name:
            config.shm_name = f"prism_infer_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # CPU offload is only safe on single-GPU (TP=1, EP=1): multi-GPU EP has no
        # mechanism to synchronise per-rank offload state across ranks.
        if (config.cpu_offload_blocks > 0
                and config.tensor_parallel_size == 1
                and config.expert_parallel_size == 1):
            from prism_infer.engine.kv_offloader import KVOffloader
            self.scheduler.block_manager.offloader = KVOffloader(
                self.model_runner.kv_cache, config.cpu_offload_blocks)
        atexit.register(self.exit)
        # KVConnector: single choke-point for PD-role behaviour.
        # unified: no-op;
        # prefill-only: fires KVBlockPusher;
        # decode-only: polls KVReceiver.
        from prism_infer.engine.kv_connector import _build_connector
        kv_cache = getattr(self.model_runner, "kv_cache", None)
        self.kv_connector = _build_connector(config, kv_cache=kv_cache)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # The sole running seq may have self-preempted under extreme memory pressure;
        # skip the model call and return empty for this step.
        if not seqs:
            return [], 0
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = []
        for seq in seqs:
            if is_prefill and seq.num_cached_tokens == seq.num_tokens:
                self.kv_connector.on_prefill_done(seq)
            if seq.is_finished:
                outputs.append((seq.seq_id, seq.completion_token_ids))
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

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
            finished, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in finished:
                by_seq_id[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        token_ids_list = [by_seq_id[seq_id] for seq_id in sorted(by_seq_id)]
        return [{"text": self.tokenizer.decode(ids), "token_ids": ids} for ids in token_ids_list]
