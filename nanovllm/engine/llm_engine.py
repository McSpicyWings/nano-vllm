import atexit
import gc
import torch
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        self._request_metrics = {}
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # tokenizer_path = kwargs.get("tokenizer", config.model)
        # try:
        #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        # except Exception:
        #     # Fall back to slow tokenizer if fast conversion is unsupported (common for eagle3 drafts without tiktoken files).
        #     self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._exited = False
        self._atexit_registered = True
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        if self._atexit_registered:
            try:
                atexit.unregister(self.exit)
            except Exception:
                pass
            self._atexit_registered = False
        try:
            if hasattr(self, "model_runner") and self.model_runner is not None:
                self.model_runner.call("exit")
        finally:
            if hasattr(self, "model_runner"):
                del self.model_runner
            for p in self.ps:
                p.join()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        arrival_time: float | None = None,
    ):
        if self.config.draft_model is not None and sampling_params.temperature > 0:
            raise ValueError(
                "EAGLE-3 speculative decoding currently supports greedy sampling only "
                "(temperature=0); rejection sampling is not implemented."
            )
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self._request_metrics[seq.seq_id] = {
            "arrival_time": perf_counter() if arrival_time is None else arrival_time,
            "first_token_time": None,
            "finished_time": None,
        }
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        completion_counts = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        step_end = perf_counter()
        finished_seq_ids = []
        for seq in seqs:
            metrics = self._request_metrics.get(seq.seq_id)
            if metrics is not None:
                if (
                    completion_counts[seq.seq_id] == 0
                    and seq.num_completion_tokens > 0
                    and metrics["first_token_time"] is None
                ):
                    metrics["first_token_time"] = step_end
                if seq.is_finished:
                    metrics["finished_time"] = step_end
            if seq.is_finished:
                finished_seq_ids.append(seq.seq_id)
        if finished_seq_ids:
            self.model_runner.call("release_sequences", finished_seq_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # Calculate num_tokens for throughput tracking
        if is_prefill:
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            # For decode, count actual tokens generated (may be multiple with spec decode)
            if token_ids and isinstance(token_ids[0], list):
                num_tokens = -sum(len(toks) for toks in token_ids)
            else:
                num_tokens = -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        return_metrics: bool = False,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")
        batch_start = perf_counter()
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp, arrival_time=batch_start)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                metrics = self._request_metrics.pop(seq_id)
                outputs[seq_id] = (token_ids, metrics)
                if use_tqdm:
                    pbar.update(1)
        ordered_outputs = []
        for seq_id in sorted(outputs.keys()):
            token_ids, metrics = outputs[seq_id]
            item = {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            if return_metrics:
                arrival_time = metrics["arrival_time"]
                first_token_time = metrics["first_token_time"]
                finished_time = metrics["finished_time"]
                assert first_token_time is not None and finished_time is not None
                item["metrics"] = {
                    "ttft_ms": (first_token_time - arrival_time) * 1000,
                    "tpot_ms": (
                        (finished_time - first_token_time) * 1000 / (len(token_ids) - 1)
                        if len(token_ids) > 1 else 0.0
                    ),
                    "e2e_ms": (finished_time - arrival_time) * 1000,
                }
            ordered_outputs.append(item)
        if use_tqdm:
            pbar.close()
        return ordered_outputs

    def get_spec_stats(self, reset: bool = False) -> dict:
        return self.model_runner.call("get_spec_stats", reset)
