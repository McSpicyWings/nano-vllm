#!/usr/bin/env python3
"""Matching vLLM baseline/EAGLE-3 benchmark for the resume workload."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.workload import build_fixed_length_prompts, workload_digest


DEFAULT_TARGET_MODEL = "./huggingface/Qwen3-1.7B"
DEFAULT_DRAFT_MODEL = "./huggingface/AngelSlim/Qwen3-1.7B_eagle3"
SPEC_METRIC_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_METRIC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_METRIC_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"


@dataclass
class RunResult:
    elapsed_s: float
    total_prompt_tokens: int
    total_output_tokens: int
    output_tokens_per_s: float
    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms_mean: float
    tpot_ms_p50: float
    tpot_ms_p95: float
    e2e_ms_mean: float
    acceptance_rate: float | None
    mean_acceptance_length: float | None
    mean_accepted_draft_length: float | None
    output_sha256: str


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = fraction * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_spec_metrics(metrics: list[Any]) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for metric in metrics:
        name = getattr(metric, "name", None)
        if name in {
            SPEC_METRIC_DRAFTS,
            SPEC_METRIC_DRAFT_TOKENS,
            SPEC_METRIC_ACCEPTED_TOKENS,
        }:
            result[name] += float(getattr(metric, "value", 0.0))
    return dict(result)


def diff_spec_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in {
            SPEC_METRIC_DRAFTS,
            SPEC_METRIC_DRAFT_TOKENS,
            SPEC_METRIC_ACCEPTED_TOKENS,
        }
    }


def aggregate_runs(runs: list[RunResult]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for field in [
        "elapsed_s",
        "output_tokens_per_s",
        "ttft_ms_mean",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "tpot_ms_mean",
        "tpot_ms_p50",
        "tpot_ms_p95",
        "e2e_ms_mean",
        "acceptance_rate",
        "mean_acceptance_length",
        "mean_accepted_draft_length",
    ]:
        values = [getattr(run, field) for run in runs]
        present = [float(value) for value in values if value is not None]
        aggregate[field] = (
            None
            if not present
            else {
                "mean": statistics.fmean(present),
                "std": statistics.pstdev(present),
            }
        )
    aggregate["all_output_digests_equal"] = len({run.output_sha256 for run in runs}) == 1
    return aggregate


class DeviceMemoryMonitor:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.start_used_bytes: int | None = None
        self.peak_used_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml = None
        self._handle = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            used = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            self.start_used_bytes = used
            self.peak_used_bytes = used
        except Exception:
            return

        def poll() -> None:
            assert self._pynvml is not None and self._handle is not None
            while not self._stop.wait(0.05):
                used = int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                self.peak_used_bytes = max(self.peak_used_bytes or 0, used)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()


def _run_case_worker(
    result_queue,
    *,
    case_name: str,
    target_model: str,
    draft_model: str,
    prompt_token_ids: list[list[int]],
    output_length: int,
    repeats: int,
    warmup_runs: int,
    num_spec_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    seed: int,
) -> None:
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        speculative_config = None
        if case_name == "vllm_eagle3":
            speculative_config = {
                "method": "eagle3",
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
            }
        llm = LLM(
            model=target_model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            disable_log_stats=False,
            speculative_config=speculative_config,
        )
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            seed=seed,
            ignore_eos=True,
            detokenize=False,
            max_tokens=output_length,
        )
        for _ in range(warmup_runs):
            llm.generate(prompts, sampling_params, use_tqdm=False)
        before = summarize_spec_metrics(llm.get_metrics())

        runs = []
        for _ in range(repeats):
            start = perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            elapsed_s = perf_counter() - start
            after = summarize_spec_metrics(llm.get_metrics())
            spec_delta = diff_spec_metrics(before, after)
            before = after

            output_ids = [list(output.outputs[0].token_ids) for output in outputs]
            metrics = [output.metrics for output in outputs]
            if any(metric is None for metric in metrics):
                raise RuntimeError("vLLM request metrics are unavailable")
            ttft = [float(metric.first_token_latency) * 1000 for metric in metrics]
            tpot = [
                (
                    (float(metric.last_token_ts) - float(metric.first_token_ts))
                    * 1000
                    / (len(ids) - 1)
                    if len(ids) > 1 else 0.0
                )
                for metric, ids in zip(metrics, output_ids)
            ]
            e2e = [
                (float(metric.last_token_ts) - float(metric.queued_ts)) * 1000
                for metric in metrics
            ]
            total_output_tokens = sum(len(ids) for ids in output_ids)
            num_drafts = spec_delta[SPEC_METRIC_DRAFTS]
            draft_tokens = spec_delta[SPEC_METRIC_DRAFT_TOKENS]
            accepted_tokens = spec_delta[SPEC_METRIC_ACCEPTED_TOKENS]
            spec_enabled = num_drafts > 0
            runs.append(
                RunResult(
                    elapsed_s=elapsed_s,
                    total_prompt_tokens=sum(len(ids) for ids in prompt_token_ids),
                    total_output_tokens=total_output_tokens,
                    output_tokens_per_s=total_output_tokens / elapsed_s,
                    ttft_ms_mean=statistics.fmean(ttft),
                    ttft_ms_p50=percentile(ttft, 0.50),
                    ttft_ms_p95=percentile(ttft, 0.95),
                    tpot_ms_mean=statistics.fmean(tpot),
                    tpot_ms_p50=percentile(tpot, 0.50),
                    tpot_ms_p95=percentile(tpot, 0.95),
                    e2e_ms_mean=statistics.fmean(e2e),
                    acceptance_rate=(
                        accepted_tokens / draft_tokens
                        if spec_enabled and draft_tokens > 0 else None
                    ),
                    mean_acceptance_length=(
                        1.0 + accepted_tokens / num_drafts if spec_enabled else None
                    ),
                    mean_accepted_draft_length=(
                        accepted_tokens / num_drafts if spec_enabled else None
                    ),
                    output_sha256=workload_digest(output_ids),
                )
            )

        result_queue.put(
            {
                "case_name": case_name,
                "runs": [asdict(run) for run in runs],
                "aggregate": aggregate_runs(runs),
                "error": None,
            }
        )
        del llm
        gc.collect()
    except Exception:
        result_queue.put(
            {
                "case_name": case_name,
                "runs": [],
                "aggregate": {},
                "error": traceback.format_exc(),
            }
        )


def run_case(**kwargs) -> dict[str, Any]:
    monitor = DeviceMemoryMonitor()
    monitor.start()
    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_run_case_worker,
        kwargs={"result_queue": queue, **kwargs},
    )
    process.start()
    process.join()
    monitor.stop()
    if queue.empty():
        result = {
            "case_name": kwargs["case_name"],
            "runs": [],
            "aggregate": {},
            "error": f"worker exited with code {process.exitcode} without a result",
        }
    else:
        result = queue.get()
    result["device_memory_start_gib"] = (
        None if monitor.start_used_bytes is None else monitor.start_used_bytes / 2**30
    )
    result["device_memory_peak_gib"] = (
        None if monitor.peak_used_bytes is None else monitor.peak_used_bytes / 2**30
    )
    return result


def render_markdown(result: dict[str, Any]) -> str:
    config = result["config"]
    lines = [
        "# vLLM EAGLE-3 Benchmark",
        "",
        (
            f"- workload: batch={config['batch_size']}, input={config['input_length']}, "
            f"output={config['output_length']}, greedy, repeats={config['repeats']}"
        ),
        f"- workload SHA256: `{config['workload_sha256']}`",
        "",
        "| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Device peak (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, case in result["cases"].items():
        if case["error"]:
            lines.append(f"| {case_name} | ERROR | - | - | - | - | - |")
            continue
        aggregate = case["aggregate"]
        def mean(name: str) -> float | None:
            value = aggregate.get(name)
            return None if value is None else float(value["mean"])
        throughput = mean("output_tokens_per_s")
        ttft = mean("ttft_ms_mean")
        tpot = mean("tpot_ms_mean")
        acceptance = mean("acceptance_rate")
        accept_length = mean("mean_acceptance_length")
        peak = case["device_memory_peak_gib"]
        lines.append(
            f"| {case_name} | {throughput:.2f} | {ttft:.2f} | {tpot:.2f} | "
            f"{'-' if acceptance is None else f'{acceptance:.2%}'} | "
            f"{'-' if accept_length is None else f'{accept_length:.3f}'} | "
            f"{'-' if peak is None else f'{peak:.2f}'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["vllm_baseline", "vllm_eagle3"],
        default=["vllm_baseline", "vllm_eagle3"],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("Set PYTHONNOUSERSITE=1 for an isolated benchmark environment.")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    prompt_token_ids = build_fixed_length_prompts(
        tokenizer, args.batch_size, args.input_length
    )
    total_memory = torch.cuda.get_device_properties(0).total_memory
    result = {
        "engine": "vllm",
        "environment": {
            "vllm": version("vllm"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_gib": total_memory / 2**30,
            "python_executable": sys.executable,
            "python_no_user_site": True,
        },
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "batch_size": args.batch_size,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "ignore_eos": True,
            "seed": args.seed,
            "num_spec_tokens": args.num_spec_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "workload_sha256": workload_digest(prompt_token_ids),
        },
        "cases": {},
    }
    common = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "prompt_token_ids": prompt_token_ids,
        "output_length": args.output_length,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "num_spec_tokens": args.num_spec_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    for case_name in args.cases:
        print(f"Running {case_name} ...", flush=True)
        case = run_case(case_name=case_name, **common)
        result["cases"][case_name] = case
        if case["error"]:
            print(case["error"], flush=True)
            raise SystemExit(1)
        throughput = case["aggregate"]["output_tokens_per_s"]["mean"]
        print(f"{case_name}: {throughput:.2f} output tok/s", flush=True)

    markdown = render_markdown(result)
    print(markdown)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown)


if __name__ == "__main__":
    main()
