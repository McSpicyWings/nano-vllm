#!/usr/bin/env python3
"""Reproducible nano-vLLM baseline/linear/tree EAGLE-3 benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers
from transformers import AutoTokenizer

from benchmarks.workload import (
    DEFAULT_TREE,
    build_fixed_length_prompts,
    workload_digest,
)
from nanovllm import LLM, SamplingParams


DEFAULT_TARGET_MODEL = "./huggingface/Qwen3-1.7B"
DEFAULT_DRAFT_MODEL = "./huggingface/AngelSlim/Qwen3-1.7B_eagle3"


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
    sequence_proposals: int | None
    proposed_tokens: int | None
    accepted_draft_tokens: int | None
    emitted_tokens: int | None
    phase_time_ms_per_step: dict[str, float] | None
    output_sha256: str


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_outputs(
    outputs: list[dict[str, Any]],
    prompt_token_ids: list[list[int]],
    elapsed_s: float,
    spec_stats: dict[str, Any],
) -> RunResult:
    ttft = [float(output["metrics"]["ttft_ms"]) for output in outputs]
    tpot = [float(output["metrics"]["tpot_ms"]) for output in outputs]
    e2e = [float(output["metrics"]["e2e_ms"]) for output in outputs]
    output_ids = [output["token_ids"] for output in outputs]
    total_output_tokens = sum(len(ids) for ids in output_ids)
    spec_enabled = spec_stats.get("sequence_proposals", 0) > 0
    spec_calls = int(spec_stats.get("spec_calls", 0))
    phase_time_ms_per_step = None
    if spec_stats.get("timing_enabled") and spec_calls > 0:
        phase_time_ms_per_step = {
            phase: float(spec_stats[f"{phase}_time"]) * 1000 / spec_calls
            for phase in (
                "seed",
                "draft",
                "verify",
                "verify_model",
                "verify_logits",
                "verify_accept",
                "accept",
            )
        }
    return RunResult(
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
        acceptance_rate=float(spec_stats["acceptance_rate"]) if spec_enabled else None,
        mean_acceptance_length=(
            float(spec_stats["mean_acceptance_length"]) if spec_enabled else None
        ),
        mean_accepted_draft_length=(
            float(spec_stats["mean_accepted_draft_length"]) if spec_enabled else None
        ),
        sequence_proposals=(
            int(spec_stats["sequence_proposals"]) if spec_enabled else None
        ),
        proposed_tokens=int(spec_stats["proposed_tokens"]) if spec_enabled else None,
        accepted_draft_tokens=(
            int(spec_stats["accepted_draft_tokens"]) if spec_enabled else None
        ),
        emitted_tokens=int(spec_stats["emitted_tokens"]) if spec_enabled else None,
        phase_time_ms_per_step=phase_time_ms_per_step,
        output_sha256=workload_digest(output_ids),
    )


def aggregate_runs(runs: list[RunResult]) -> dict[str, Any]:
    numeric_fields = [
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
    ]
    aggregate: dict[str, Any] = {}
    for field in numeric_fields:
        values = [getattr(run, field) for run in runs]
        present = [float(value) for value in values if value is not None]
        if not present:
            aggregate[field] = None
            continue
        aggregate[field] = {
            "mean": statistics.fmean(present),
            "std": statistics.pstdev(present),
        }
    aggregate["all_output_digests_equal"] = len({run.output_sha256 for run in runs}) == 1
    return aggregate


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
    spec_verifier_mode: str,
    speculative_token_tree: str,
    max_model_len: int,
    num_kvcache_blocks: int,
    enforce_eager: bool,
    seed: int,
) -> None:
    llm = None
    try:
        torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max(
                max_model_len, sum(len(ids) for ids in prompt_token_ids)
            ),
            "max_num_seqs": len(prompt_token_ids),
            "num_kvcache_blocks": num_kvcache_blocks,
            "num_spec_tokens": num_spec_tokens,
            "spec_verifier_mode": spec_verifier_mode,
        }
        if case_name != "nano_baseline":
            kwargs["draft_model"] = draft_model
        if case_name == "nano_eagle3_tree":
            kwargs["speculative_token_tree"] = speculative_token_tree

        llm = LLM(target_model, **kwargs)
        sampling_params = SamplingParams(
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            max_tokens=output_length,
        )
        for _ in range(warmup_runs):
            llm.generate(
                prompt_token_ids,
                sampling_params,
                use_tqdm=False,
                return_metrics=True,
            )
        llm.get_spec_stats(reset=True)

        runs = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = perf_counter()
            outputs = llm.generate(
                prompt_token_ids,
                sampling_params,
                use_tqdm=False,
                return_metrics=True,
            )
            torch.cuda.synchronize()
            elapsed_s = perf_counter() - start
            spec_stats = llm.get_spec_stats(reset=True)
            runs.append(summarize_outputs(outputs, prompt_token_ids, elapsed_s, spec_stats))

        result_queue.put(
            {
                "case_name": case_name,
                "runs": [asdict(run) for run in runs],
                "aggregate": aggregate_runs(runs),
                "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "error": None,
            }
        )
    except Exception:
        result_queue.put(
            {
                "case_name": case_name,
                "runs": [],
                "aggregate": {},
                "torch_peak_allocated_gib": None,
                "error": traceback.format_exc(),
            }
        )
    finally:
        if llm is not None:
            llm.exit()
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_case(**kwargs) -> dict[str, Any]:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_run_case_worker,
        kwargs={"result_queue": queue, **kwargs},
    )
    process.start()
    process.join()
    if queue.empty():
        return {
            "case_name": kwargs["case_name"],
            "runs": [],
            "aggregate": {},
            "torch_peak_allocated_gib": None,
            "error": f"worker exited with code {process.exitcode} without a result",
        }
    return queue.get()


def environment_info() -> dict[str, Any]:
    total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    return {
        "nano_vllm": version("nano-vllm"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_gib": total_memory / 2**30,
        "python_executable": sys.executable,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
    }


def render_markdown(result: dict[str, Any]) -> str:
    config = result["config"]
    lines = [
        "# nano-vLLM EAGLE-3 Benchmark",
        "",
        (
            f"- workload: batch={config['batch_size']}, input={config['input_length']}, "
            f"output={config['output_length']}, greedy, repeats={config['repeats']}"
        ),
        f"- workload SHA256: `{config['workload_sha256']}`",
        "",
        "| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Peak allocated (GiB) |",
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
        lines.append(
            f"| {case_name} | {throughput:.2f} | {ttft:.2f} | {tpot:.2f} | "
            f"{'-' if acceptance is None else f'{acceptance:.2%}'} | "
            f"{'-' if accept_length is None else f'{accept_length:.3f}'} | "
            f"{case['torch_peak_allocated_gib']:.2f} |"
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
    parser.add_argument(
        "--spec-verifier-mode",
        choices=["packed", "sequential"],
        default="packed",
    )
    parser.add_argument("--speculative-token-tree", default=DEFAULT_TREE)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["nano_baseline", "nano_eagle3_linear", "nano_eagle3_tree"],
        default=["nano_baseline", "nano_eagle3_linear", "nano_eagle3_tree"],
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError(
            "Set PYTHONNOUSERSITE=1 so user-site PyTorch cannot shadow the conda environment."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    prompt_token_ids = build_fixed_length_prompts(
        tokenizer, args.batch_size, args.input_length
    )
    common = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "prompt_token_ids": prompt_token_ids,
        "output_length": args.output_length,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "num_spec_tokens": args.num_spec_tokens,
        "spec_verifier_mode": args.spec_verifier_mode,
        "speculative_token_tree": args.speculative_token_tree,
        "max_model_len": args.max_model_len,
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    result = {
        "engine": "nano-vllm",
        "environment": environment_info(),
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
            "spec_verifier_mode": args.spec_verifier_mode,
            "speculative_token_tree": args.speculative_token_tree,
            "max_model_len": args.max_model_len,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "workload_sha256": workload_digest(prompt_token_ids),
        },
        "cases": {},
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
