#!/usr/bin/env python3
"""
Benchmark vLLM baseline vs Eagle3 speculative decoding using local models.

This script intentionally uses tokenized prompts and detokenize=False so the
measured wall time stays focused on inference instead of tokenizer overhead.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any

from transformers import AutoTokenizer


DEFAULT_TARGET_MODEL = "./huggingface/Qwen3-1.7B"
DEFAULT_DRAFT_MODEL = "./huggingface/AngelSlim/Qwen3-1.7B_eagle3"

PROMPT_BANK = [
    "Explain the practical difference between throughput and latency for large language model inference.",
    "Write a concise note about why KV cache layout matters during transformer decoding.",
    "Summarize how speculative decoding trades verifier cost against acceptance rate.",
    "Describe one reason CUDA graphs can improve small-batch decode performance.",
    "Explain why a draft model with low acceptance can become a negative optimization.",
    "Give a concise explanation of tree attention in speculative decoding.",
    "Write a short paragraph about why Python-side tensor assembly can become a hot path.",
    "Explain why benchmark prompts should resemble natural language instead of random token ids.",
    "Describe how greedy decoding and temperature sampling can change speculative acceptance.",
    "Write a brief explanation of why verifier-side target forwards dominate latency in many speculative implementations.",
    "Explain the difference between prefill throughput and decode throughput in one paragraph.",
    "Describe how chunked prefill can help utilization for offline inference workloads.",
    "Write a short answer explaining why acceptance length matters more than raw draft speed.",
    "Explain why model-pair compatibility is necessary but not sufficient for speculative speedup.",
    "Give a brief note on why attention metadata can become a bottleneck in custom runtimes.",
    "Summarize why exactness constraints can reduce speculative decoding performance.",
]

SPEC_METRIC_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_METRIC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_METRIC_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"
SPEC_METRIC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"


@dataclass
class CaseResult:
    case_name: str
    elapsed_s: float
    total_prompt_tokens: int
    total_output_tokens: int
    overall_tokens_per_s: float
    output_tokens_per_s: float
    num_drafts: float = 0.0
    num_draft_tokens: float = 0.0
    num_accepted_tokens: float = 0.0
    mean_acceptance_length: float | None = None
    draft_acceptance_rate: float | None = None
    acceptance_per_pos: list[float] | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-spec-tokens", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--warmup-max-tokens", type=int, default=8)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true", default=True)
    parser.add_argument("--disable-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-spec", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_prompt_texts(prompt_file: Path | None, num_prompts: int) -> list[str]:
    if prompt_file is None:
        return list(islice(cycle(PROMPT_BANK), num_prompts))

    lines = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No usable prompts found in {prompt_file}")
    return list(islice(cycle(lines), num_prompts))


def tokenize_prompts(model_path: str, prompts: list[str]) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    return [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]


def summarize_spec_metrics(metrics: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(float)
    per_pos: list[float] | None = None

    for metric in metrics:
        name = getattr(metric, "name", None)
        if name not in {
            SPEC_METRIC_DRAFTS,
            SPEC_METRIC_DRAFT_TOKENS,
            SPEC_METRIC_ACCEPTED_TOKENS,
            SPEC_METRIC_ACCEPTED_PER_POS,
        }:
            continue

        if name == SPEC_METRIC_ACCEPTED_PER_POS:
            values = list(getattr(metric, "values", []))
            if per_pos is None:
                per_pos = [0.0] * len(values)
            for i, value in enumerate(values):
                per_pos[i] += float(value)
        else:
            out[name] += float(getattr(metric, "value", 0.0))

    if per_pos is not None:
        out[SPEC_METRIC_ACCEPTED_PER_POS] = per_pos
    return dict(out)


def diff_spec_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = {
        SPEC_METRIC_DRAFTS,
        SPEC_METRIC_DRAFT_TOKENS,
        SPEC_METRIC_ACCEPTED_TOKENS,
        SPEC_METRIC_ACCEPTED_PER_POS,
    }
    out: dict[str, Any] = {}
    for key in keys:
        if key == SPEC_METRIC_ACCEPTED_PER_POS:
            before_vals = before.get(key, [])
            after_vals = after.get(key, [])
            size = max(len(before_vals), len(after_vals))
            out[key] = [
                float(after_vals[i] if i < len(after_vals) else 0.0)
                - float(before_vals[i] if i < len(before_vals) else 0.0)
                for i in range(size)
            ]
        else:
            out[key] = float(after.get(key, 0.0)) - float(before.get(key, 0.0))
    return out


def _run_case_worker(
    result_queue,
    *,
    case_name: str,
    target_model: str,
    draft_model: str | None,
    prompt_token_ids: list[list[int]],
    max_tokens: int,
    warmup_max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    enable_chunked_prefill: bool,
    enforce_eager: bool,
    num_spec_tokens: int,
):
    try:
        import gc

        import torch
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        effective_top_k = 1 if temperature == 0.0 and top_k <= 0 else top_k

        speculative_config = None
        if draft_model is not None:
            speculative_config = {
                "method": "eagle3",
                "model": draft_model,
                "num_speculative_tokens": num_spec_tokens,
            }

        llm = LLM(
            model=target_model,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
            disable_log_stats=False,
            speculative_config=speculative_config,
        )

        warmup_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=effective_top_k,
            seed=seed,
            ignore_eos=True,
            detokenize=False,
            max_tokens=warmup_max_tokens,
        )
        warmup_prompt = [TokensPrompt(prompt_token_ids=prompt_token_ids[0])]
        llm.generate(warmup_prompt, warmup_params, use_tqdm=False)

        before_metrics = summarize_spec_metrics(llm.get_metrics())

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=effective_top_k,
            seed=seed,
            ignore_eos=True,
            detokenize=False,
            max_tokens=max_tokens,
        )
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
        total_prompt_tokens = sum(len(ids) for ids in prompt_token_ids)

        torch.cuda.synchronize()
        t0 = perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - t0

        after_metrics = summarize_spec_metrics(llm.get_metrics())
        spec_delta = diff_spec_metrics(before_metrics, after_metrics)

        total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        overall_tokens = total_prompt_tokens + total_output_tokens
        result = CaseResult(
            case_name=case_name,
            elapsed_s=elapsed,
            total_prompt_tokens=total_prompt_tokens,
            total_output_tokens=total_output_tokens,
            overall_tokens_per_s=overall_tokens / elapsed if elapsed > 0 else 0.0,
            output_tokens_per_s=total_output_tokens / elapsed if elapsed > 0 else 0.0,
        )

        num_drafts = spec_delta.get(SPEC_METRIC_DRAFTS, 0.0)
        num_draft_tokens = spec_delta.get(SPEC_METRIC_DRAFT_TOKENS, 0.0)
        num_accepted_tokens = spec_delta.get(SPEC_METRIC_ACCEPTED_TOKENS, 0.0)
        if num_drafts > 0:
            result.num_drafts = num_drafts
            result.num_draft_tokens = num_draft_tokens
            result.num_accepted_tokens = num_accepted_tokens
            result.mean_acceptance_length = 1.0 + (num_accepted_tokens / num_drafts)
            result.draft_acceptance_rate = (
                num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0
            )
            per_pos = spec_delta.get(SPEC_METRIC_ACCEPTED_PER_POS, [])
            result.acceptance_per_pos = [
                float(x) / num_drafts for x in per_pos
            ]

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        result_queue.put(asdict(result))
    except Exception:
        result_queue.put(
            asdict(
                CaseResult(
                    case_name=case_name,
                    elapsed_s=0.0,
                    total_prompt_tokens=0,
                    total_output_tokens=0,
                    overall_tokens_per_s=0.0,
                    output_tokens_per_s=0.0,
                    error=traceback.format_exc(),
                )
            )
        )


def run_case(**kwargs) -> CaseResult:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_case_worker, kwargs={"result_queue": queue, **kwargs})
    proc.start()
    proc.join()

    if queue.empty():
        return CaseResult(
            case_name=kwargs["case_name"],
            elapsed_s=0.0,
            total_prompt_tokens=0,
            total_output_tokens=0,
            overall_tokens_per_s=0.0,
            output_tokens_per_s=0.0,
            error="Subprocess exited without returning a result.",
        )
    return CaseResult(**queue.get())


def print_case(result: CaseResult) -> None:
    print("-" * 64)
    print(result.case_name)
    print("-" * 64)
    if result.error is not None:
        print(result.error)
        return
    print(f"elapsed: {result.elapsed_s:.3f}s")
    print(f"prompt tokens: {result.total_prompt_tokens}")
    print(f"output tokens: {result.total_output_tokens}")
    print(f"overall throughput: {result.overall_tokens_per_s:.2f} tok/s")
    print(f"output throughput: {result.output_tokens_per_s:.2f} tok/s")
    if result.mean_acceptance_length is not None:
        print(f"num drafts: {result.num_drafts:.0f}")
        print(f"draft tokens: {result.num_draft_tokens:.0f}")
        print(f"accepted tokens: {result.num_accepted_tokens:.0f}")
        print(f"mean acceptance length: {result.mean_acceptance_length:.3f}")
        print(f"draft acceptance rate: {result.draft_acceptance_rate:.3%}")
        if result.acceptance_per_pos is not None:
            rates = ", ".join(f"{rate:.3f}" for rate in result.acceptance_per_pos)
            print(f"acceptance per position: [{rates}]")


def main() -> None:
    args = parse_args()
    prompt_texts = load_prompt_texts(args.prompt_file, args.num_prompts)
    prompt_token_ids = tokenize_prompts(args.target_model, prompt_texts)

    common_kwargs = dict(
        target_model=args.target_model,
        prompt_token_ids=prompt_token_ids,
        max_tokens=args.max_tokens,
        warmup_max_tokens=args.warmup_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        num_spec_tokens=args.num_spec_tokens,
    )

    results: dict[str, Any] = {
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "effective_top_k": 1 if args.temperature == 0.0 and args.top_k <= 0 else args.top_k,
            "seed": args.seed,
            "num_spec_tokens": args.num_spec_tokens,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "enforce_eager": args.enforce_eager,
        }
    }

    baseline_result = None
    spec_result = None

    if not args.skip_baseline:
        baseline_result = run_case(
            case_name="baseline",
            draft_model=None,
            **common_kwargs,
        )
        print_case(baseline_result)
        results["baseline"] = asdict(baseline_result)

    if not args.skip_spec:
        spec_result = run_case(
            case_name="eagle3_spec",
            draft_model=args.draft_model,
            **common_kwargs,
        )
        print_case(spec_result)
        results["eagle3_spec"] = asdict(spec_result)

    if baseline_result is not None and spec_result is not None:
        if baseline_result.error is None and spec_result.error is None:
            speedup = spec_result.output_tokens_per_s / baseline_result.output_tokens_per_s
            print("=" * 64)
            print(f"output-throughput speedup: {speedup:.3f}x")
            results["speedup_vs_baseline"] = speedup
        else:
            results["speedup_vs_baseline"] = None

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"wrote results to {args.output_json}")


if __name__ == "__main__":
    main()
