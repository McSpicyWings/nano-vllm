#!/usr/bin/env python3
"""GPU correctness gate for baseline, linear EAGLE-3, and tree EAGLE-3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


DEFAULT_TARGET = "./huggingface/Qwen3-1.7B"
DEFAULT_DRAFT = "./huggingface/AngelSlim/Qwen3-1.7B_eagle3"
DEFAULT_TREE = str(
    [(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)]
)
PROMPTS = [
    "Explain why speculative decoding must preserve the target model distribution.",
    "Describe how a KV cache reduces autoregressive decoding work.",
    "请简要解释投机解码中的接受长度。",
    "Why can target verification dominate EAGLE-3 latency?",
    "Summarize the difference between TTFT and TPOT.",
    "给出一个批处理推理吞吐量的定义。",
    "Explain why vocabulary mapping must not restrict target fallback tokens.",
    "Describe tree-based draft proposal in one paragraph.",
    "What does a scheduler do in an LLM inference engine?",
    "Explain why benchmark warmup matters.",
    "What is the purpose of CUDA graph capture?",
    "Describe one failure mode of speculative KV cache rollback.",
    "解释为什么固定随机种子有助于性能测试。",
    "What is greedy decoding?",
    "Explain the tradeoff between batch size and latency.",
    "Summarize why negative benchmark results can still be useful.",
]


def normalize_prompts(tokenizer, batch_size: int, input_length: int) -> list[list[int]]:
    prompts = []
    for index in range(batch_size):
        source = PROMPTS[index % len(PROMPTS)]
        ids = tokenizer.encode(source, add_special_tokens=False)
        while len(ids) < input_length:
            ids.extend(tokenizer.encode(" " + source, add_special_tokens=False))
        prompts.append(ids[:input_length])
    return prompts


def run_case(
    name: str,
    target: str,
    draft: str,
    prompts: list[list[int]],
    output_length: int,
    tree: str | None,
) -> list[list[int]]:
    kwargs = {
        "enforce_eager": True,
        "max_model_len": max(512, len(prompts[0]) + output_length + 16),
        "max_num_batched_tokens": max(512, len(prompts) * len(prompts[0])),
        "max_num_seqs": len(prompts),
        "num_kvcache_blocks": max(16, len(prompts) * 2),
        "num_spec_tokens": 3,
    }
    if name != "baseline":
        kwargs["draft_model"] = draft
    if tree is not None:
        kwargs["speculative_token_tree"] = tree
    torch.manual_seed(42)
    llm = LLM(target, **kwargs)
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, ignore_eos=True, max_tokens=output_length),
            use_tqdm=False,
        )
        if name != "baseline":
            assert not llm.model_runner.seq_prev_hidden
        return [output["token_ids"] for output in outputs]
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def first_mismatch(
    expected: list[list[int]],
    actual: list[list[int]],
) -> str:
    for request_index, (expected_tokens, actual_tokens) in enumerate(
        zip(expected, actual, strict=True)
    ):
        for token_index, (expected_token, actual_token) in enumerate(
            zip(expected_tokens, actual_tokens, strict=True)
        ):
            if expected_token != actual_token:
                start = max(0, token_index - 3)
                end = min(len(expected_tokens), token_index + 4)
                return (
                    f"request={request_index}, token={token_index}, "
                    f"expected={expected_token}, actual={actual_token}, "
                    f"expected_window={expected_tokens[start:end]}, "
                    f"actual_window={actual_tokens[start:end]}"
                )
    return "sequence lengths differ"


def check_repeated_cleanup(
    target: str,
    draft: str,
    prompt: list[int],
) -> list[int]:
    torch.manual_seed(42)
    llm = LLM(
        target,
        draft_model=draft,
        speculative_token_tree=DEFAULT_TREE,
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        num_kvcache_blocks=16,
        num_spec_tokens=3,
    )
    allocations = []
    reference = None
    try:
        for _ in range(3):
            output = llm.generate(
                [prompt],
                SamplingParams(temperature=0, ignore_eos=True, max_tokens=16),
                use_tqdm=False,
            )[0]["token_ids"]
            reference = output if reference is None else reference
            assert output == reference
            assert not llm.model_runner.seq_prev_hidden
            assert llm.scheduler.is_finished()
            assert (
                len(llm.scheduler.block_manager.free_block_ids)
                == llm.config.num_kvcache_blocks
            )
            torch.cuda.synchronize()
            allocations.append(torch.cuda.memory_allocated())
        assert max(allocations) - min(allocations) <= 2 * 1024 * 1024
        return allocations
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--skip-boundary", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    workloads = [
        (f"batch_{batch_size}", normalize_prompts(tokenizer, batch_size, args.input_length))
        for batch_size in args.batch_sizes
    ]
    if not args.skip_boundary:
        workloads.append(("block_boundary", normalize_prompts(tokenizer, 1, 255)))

    summary = {}
    for workload_name, prompts in workloads:
        baseline = run_case(
            "baseline", args.target_model, args.draft_model, prompts, args.output_length, None
        )
        linear = run_case(
            "linear", args.target_model, args.draft_model, prompts, args.output_length, None
        )
        tree = run_case(
            "tree", args.target_model, args.draft_model, prompts, args.output_length, DEFAULT_TREE
        )
        assert linear == baseline, (
            f"linear mismatch for {workload_name}: {first_mismatch(baseline, linear)}"
        )
        assert tree == baseline, (
            f"tree mismatch for {workload_name}: {first_mismatch(baseline, tree)}"
        )
        digest = hashlib.sha256(
            json.dumps(baseline, separators=(",", ":")).encode()
        ).hexdigest()
        summary[workload_name] = {"requests": len(prompts), "output_sha256": digest}
        print(f"PASS {workload_name}: {digest}")

    cleanup_allocations = check_repeated_cleanup(
        args.target_model,
        args.draft_model,
        workloads[0][1][0],
    )
    summary["repeated_cleanup"] = {
        "runs": len(cleanup_allocations),
        "allocated_bytes": cleanup_allocations,
    }
    print(f"PASS repeated_cleanup: {cleanup_allocations}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
