#!/usr/bin/env python3
"""Validate and merge nano-vLLM/vLLM benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHARED_CONFIG_KEYS = [
    "target_model",
    "draft_model",
    "batch_size",
    "input_length",
    "output_length",
    "temperature",
    "top_p",
    "top_k",
    "ignore_eos",
    "seed",
    "num_spec_tokens",
    "max_model_len",
    "warmup_runs",
    "repeats",
    "workload_sha256",
]


def metric(case: dict[str, Any], name: str) -> float | None:
    value = case["aggregate"].get(name)
    return None if value is None else float(value["mean"])


def render_markdown(combined: dict[str, Any]) -> str:
    config = combined["config"]
    lines = [
        "# Qwen3-1.7B EAGLE-3 Unified Benchmark",
        "",
        (
            f"RTX 5060 Ti 16GB; batch={config['batch_size']}; "
            f"input/output={config['input_length']}/{config['output_length']}; "
            f"greedy; K={config['num_spec_tokens']}; "
            f"{config['warmup_runs']} warmup + {config['repeats']} measured runs."
        ),
        (
            "- correctness: nano baseline/linear/tree token IDs match exactly; "
            "vLLM speculative output digest "
            f"{'matches' if combined['correctness']['vllm_spec_output_matches_baseline'] else 'does not match'} "
            "its baseline on this software stack."
        ),
        "",
        "| Case | Output tok/s (mean ± std) | TTFT (ms) | TPOT (ms) | Acceptance | Accept length | Peak memory (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, case in combined["cases"].items():
        throughput = case["aggregate"]["output_tokens_per_s"]
        acceptance = metric(case, "acceptance_rate")
        accept_length = metric(case, "mean_acceptance_length")
        peak = case.get("torch_peak_allocated_gib", case.get("device_memory_peak_gib"))
        lines.append(
            f"| {case_name} | {throughput['mean']:.2f} ± {throughput['std']:.2f} | "
            f"{metric(case, 'ttft_ms_mean'):.2f} | {metric(case, 'tpot_ms_mean'):.2f} | "
            f"{'-' if acceptance is None else f'{acceptance:.2%}'} | "
            f"{'-' if accept_length is None else f'{accept_length:.3f}'} | "
            f"{'-' if peak is None else f'{float(peak):.2f}'} |"
        )

    speedups = combined["speedups"]
    lines.extend(
        [
            "",
            "## Speedups",
            "",
            f"- nano linear / nano baseline: `{speedups['nano_linear_vs_baseline']:.3f}x`",
            f"- nano tree / nano baseline: `{speedups['nano_tree_vs_baseline']:.3f}x`",
            f"- vLLM EAGLE-3 / vLLM baseline: `{speedups['vllm_eagle3_vs_baseline']:.3f}x`",
            "",
            "## Resume-ready wording",
            "",
            combined["resume_wording"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nano-json", type=Path, required=True)
    parser.add_argument("--vllm-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nano = json.loads(args.nano_json.read_text())
    vllm = json.loads(args.vllm_json.read_text())
    for key in SHARED_CONFIG_KEYS:
        if nano["config"][key] != vllm["config"][key]:
            raise ValueError(
                f"benchmark config mismatch for {key}: "
                f"{nano['config'][key]!r} != {vllm['config'][key]!r}"
            )
    cases = {**nano["cases"], **vllm["cases"]}
    for case_name, case in cases.items():
        if case.get("error"):
            raise ValueError(f"{case_name} failed: {case['error']}")
        if not case["aggregate"].get("all_output_digests_equal", False):
            raise ValueError(f"{case_name} produced different output token digests across repeats")
    nano_correctness_group = (
        "nano_baseline",
        "nano_eagle3_linear",
        "nano_eagle3_tree",
    )
    nano_digests = {
        cases[name]["runs"][0]["output_sha256"] for name in nano_correctness_group
    }
    if len(nano_digests) != 1:
        raise ValueError(
            f"greedy correctness mismatch across cases: {nano_correctness_group}"
        )
    # vLLM is an external comparison rather than code controlled by this repo.
    # Preserve its digest comparison in the report, but do not turn a
    # version/kernel-dependent numerical difference into a nano-vLLM failure.
    vllm_output_matches_baseline = (
        cases["vllm_baseline"]["runs"][0]["output_sha256"]
        == cases["vllm_eagle3"]["runs"][0]["output_sha256"]
    )

    def throughput(case_name: str) -> float:
        return metric(cases[case_name], "output_tokens_per_s") or 0.0

    speedups = {
        "nano_linear_vs_baseline": (
            throughput("nano_eagle3_linear") / throughput("nano_baseline")
        ),
        "nano_tree_vs_baseline": (
            throughput("nano_eagle3_tree") / throughput("nano_baseline")
        ),
        "vllm_eagle3_vs_baseline": (
            throughput("vllm_eagle3") / throughput("vllm_baseline")
        ),
    }
    tree_acceptance = metric(cases["nano_eagle3_tree"], "acceptance_rate")
    tree_length = metric(cases["nano_eagle3_tree"], "mean_acceptance_length")
    tree_tpot = metric(cases["nano_eagle3_tree"], "tpot_ms_mean")
    baseline_tpot = metric(cases["nano_baseline"], "tpot_ms_mean")
    resume_wording = (
        "在 nano-vLLM 中实现 EAGLE-3 双模型 KV Cache、跨 tokenizer vocabulary mapping "
        "与多分支 tree proposal/target verification；构建固定 128/128、batch 16、"
        f"3 次重复的正确性与性能基准，实测 draft acceptance rate {tree_acceptance:.1%}、"
        f"mean acceptance length {tree_length:.2f}，TPOT {tree_tpot:.2f} ms "
        f"（自回归 {baseline_tpot:.2f} ms），吞吐为自回归基线的 "
        f"{speedups['nano_tree_vs_baseline']:.2f}x，并定位 target verification 与 cache replay 开销。"
    )
    combined = {
        "config": {key: nano["config"][key] for key in SHARED_CONFIG_KEYS},
        "hardware": nano["environment"],
        "nano_environment": nano["environment"],
        "vllm_environment": vllm["environment"],
        "correctness": {
            "nano_greedy_outputs_match": True,
            "vllm_spec_output_matches_baseline": vllm_output_matches_baseline,
        },
        "cases": cases,
        "speedups": speedups,
        "resume_wording": resume_wording,
    }
    markdown = render_markdown(combined)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    args.output_markdown.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
