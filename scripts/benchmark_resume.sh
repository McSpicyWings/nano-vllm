#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

conda_bin="${CONDA_EXE:-}"
if [[ -z "$conda_bin" ]]; then
    conda_bin="$(command -v conda || true)"
fi
if [[ -z "$conda_bin" ]]; then
    fallback_conda="/home/wpc/.local/opt/miniforge3/bin/conda"
    if [[ -x "$fallback_conda" ]]; then
        conda_bin="$fallback_conda"
    else
        echo "conda executable not found; set CONDA_EXE" >&2
        exit 1
    fi
fi

nano_env="${NANO_VLLM_ENV:-nano-vllm}"
vllm_env="${VLLM_ENV:-vllm}"
expected_nano_version="${EXPECTED_NANO_VLLM_VERSION:-0.2.0}"
expected_vllm_version="${EXPECTED_VLLM_VERSION:-0.13.0}"
target_model="${TARGET_MODEL:-./huggingface/Qwen3-1.7B}"
draft_model="${DRAFT_MODEL:-./huggingface/AngelSlim/Qwen3-1.7B_eagle3}"
output_dir="${BENCHMARK_OUTPUT_DIR:-benchmarks/results/rtx5060ti_qwen3_1_7b}"

batch_size=16
input_length=128
output_length=128
repeats=3
warmup_runs=1
nano_cases=(nano_baseline nano_eagle3_linear nano_eagle3_tree)
vllm_cases=(vllm_baseline vllm_eagle3)

if [[ "${1:-}" == "--smoke" ]]; then
    output_dir="${BENCHMARK_OUTPUT_DIR:-benchmarks/results/smoke}"
    batch_size=2
    input_length=32
    output_length=8
    repeats=1
    nano_cases=(nano_baseline nano_eagle3_tree)
    vllm_cases=(vllm_baseline vllm_eagle3)
elif [[ -n "${1:-}" ]]; then
    output_dir="$1"
fi

if [[ ! -d "$target_model" || ! -d "$draft_model" ]]; then
    echo "model directories not found: $target_model / $draft_model" >&2
    exit 1
fi

mkdir -p "$output_dir"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nano_version="$(
    PYTHONNOUSERSITE=1 "$conda_bin" run -n "$nano_env" \
        python -c 'import importlib.metadata; print(importlib.metadata.version("nano-vllm"))'
)"
vllm_version="$(
    PYTHONNOUSERSITE=1 "$conda_bin" run -n "$vllm_env" \
        python -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))'
)"
if [[ "$nano_version" != "$expected_nano_version" ]]; then
    echo "nano-vllm version mismatch: expected $expected_nano_version, got $nano_version" >&2
    exit 1
fi
if [[ "${vllm_version%%+*}" != "$expected_vllm_version" ]]; then
    echo "vLLM version mismatch: expected $expected_vllm_version, got $vllm_version" >&2
    exit 1
fi
echo "nano-vllm=$nano_version, vllm=$vllm_version, PYTHONNOUSERSITE=1"

common_args=(
    --target-model "$target_model"
    --draft-model "$draft_model"
    --batch-size "$batch_size"
    --input-length "$input_length"
    --output-length "$output_length"
    --repeats "$repeats"
    --warmup-runs "$warmup_runs"
    --num-spec-tokens 3
    --max-model-len 2048
)

PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -n "$nano_env" \
    python bench_spec_decode.py \
    "${common_args[@]}" \
    --num-kvcache-blocks 64 \
    --cases "${nano_cases[@]}" \
    --output-json "$output_dir/nano.json" \
    --output-markdown "$output_dir/nano.md"

PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -n "$vllm_env" \
    python vllm_test/bench_vllm_eagle3.py \
    "${common_args[@]}" \
    --gpu-memory-utilization 0.7 \
    --cases "${vllm_cases[@]}" \
    --output-json "$output_dir/vllm.json" \
    --output-markdown "$output_dir/vllm.md"

if [[ "${1:-}" != "--smoke" ]]; then
    PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -n "$nano_env" \
        python benchmarks/merge_results.py \
        --nano-json "$output_dir/nano.json" \
        --vllm-json "$output_dir/vllm.json" \
        --output-json "$output_dir/combined.json" \
        --output-markdown "$output_dir/report.md"
else
    echo "Smoke benchmark completed: $output_dir"
fi
