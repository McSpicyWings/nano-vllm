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
vllm_source_repo="${VLLM_SOURCE_REPO:-/home/wpc/project/vllm}"
vllm_flash_attn_source="${VLLM_FLASH_ATTN_SOURCE:-$vllm_source_repo/.deps/vllm-flash-attn-src/vllm_flash_attn}"
expected_flash_attn_commit="${EXPECTED_VLLM_FLASH_ATTN_COMMIT:-86f8f157cf82aa2342743752b97788922dd7de43}"
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
correctness_args=()

if [[ "${1:-}" == "--smoke" ]]; then
    output_dir="${BENCHMARK_OUTPUT_DIR:-benchmarks/results/smoke}"
    batch_size=2
    input_length=32
    output_length=8
    repeats=1
    nano_cases=(nano_baseline nano_eagle3_linear)
    vllm_cases=(vllm_baseline vllm_eagle3)
    correctness_args=(--batch-sizes 1 --output-length 8 --skip-boundary)
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
    python tests/test_eagle3_correctness.py \
    --target-model "$target_model" \
    --draft-model "$draft_model" \
    "${correctness_args[@]}" \
    --output-json "$output_dir/nano_correctness.json"

PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -n "$nano_env" \
    python bench_spec_decode.py \
    "${common_args[@]}" \
    --num-kvcache-blocks 64 \
    --cases "${nano_cases[@]}" \
    --output-json "$output_dir/nano.json" \
    --output-markdown "$output_dir/nano.md"

if [[ ! -d "$vllm_source_repo/.git" && ! -f "$vllm_source_repo/.git" ]]; then
    echo "vLLM source repository not found: $vllm_source_repo" >&2
    exit 1
fi
if [[ ! -d "$vllm_flash_attn_source" ]]; then
    echo "vLLM flash-attention source not found: $vllm_flash_attn_source" >&2
    exit 1
fi
actual_flash_attn_commit="$(
    git -C "$vllm_flash_attn_source" rev-parse HEAD 2>/dev/null || true
)"
if [[ "$actual_flash_attn_commit" != "$expected_flash_attn_commit" ]]; then
    echo "vLLM flash-attention commit mismatch: expected $expected_flash_attn_commit, got $actual_flash_attn_commit" >&2
    exit 1
fi
vllm_tag="v${expected_vllm_version%%+*}"
if ! git -C "$vllm_source_repo" rev-parse --verify --quiet \
    "refs/tags/$vllm_tag^{commit}" >/dev/null; then
    echo "vLLM source tag not found: $vllm_tag" >&2
    exit 1
fi
vllm_temp_root="$(mktemp -d "${TMPDIR:-/tmp}/nano-vllm-vllm.XXXXXX")"
vllm_worktree="$vllm_temp_root/source"
cleanup_vllm_worktree() {
    git -C "$vllm_source_repo" worktree remove --force \
        "$vllm_worktree" >/dev/null 2>&1 || true
    rmdir "$vllm_temp_root" >/dev/null 2>&1 || true
}
trap cleanup_vllm_worktree EXIT
git -C "$vllm_source_repo" worktree add --detach \
    "$vllm_worktree" "$vllm_tag" >/dev/null
mapfile -d '' vllm_runtime_binaries < <(
    find "$vllm_source_repo/vllm" -maxdepth 1 \
        -type f -name '*.so' -print0
    find "$vllm_source_repo/vllm/vllm_flash_attn" \
        -type f -name '*.so' -print0
)
if (( ${#vllm_runtime_binaries[@]} == 0 )); then
    echo "compiled vLLM extensions not found in $vllm_source_repo/vllm" >&2
    exit 1
fi
for runtime_file in "${vllm_runtime_binaries[@]}"; do
    relative_file="${runtime_file#"$vllm_source_repo/vllm/"}"
    file_destination="$vllm_worktree/vllm/$relative_file"
    mkdir -p "$(dirname "$file_destination")"
    ln -s "$runtime_file" "$file_destination"
done
while IFS= read -r flash_attn_file; do
    relative_file="${flash_attn_file#"$vllm_flash_attn_source/"}"
    file_destination="$vllm_worktree/vllm/vllm_flash_attn/$relative_file"
    mkdir -p "$(dirname "$file_destination")"
    ln -s "$flash_attn_file" "$file_destination"
done < <(
    find "$vllm_flash_attn_source" \
        -type f -name '*.py'
)

PYTHONPATH="$vllm_worktree${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -n "$vllm_env" \
    python vllm_test/bench_vllm_eagle3.py \
    "${common_args[@]}" \
    --gpu-memory-utilization 0.7 \
    --cases "${vllm_cases[@]}" \
    --output-json "$output_dir/vllm.json" \
    --output-markdown "$output_dir/vllm.md"
cleanup_vllm_worktree
trap - EXIT

if [[ "${1:-}" != "--smoke" ]]; then
    PYTHONNOUSERSITE=1 "$conda_bin" run --no-capture-output -n "$nano_env" \
        python benchmarks/merge_results.py \
        --nano-json "$output_dir/nano.json" \
        --vllm-json "$output_dir/vllm.json" \
        --nano-correctness-json "$output_dir/nano_correctness.json" \
        --output-json "$output_dir/combined.json" \
        --output-markdown "$output_dir/report.md"
else
    echo "Smoke benchmark completed: $output_dir"
fi
