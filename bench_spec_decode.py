"""
Benchmark script to compare performance with and without speculative decoding.
Tests Eagle3 as draft model for accelerating inference.
"""
import os
import time
import traceback
from multiprocessing import get_context
from random import randint, seed
import torch
from nanovllm import LLM, SamplingParams


def run_benchmark(
    llm: LLM,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    warmup: bool = True,
) -> tuple[float, int]:
    """Run benchmark and return (time, total_tokens)."""
    if warmup:
        # Warmup run
        llm.generate(["Benchmark warmup: "], SamplingParams(max_tokens=10))
    
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    return elapsed, total_tokens


def _run_test_worker(
    result_queue,
    test_name: str,
    target_model: str,
    draft_model: str | None,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    num_spec_tokens: int,
):
    try:
        llm = LLM(
            target_model,
            draft_model=draft_model,
            enforce_eager=False,
            max_model_len=4096,
            num_spec_tokens=num_spec_tokens,
        )
        elapsed, total_tokens = run_benchmark(llm, prompt_token_ids, sampling_params)
        throughput = total_tokens / elapsed
        llm.exit()
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        result_queue.put({
            "test_name": test_name,
            "elapsed": elapsed,
            "total_tokens": total_tokens,
            "throughput": throughput,
        })
    except Exception:
        result_queue.put({
            "test_name": test_name,
            "error": traceback.format_exc(),
        })


def run_test_in_subprocess(
    test_name: str,
    target_model: str,
    draft_model: str | None,
    prompt_token_ids: list[list[int]],
    sampling_params: list[SamplingParams],
    num_spec_tokens: int,
) -> dict:
    """Run one benchmark in a subprocess and return results or error."""
    ctx = get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_run_test_worker,
        args=(
            result_queue,
            test_name,
            target_model,
            draft_model,
            prompt_token_ids,
            sampling_params,
            num_spec_tokens,
        ),
    )
    proc.start()
    proc.join()
    if result_queue.empty():
        return {"test_name": test_name, "error": "Subprocess finished without returning a result."}
    return result_queue.get()


def main():
    seed(42)
    
    # Benchmark parameters
    num_seqs = 32  # Reduced for faster testing
    max_input_len = 512
    max_output_len = 256
    
    # Model paths - adjust these to your local paths
    target_model = os.path.expanduser("./huggingface/Qwen3-1.7B/")
    draft_model = os.path.expanduser("./huggingface/AngelSlim/Qwen3-1.7B_eagle3")
    
    # Check if models exist
    if not os.path.isdir(target_model):
        print(f"Target model not found at {target_model}")
        print("Please update the target_model path in the script.")
        return
    
    # Generate test prompts
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(50, max_output_len))
        for _ in range(num_seqs)
    ]
    
    print("=" * 60)
    print("Speculative Decoding Benchmark")
    print("=" * 60)
    print(f"Number of sequences: {num_seqs}")
    print(f"Max input length: {max_input_len}")
    print(f"Max output length: {max_output_len}")
    print()
    
    
    # Test 1: Without speculative decoding (baseline)
    
    print("-" * 60)
    print("Test 1: Baseline (no speculative decoding)")
    print("-" * 60)
    baseline_result = run_test_in_subprocess(
        "baseline",
        target_model,
        None,
        prompt_token_ids,
        sampling_params,
        num_spec_tokens=1,
    )
    if "error" in baseline_result:
        print(baseline_result["error"])
        return
    elapsed_baseline = baseline_result["elapsed"]
    total_tokens = baseline_result["total_tokens"]
    throughput_baseline = baseline_result["throughput"]

    print(f"Total tokens: {total_tokens}")
    print(f"Time: {elapsed_baseline:.2f}s")
    print(f"Throughput: {throughput_baseline:.2f} tok/s")
    
    # Test 2: With speculative decoding (if draft model exists)
    if os.path.isdir(draft_model):
        print()
        print("-" * 60)
        print("Test 2: With speculative decoding (Eagle3 draft)")
        print("-" * 60)
        
        spec_result = run_test_in_subprocess(
            "speculative",
            target_model,
            draft_model,
            prompt_token_ids,
            sampling_params,
            num_spec_tokens=5,
        )
        if "error" in spec_result:
            print(spec_result["error"])
            return
        elapsed_spec = spec_result["elapsed"]
        total_tokens = spec_result["total_tokens"]
        throughput_spec = spec_result["throughput"]

        print(f"Total tokens: {total_tokens}")
        print(f"Time: {elapsed_spec:.2f}s")
        print(f"Throughput: {throughput_spec:.2f} tok/s")
        
        # Comparison
        print()
        print("=" * 60)
        print("Comparison")
        print("=" * 60)
        speedup = throughput_spec / throughput_baseline
        print(f"Baseline throughput: {throughput_baseline:.2f} tok/s")
        print(f"Speculative throughput: {throughput_spec:.2f} tok/s")
        print(f"Speedup: {speedup:.2f}x")
        
        
    else:
        print()
        print(f"Draft model not found at {draft_model}")
        print("Skipping speculative decoding test.")
        print("To test speculative decoding, update the draft_model path.")
    
    print()
    print("Benchmark completed.")


if __name__ == "__main__":
    main()
