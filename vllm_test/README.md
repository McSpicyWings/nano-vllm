# vLLM EAGLE-3 对照基准

这个脚本和 `bench_spec_decode.py` 共用固定自然语言 workload、tokenizer
整理逻辑和 digest，用于对照 vLLM baseline 与 vLLM EAGLE-3。正式结果使用
vLLM 0.13.0 和 `conda` 环境 `vllm`。

推荐从仓库根目录运行一键脚本：

```bash
./scripts/benchmark_resume.sh
```

只运行 vLLM：

```bash
conda activate vllm
PYTHONNOUSERSITE=1 python vllm_test/bench_vllm_eagle3.py \
  --target-model ./huggingface/Qwen3-1.7B \
  --draft-model ./huggingface/AngelSlim/Qwen3-1.7B_eagle3 \
  --batch-size 16 \
  --input-length 128 \
  --output-length 128 \
  --warmup-runs 1 \
  --repeats 3 \
  --num-spec-tokens 3 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --output-json benchmarks/results/rtx5060ti_qwen3_1_7b/vllm.json \
  --output-markdown benchmarks/results/rtx5060ti_qwen3_1_7b/vllm.md
```

脚本固定使用 `temperature=0`、`top_p=1`、`top_k=1`、`ignore_eos=True`
和 seed 42。吞吐按实际 output token 数计算。请求级 TTFT、TPOT 和端到端延迟
来自 vLLM request metrics，接受指标来自 vLLM Prometheus metrics。device peak
由 NVML 采样，因此不要和 nano 的 PyTorch allocator peak 直接比较。

旧的温度和适配实验保存在 [`legacy`](legacy)，不属于正式三次结果。
