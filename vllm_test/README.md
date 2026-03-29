# vLLM Eagle3 Benchmark

在 `conda activate vllm` 环境里运行。

默认脚本会：
- 使用本地 `Qwen3-1.7B` 作为 target
- 使用本地 `Qwen3-1.7B_eagle3` 作为 draft
- 默认使用更适合本地 16GB GPU 的 `--gpu-memory-utilization 0.7` 和 `--max-model-len 2048`
- 先 warmup，再分别跑 baseline 和 Eagle3 speculative decoding
- 输出 output throughput、acceptance length、draft acceptance rate

脚本在 `temperature=0` 且 `top_k<=0` 时，会自动把实际采样配置收成 `top_k=1`。
这和 greedy 语义等价，但可以避免本地小显存机器在 speculative rejection sampler 初始化时做全 vocab sort 导致 OOM。

## 常用命令

温度 `0`：

```bash
python vllm_test/bench_vllm_eagle3.py \
  --temperature 0.0 \
  --num-spec-tokens 3 \
  --max-tokens 128 \
  --num-prompts 16 \
  --output-json vllm_test/result_t0.json
```

温度 `1`：

```bash
python vllm_test/bench_vllm_eagle3.py \
  --temperature 1.0 \
  --num-spec-tokens 3 \
  --max-tokens 128 \
  --num-prompts 16 \
  --output-json vllm_test/result_t1.json
```

如果想换 prompt 文件：

```bash
python vllm_test/bench_vllm_eagle3.py \
  --prompt-file path/to/prompts.txt
```

`prompts.txt` 一行一个 prompt，脚本会循环使用直到满足 `--num-prompts`。
