# Qwen3-1.7B EAGLE-3 Unified Benchmark

RTX 5060 Ti 16GB; batch=16; input/output=128/128; greedy; K=3; 1 warmup + 3 measured runs.
- correctness: nano baseline/linear/tree token IDs match exactly; vLLM speculative output digest does not match its baseline on this software stack.

| Case | Output tok/s (mean ± std) | TTFT (ms) | TPOT (ms) | Acceptance | Accept length | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_baseline | 1398.54 ± 7.32 | 137.68 | 10.44 | - | - | 5.10 |
| nano_eagle3_linear | 287.59 ± 0.25 | 144.97 | 52.92 | 1.66% | 1.049 | 5.46 |
| nano_eagle3_tree | 212.83 ± 0.94 | 145.42 | 69.99 | 4.89% | 1.144 | 5.46 |
| vllm_baseline | 1366.24 ± 1.94 | 138.94 | 10.70 | - | - | 12.15 |
| vllm_eagle3 | 1291.15 ± 80.52 | 221.83 | 7.72 | 38.05% | 2.142 | 12.92 |

## Speedups

- nano linear / nano baseline: `0.206x`
- nano tree / nano baseline: `0.152x`
- vLLM EAGLE-3 / vLLM baseline: `0.945x`

## Resume-ready wording

在 nano-vLLM 中实现 EAGLE-3 双模型 KV Cache、跨 tokenizer vocabulary mapping 与多分支 tree proposal/target verification；构建固定 128/128、batch 16、3 次重复的正确性与性能基准，实测 draft acceptance rate 4.9%、mean acceptance length 1.14，TPOT 69.99 ms （自回归 10.44 ms），吞吐为自回归基线的 0.15x，并定位 target verification 与 cache replay 开销。
