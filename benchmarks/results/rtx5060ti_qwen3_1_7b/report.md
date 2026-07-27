# Qwen3-1.7B EAGLE-3 Unified Benchmark

RTX 5060 Ti 16GB; batch=16; input/output=128/128; greedy; K=3; 1 warmup + 3 measured runs.
- correctness: nano sequential verifier passed the separate token-level gate; packed benchmark output digests do not match its baseline because packed and tail batches use different BF16 GEMM shapes. vLLM speculative output digest does not match its baseline on this software stack.

| Case | Output tok/s (mean ± std) | TTFT (ms) | TPOT (ms) | Acceptance | Accept length | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_baseline | 1406.88 ± 0.95 | 137.68 | 10.38 | - | - | 5.10 |
| nano_eagle3_linear | 1124.60 ± 10.71 | 144.96 | 8.13 | 47.03% | 2.385 | 5.47 |
| nano_eagle3_tree | 248.26 ± 0.27 | 144.99 | 48.60 | 49.63% | 2.460 | 5.47 |
| vllm_baseline | 1366.24 ± 1.94 | 138.94 | 10.70 | - | - | 12.15 |
| vllm_eagle3 | 1291.15 ± 80.52 | 221.83 | 7.72 | 38.05% | 2.142 | 12.92 |

## Speedups

- nano linear / nano baseline: `0.799x`
- nano tree / nano baseline: `0.176x`
- vLLM EAGLE-3 / vLLM baseline: `0.945x`

## Resume-ready wording

在 nano-vLLM 中实现 EAGLE-3 双模型 KV Cache、跨 tokenizer vocabulary mapping 与 packed/tree proposal/target verification；构建固定 128/128、batch 16、3 次重复的正确性与性能基准，linear 路径实测 draft acceptance rate 47.0%、mean acceptance length 2.38，TPOT 8.13 ms （自回归 10.38 ms），吞吐为自回归基线的 0.80x，并定位 packed target forward 与尾批开销。
