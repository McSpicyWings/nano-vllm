# Qwen3-1.7B EAGLE-3 Unified Benchmark

RTX 5060 Ti 16GB; batch=16; input/output=128/128; greedy; K=3; spec tail fallback<16; 1 warmup + 3 measured runs.
- correctness: nano sequential verifier passed the separate token-level gate; packed benchmark output digests do not match its baseline because packed and tail batches use different BF16 GEMM shapes. vLLM speculative output digest does not match its baseline on this software stack.

| Case | Output tok/s (mean ± std) | TTFT (ms) | TPOT (ms) | Acceptance | Accept length | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_baseline | 1407.53 ± 0.77 | 137.70 | 10.37 | - | - | 5.10 |
| nano_eagle3_linear | 1337.26 ± 1.01 | 145.46 | 7.56 | 64.51% | 2.928 | 5.47 |
| nano_eagle3_tree | 363.41 ± 1.83 | 145.67 | 40.18 | 65.34% | 2.959 | 5.47 |
| vllm_baseline | 1366.24 ± 1.94 | 138.94 | 10.70 | - | - | 12.15 |
| vllm_eagle3 | 1291.15 ± 80.52 | 221.83 | 7.72 | 38.05% | 2.142 | 12.92 |

## Speedups

- nano linear / nano baseline: `0.950x`
- nano tree / nano baseline: `0.258x`
- vLLM EAGLE-3 / vLLM baseline: `0.945x`

## Resume-ready wording

在 nano-vLLM 中实现 EAGLE-3 双模型 KV Cache、跨 tokenizer vocabulary mapping 与 packed/tree proposal/target verification，并增加小 batch 自适应 AR 回退；构建固定 128/128、batch 16、3 次重复的正确性与性能基准，linear 投机阶段实测 draft acceptance rate 64.5%、mean acceptance length 2.93，TPOT 7.56 ms （自回归 10.37 ms），吞吐为自回归基线的 0.95x，与同负载 vLLM 的 0.95x 相当。
