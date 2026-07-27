# vLLM EAGLE-3 Benchmark

- workload: batch=16, input=128, output=128, greedy, repeats=3
- workload SHA256: `6cc3bda376604d06b504f2f16d1c1abe8bd36dc64d3518bf7b14391d5e1790e7`

| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Device peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| vllm_baseline | 1366.24 | 138.94 | 10.70 | - | - | 12.15 |
| vllm_eagle3 | 1291.15 | 221.83 | 7.72 | 38.05% | 2.142 | 12.92 |
