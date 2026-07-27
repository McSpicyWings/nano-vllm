# nano-vLLM EAGLE-3 Benchmark

- workload: batch=16, input=128, output=128, greedy, repeats=3
- workload SHA256: `6cc3bda376604d06b504f2f16d1c1abe8bd36dc64d3518bf7b14391d5e1790e7`

| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_baseline | 1406.88 | 137.68 | 10.38 | - | - | 5.10 |
| nano_eagle3_linear | 1124.60 | 144.96 | 8.13 | 47.03% | 2.385 | 5.47 |
| nano_eagle3_tree | 248.26 | 144.99 | 48.60 | 49.63% | 2.460 | 5.47 |
