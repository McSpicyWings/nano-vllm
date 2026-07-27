# nano-vLLM EAGLE-3 Benchmark

- workload: batch=16, input=128, output=128, greedy, repeats=3
- speculative tail fallback: active batch < 16 switches to baseline decode
- workload SHA256: `6cc3bda376604d06b504f2f16d1c1abe8bd36dc64d3518bf7b14391d5e1790e7`

| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_baseline | 1407.53 | 137.70 | 10.37 | - | - | 5.10 |
| nano_eagle3_linear | 1337.26 | 145.46 | 7.56 | 64.51% | 2.928 | 5.47 |
| nano_eagle3_tree | 363.41 | 145.67 | 40.18 | 65.34% | 2.959 | 5.47 |
