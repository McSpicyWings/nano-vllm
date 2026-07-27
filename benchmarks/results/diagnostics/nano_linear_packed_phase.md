# nano-vLLM EAGLE-3 Benchmark

- workload: batch=16, input=128, output=32, greedy, repeats=1
- workload SHA256: `6cc3bda376604d06b504f2f16d1c1abe8bd36dc64d3518bf7b14391d5e1790e7`

| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_eagle3_linear | 860.76 | 144.62 | 8.56 | 62.82% | 2.756 | 5.47 |
