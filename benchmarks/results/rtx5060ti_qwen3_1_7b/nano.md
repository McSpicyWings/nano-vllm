# nano-vLLM EAGLE-3 Benchmark

- workload: batch=16, input=128, output=128, greedy, repeats=3
- workload SHA256: `6cc3bda376604d06b504f2f16d1c1abe8bd36dc64d3518bf7b14391d5e1790e7`

| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_baseline | 1398.54 | 137.68 | 10.44 | - | - | 5.10 |
| nano_eagle3_linear | 287.59 | 144.97 | 52.92 | 1.66% | 1.049 | 5.46 |
| nano_eagle3_tree | 212.83 | 145.42 | 69.99 | 4.89% | 1.144 | 5.46 |
