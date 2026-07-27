# nano-vLLM EAGLE-3 Benchmark

- workload: batch=16, input=128, output=32, greedy, repeats=1
- workload SHA256: `6cc3bda376604d06b504f2f16d1c1abe8bd36dc64d3518bf7b14391d5e1790e7`

| Case | Output tok/s | TTFT mean (ms) | TPOT mean (ms) | Acceptance | Accept length | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| nano_eagle3_tree | 235.68 | 145.78 | 62.91 | 2.64% | 1.074 | 5.46 |
