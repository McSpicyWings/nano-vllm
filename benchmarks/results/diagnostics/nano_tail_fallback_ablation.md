# Sticky tail fallback ablation

RTX 5060 Ti 16GB; Qwen3-1.7B; batch=16; input/output=128/128; greedy;
EAGLE-3 K=3. Each threshold run used one warmup. Threshold means that an
active batch smaller than the value permanently switches those requests to
ordinary autoregressive decoding.

| Threshold | Measured runs | Output tok/s | Ratio vs 1407.53 tok/s baseline |
|---:|---:|---:|---:|
| 1 (disabled) | 3 | 1124.60 ± 10.71 | 0.799x |
| 8 | 1 | 1278.43 | 0.908x |
| 12 | 1 | 1314.34 | 0.934x |
| 16 | 1 | 1340.11 | 0.952x |
| 16 (formal rerun) | 3 | 1337.26 ± 1.01 | 0.950x |

The monotonic single-run ablation and the three-run confirmation show that the
remaining end-to-end loss was dominated by low-utilization speculative tail
batches. Acceptance metrics in the thresholded runs cover only actual
speculative proposals; autoregressive fallback tokens are excluded.
