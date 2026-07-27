# Legacy nano-vLLM benchmark

`bench.log` 是旧版 `bench_spec_decode.py` 的单次输出，当时 nano baseline 约为
1250 tok/s，EAGLE-3 约为 381 tok/s，即 `0.30x`。它没有使用当前固定 workload、
完整词表 verifier、严格输出一致性检查或三次聚合，因此不能和正式结果直接比较。

正式结果见 [`../results/rtx5060ti_qwen3_1_7b`](../results/rtx5060ti_qwen3_1_7b)。
