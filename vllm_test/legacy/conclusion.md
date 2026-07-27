# vLLM Eagle3 本地验证结论

## 背景

本目录的脚本用于验证在本地 `conda activate vllm` 环境下，
`Qwen3-1.7B + Qwen3-1.7B_eagle3` 这组模型在 vLLM 中启用 Eagle3 speculative decoding 后，
是否能够带来正向吞吐收益。

相关脚本与结果文件：

- `vllm_test/bench_vllm_eagle3.py`
- `vllm_test/result_t0_local.json`
- `vllm_test/result_t1_local.json`
- `vllm_test/smoke_t0.json`
- `vllm_test/spec_fit_probe.json`

## 核心结论

结论很直接：

1. 在当前本地机器和当前测试条件下，我没有复现出 vLLM Eagle3 的正收益。
2. 默认更激进的 spec 配置在本地 15.46GB 显存卡上会 OOM，连初始化都无法稳定通过。
3. 在调整到可稳定运行的本地安全配置后，Eagle3 可以正常跑通，但吞吐仍然低于 baseline。
4. 因此，“Eagle3 在 vLLM 中理论上能加速”这件事，与“它在当前本地环境里已经能稳定正优化”是两回事；就本地实测而言，答案是否定的。

## 本地可运行配置

为了让 vLLM Eagle3 在当前机器上稳定运行，测试时采用了以下配置：

- `gpu_memory_utilization = 0.7`
- `max_model_len = 2048`
- `num_spec_tokens = 3`
- `tensor_parallel_size = 1`
- `enable_chunked_prefill = true`

另外，在 `temperature = 0` 的场景下，脚本会自动把 `top_k` 收成 `1`。
这和 greedy 语义等价，但可以避免 speculative rejection sampler 在初始化时做全 vocab sort，
从而减少 OOM 风险。

## 关键结果

### T = 0

结果文件：`vllm_test/result_t0_local.json`

- baseline output throughput: `1524.97 tok/s`
- Eagle3 output throughput: `1220.91 tok/s`
- speedup: `0.80x`
- mean acceptance length: `1.565`
- draft acceptance rate: `18.8%`

### T = 1

结果文件：`vllm_test/result_t1_local.json`

- baseline output throughput: `1511.45 tok/s`
- Eagle3 output throughput: `1030.00 tok/s`
- speedup: `0.68x`
- mean acceptance length: `1.313`
- draft acceptance rate: `10.4%`

## 如何解读这个结果

这组结果说明：

1. 在本地 vLLM 上，Eagle3 的 acceptance 并不高。
2. `T=1` 时 acceptance 进一步下降，导致 speculative decoding 更难摊平 verifier 成本。
3. 即使在 `T=0` 下，当前本地配置也只有 `0.80x`，仍然是负优化。
4. 所以本地没有观察到“启用 Eagle3 后吞吐显著提升”的现象。

## 与模型卡结果的关系

`AngelSlim/Qwen3-1.7B_eagle3` 的模型卡给出了大约 `1.9x ~ 2.0x` 的 speedup。
本地没有复现到该结果，更可能说明测试条件不同，而不是单纯说明模型卡错误。

可能的差异包括：

- 本地显存更小，需要压缩配置
- 本地使用的是一组小规模自然语言 prompt
- 本地不是模型卡中的 MT-bench / HumanEval / GSM8K / Alpaca 场景
- 本地 acceptance 明显低于模型卡隐含对应的水平

## 最终判断

截至当前验证结论：

1. Eagle3 在 vLLM 中“理论上可能有收益”这件事不能直接否定。
2. 但在当前本地机器、当前显存约束、当前测试配置下，没有拿到正收益。
3. 因而不能把“vLLM 官方支持 Eagle3”直接当作“本地已经验证可稳定加速”的证据。
4. 如果后续要继续追查，应该优先从更贴近官方口径的数据集、更大显存环境、以及 acceptance 差异入手。
