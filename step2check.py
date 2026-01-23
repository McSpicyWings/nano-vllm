from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

llm = LLM(
    model="./huggingface/Qwen3-1.7B",
    draft_model="./huggingface/AngelSlim/Qwen3-1.7B_eagle3",
    speculative_k=8,              # draft propose 8 tokens（Step2 只 propose）
    tensor_parallel_size=1,
)

out = llm.generate(["hello"], SamplingParams(max_tokens=32, temperature=0.8))
print(out[0]["text"])
