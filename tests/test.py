from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

llm = LLM(model="/home/wpc/huggingface/Qwen3-1.7B",
          draft_model="/home/wpc/huggingface/AngelSlim/Qwen3-1.7B_eagle3",
          max_model_len=2048,
          tensor_parallel_size=1)


print("target model:", type(llm.model_runner.model))
print("draft model:", type(llm.model_runner.draft_model))
model = llm.model_runner.model
print("draft layer0 qkv_proj weight shape:",
      model.model.layers[0].self_attn.qkv_proj.weight.shape)

draft = llm.model_runner.draft_model
print("draft layer0 qkv_proj weight shape:",
      draft.model.layers[0].self_attn.qkv_proj.weight.shape)
out = llm.generate(
    ["新年好！"],
    SamplingParams(max_tokens=100, temperature=1),
    use_tqdm=False,
)
print(out)