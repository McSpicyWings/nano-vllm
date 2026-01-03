from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

# 直接使用 Eagle3 模型作为主模型，不启用单步草稿
# 请将路径替换为你的 Eagle3 权重所在目录（需包含 config.json 与 *.safetensors）
EAGLE3_MODEL_DIR = "/home/wpc/huggingface/AngelSlim/Qwen3-1.7B_eagle3"

if __name__ == "__main__":
    llm = LLM(
        model=EAGLE3_MODEL_DIR,
        max_model_len=2048,
        tensor_parallel_size=1,
    )

    print("loaded model:", type(llm.model_runner.model))
    print(
        "layer0 qkv_proj weight shape:",
        llm.model_runner.model.model.layers[0].self_attn.qkv_proj.weight.shape,
    )

    outputs = llm.generate(
        ["说我爱你一百次"],
        SamplingParams(max_tokens=64, temperature=1.0),
        use_tqdm=False,
    )
    print(outputs)
