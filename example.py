import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    tpath = os.path.expanduser("./huggingface/Qwen3-1.7B")
    dpath = os.path.expanduser("./huggingface/AngelSlim/Qwen3-1.7B_eagle3")
    tokenizer = AutoTokenizer.from_pretrained(tpath)
    llm = LLM(
        tpath,
        draft_model=dpath,
        tensor_parallel_size=1,
        max_model_len=2048,
        num_spec_tokens=3,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=128)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    try:
        outputs = llm.generate(prompts, sampling_params, return_metrics=True)
        for prompt, output in zip(prompts, outputs):
            print("\n")
            print(f"Prompt: {prompt!r}")
            print(f"Completion: {output['text']!r}")
            print(f"Metrics: {output['metrics']}")
        print(f"Speculative stats: {llm.get_spec_stats()}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
