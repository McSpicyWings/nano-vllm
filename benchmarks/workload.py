from __future__ import annotations

import hashlib
import json


PROMPT_BANK = [
    "Explain the practical difference between throughput and latency for large language model inference.",
    "Write a concise note about why KV cache layout matters during transformer decoding.",
    "Summarize how speculative decoding trades verifier cost against acceptance rate.",
    "Describe one reason CUDA graphs can improve small-batch decode performance.",
    "Explain why a draft model with low acceptance can become a negative optimization.",
    "Give a concise explanation of tree attention in speculative decoding.",
    "Write a short paragraph about why Python-side tensor assembly can become a hot path.",
    "Explain why benchmark prompts should resemble natural language instead of random token ids.",
    "Describe how greedy decoding changes speculative acceptance.",
    "Explain why verifier-side target forwards often dominate speculative decoding latency.",
    "Explain the difference between prefill throughput and decode throughput in one paragraph.",
    "Describe how fixed-length workloads improve benchmark reproducibility.",
    "Write a short answer explaining why acceptance length matters more than raw draft speed.",
    "Explain why model-pair compatibility is necessary but not sufficient for speculative speedup.",
    "Give a brief note on why attention metadata can become a bottleneck in custom runtimes.",
    "Summarize why exactness constraints can reduce speculative decoding performance.",
]

DEFAULT_TREE = str(
    [(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)]
)


def build_fixed_length_prompts(tokenizer, batch_size: int, input_length: int) -> list[list[int]]:
    if batch_size <= 0 or input_length <= 0:
        raise ValueError("batch_size and input_length must be positive")
    prompts = []
    for index in range(batch_size):
        source = PROMPT_BANK[index % len(PROMPT_BANK)]
        token_ids = tokenizer.encode(source, add_special_tokens=False)
        continuation = (
            " Provide a technically precise answer and include the most important implementation tradeoff."
        )
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
        while len(token_ids) < input_length:
            token_ids.extend(continuation_ids)
            token_ids.extend(tokenizer.encode(" " + source, add_special_tokens=False))
        prompts.append(token_ids[:input_length])
    return prompts


def workload_digest(prompt_token_ids: list[list[int]]) -> str:
    payload = json.dumps(prompt_token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
