import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    # Optional path to a draft (speculative) model; when set we run a single-step drafter.
    draft_model: str | None = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    draft_hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Single-step speculative tokens; keep 1 for MVP.
    num_spec_tokens: int = 5
    speculative_token_tree: str | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if self.draft_model is not None:
            assert os.path.isdir(self.draft_model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.draft_model is not None:
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
