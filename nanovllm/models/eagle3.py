import torch
from torch import nn

from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.models.qwen3 import Qwen3DecoderLayer


class Eagle3Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.embed_tokens = VocabParallelEmbedding(draft_vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Eagle3ForCausalLM(nn.Module):
    """A minimal Eagle3 draft model (Step2: load + forward + propose).

    Notes:
      - Extra checkpoint keys are ignored by using load_model(..., strict=False).
      - Step3 will extend this with Eagle3-specific feature fusion if needed.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.model = Eagle3Model(config)
        self.lm_head = ParallelLMHead(draft_vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

        # Token-id mapping buffers (expected checkpoint keys: "t2d" and "d2t").
        target_vocab_size = getattr(config, "vocab_size", draft_vocab_size)
        self.register_buffer("d2t", torch.arange(draft_vocab_size, dtype=torch.int64), persistent=False)
        self.register_buffer("t2d", torch.full((target_vocab_size,), -1, dtype=torch.int64), persistent=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def map_target_to_draft(self, target_token_ids: torch.Tensor) -> torch.Tensor:
        """Map target vocab ids -> draft vocab ids. Unknown ids -> 0."""
        draft_ids = self.t2d[target_token_ids]
        if draft_ids.dtype != torch.int64:
            draft_ids = draft_ids.to(torch.int64)
        draft_ids = torch.where(draft_ids >= 0, draft_ids, torch.zeros_like(draft_ids))
        return draft_ids

    def map_draft_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Map draft vocab ids -> target vocab ids."""
        return self.d2t[draft_token_ids]