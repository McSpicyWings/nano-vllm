import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead, compute_logits_with_mapping


class LlamaEagle3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        qkv_input_size: int | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            qkv_input_size or hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class LlamaEagle3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class LlamaEagle3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        qkv_input_size = config.hidden_size * 2 if layer_idx == 0 else config.hidden_size
        self.self_attn = LlamaEagle3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            qkv_input_size=qkv_input_size,
        )
        self.mlp = LlamaEagle3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        self.norm_before_residual = getattr(config, "norm_before_residual", False)

    def _norm_before_residual(self, hidden_states: torch.Tensor):
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            embeds = self.input_layernorm(embeds)
            norm_fn = self._norm_before_residual if self.norm_before_residual else self._norm_after_residual
            hidden_states, residual = norm_fn(hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaEagle3Model(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaEagle3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.embed_input_ids(input_ids)
        hidden_states = embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, embeds, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaEagle3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.model = LlamaEagle3Model(config)
        # Draft vocab size may differ from target vocab size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.target_vocab_size = getattr(config, "target_vocab_size", config.vocab_size)
        self.lm_head = ParallelLMHead(self.draft_vocab_size, config.hidden_size)
        tie_embeddings = getattr(config, "tie_word_embeddings", True) and self.draft_vocab_size == config.vocab_size
        if tie_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        
        # fc layer for Eagle3 speculator: fuses (prev_hidden, curr_hidden, embed) -> hidden
        # Input: 3 * hidden_size (concat of prev_hidden + curr_hidden + embed)
        # Output: hidden_size
        self.fc = ReplicatedLinear(
            input_size=3 * config.hidden_size,
            output_size=config.hidden_size,
            bias=False,
        )
        
        # Optional: vocab mapping tensor for draft->target vocab (loaded from config or file)
        self.vocab_mapping: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)
    
    def forward_with_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        prev_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that accepts previous hidden states for multi-step speculative drafting.
        
        Args:
            input_ids: [N] input token ids
            positions: [N] position indices
            prev_hidden_states: [N, H] hidden states from previous step (None for first step)
        
        Returns:
            hidden_states: [N, H] output hidden states
            fused_hidden: [N, H] fused hidden states (for passing to next step)
        """
        embeds = self.model.embed_input_ids(input_ids)
        hidden_states = embeds
        
        if prev_hidden_states is not None:
            # Fuse previous hidden, current hidden (embeds initially), and embed
            # using the fc layer: [prev_hidden, curr_hidden, embed] -> fused
            fused_input = torch.cat([prev_hidden_states, hidden_states, embeds], dim=-1)
            hidden_states = self.fc(fused_input)
        
        residual = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, embeds, hidden_states, residual)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        
        return hidden_states, hidden_states  # Return hidden_states as both output and state for next step

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits in draft vocab space."""
        logits = self.lm_head(hidden_states)
        return logits
    
    def compute_logits_mapped(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits and map them to target vocab space.
        Used during speculative decoding verification.
        
        Args:
            hidden_states: [N, H] hidden states
        
        Returns:
            target_logits: [N, target_vocab] logits in target vocab space
        """
        draft_logits = self.lm_head(hidden_states)
        if draft_logits is None:  # Non-rank-0 in TP
            return None
        
        # If draft and target vocab are the same, no mapping needed
        if self.draft_vocab_size == self.target_vocab_size and self.vocab_mapping is None:
            return draft_logits
        
        # Map draft logits to target vocab space
        target_logits = compute_logits_with_mapping(
            draft_logits,
            self.draft_vocab_size,
            self.target_vocab_size,
            self.vocab_mapping,
        )
        return target_logits
    
    def set_vocab_mapping(self, mapping: torch.Tensor):
        """
        Set the vocab mapping tensor for draft->target vocabulary.
        
        Args:
            mapping: [draft_vocab] tensor where mapping[i] = target token id for draft token i
                     Use -1 to indicate no mapping exists.
        """
        self.vocab_mapping = mapping