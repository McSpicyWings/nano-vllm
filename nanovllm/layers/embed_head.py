import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


def compute_logits_with_mapping(
    draft_logits: torch.Tensor,
    draft_vocab_size: int,
    target_vocab_size: int,
    vocab_mapping: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Map draft model logits to target model vocab space.
    
    Args:
        draft_logits: [N, draft_vocab] logits from draft model
        draft_vocab_size: vocab size of the draft model
        target_vocab_size: vocab size of the target model
        vocab_mapping: Optional [draft_vocab] tensor mapping draft token ids to target token ids.
                       If None, assumes identity mapping for overlapping tokens.
    
    Returns:
        target_logits: [N, target_vocab] with -inf for unmapped positions
    """
    N = draft_logits.size(0)
    device = draft_logits.device
    dtype = draft_logits.dtype
    
    # Initialize with -inf (tokens not covered by draft model are impossible)
    target_logits = torch.full(
        (N, target_vocab_size),
        float('-inf'),
        device=device,
        dtype=dtype,
    )
    
    if vocab_mapping is not None:
        # Use explicit mapping: vocab_mapping[draft_idx] = target_idx
        # -1 in mapping means no corresponding target token
        valid_mask = vocab_mapping >= 0
        valid_draft_indices = torch.arange(draft_vocab_size, device=device)[valid_mask]
        valid_target_indices = vocab_mapping[valid_mask]
        target_logits[:, valid_target_indices] = draft_logits[:, valid_draft_indices]
    else:
        # Identity mapping: draft token i maps to target token i
        # Only map up to min(draft_vocab, target_vocab)
        overlap_size = min(draft_vocab_size, target_vocab_size)
        target_logits[:, :overlap_size] = draft_logits[:, :overlap_size]
    
    return target_logits


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        # Prefill returns all logits when prefill_last_only is False (needed for spec verify).
        if context.is_prefill and context.prefill_last_only:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
