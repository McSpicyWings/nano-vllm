import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _expand_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    assert num_heads % x.size(1) == 0
    return x.repeat_interleave(num_heads // x.size(1), dim=1)


def _gather_prefix_cache(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    if prefix_len <= 0:
        return cache.new_empty((0, cache.size(2), cache.size(3)))
    block_size = cache.size(1)
    positions = torch.arange(prefix_len, device=cache.device, dtype=torch.int64)
    block_indices = positions // block_size
    block_ids = block_table_row.index_select(0, block_indices)
    slots = block_ids.to(torch.int64) * block_size + (positions % block_size)
    flat_cache = cache.flatten(0, 1)
    return flat_cache.index_select(0, slots)


def _tree_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_heads: int,
    scale: float,
) -> torch.Tensor:
    context = get_context()
    assert context.cu_seqlens_q is not None and context.cu_seqlens_k is not None
    assert context.tree_attn_bias is not None
    batch_size = context.cu_seqlens_q.numel() - 1
    tree_attn_bias = context.tree_attn_bias
    block_tables = context.block_tables
    q_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
    k_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
    max_q_len = int(q_lens.max().item())
    max_k_len = int(k_lens.max().item())
    q_batch = q.new_zeros((batch_size, num_heads, max_q_len, q.size(-1)))
    k_batch = k.new_zeros((batch_size, num_heads, max_k_len, k.size(-1)))
    v_batch = v.new_zeros((batch_size, num_heads, max_k_len, v.size(-1)))
    attn_bias = q.new_full((batch_size, 1, max_q_len, max_k_len), float("-inf"))
    for b in range(batch_size):
        q_start = int(context.cu_seqlens_q[b].item())
        q_end = int(context.cu_seqlens_q[b + 1].item())
        k_start = int(context.cu_seqlens_k[b].item())
        k_end = int(context.cu_seqlens_k[b + 1].item())
        q_len = q_end - q_start
        k_len = k_end - k_start
        prefix_len = k_len - q_len

        q_seq = q[q_start:q_end].permute(1, 0, 2)
        k_seq = k[q_start:q_end]
        v_seq = v[q_start:q_end]
        if prefix_len > 0:
            assert block_tables is not None
            prefix_k = _gather_prefix_cache(k_cache, block_tables[b], prefix_len)
            prefix_v = _gather_prefix_cache(v_cache, block_tables[b], prefix_len)
            k_seq = torch.cat([prefix_k, k_seq], dim=0)
            v_seq = torch.cat([prefix_v, v_seq], dim=0)
        k_seq = _expand_kv_heads(k_seq, num_heads).permute(1, 0, 2)
        v_seq = _expand_kv_heads(v_seq, num_heads).permute(1, 0, 2)
        q_batch[b, :, :q_len] = q_seq
        k_batch[b, :, :k_len] = k_seq
        v_batch[b, :, :k_len] = v_seq

        seq_bias = tree_attn_bias[:q_len, :q_len].to(dtype=q.dtype)
        if prefix_len > 0:
            attn_bias[b, 0, :q_len, :prefix_len] = 0
        attn_bias[b, 0, :q_len, prefix_len:k_len] = seq_bias
        if q_len < max_q_len:
            attn_bias[b, 0, q_len:, 0] = 0
    out = F.scaled_dot_product_attention(
        q_batch,
        k_batch,
        v_batch,
        attn_mask=attn_bias,
        dropout_p=0.0,
        scale=scale,
    )
    outputs = [
        out[b, :, : int(q_lens[b].item())].permute(1, 0, 2)
        for b in range(batch_size)
    ]
    return torch.cat(outputs, dim=0)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.tree_attn_bias is not None:
                o = _tree_attn_prefill(q, k, v, k_cache, v_cache, self.num_heads, self.scale)
            elif context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
