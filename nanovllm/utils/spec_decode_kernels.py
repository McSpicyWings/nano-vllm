import torch
import triton
import triton.language as tl


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,
    valid_sampled_tokens_count_ptr,
    query_start_loc_gpu_ptr,
    token_indices_to_sample_ptr,
    num_reqs,
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)
    cu_draft_prev = tl.where(
        req_idx == 0,
        tl.zeros_like(cu_draft_curr),
        tl.load(cu_num_draft_tokens_ptr + req_idx - 1),
    )
    num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1
    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,
    discard_request_mask_ptr,
    backup_next_token_ids_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride_sampled_token_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    is_discarded = tl.load(discard_request_mask_ptr + req_idx)
    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
        return

    token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
    token_mask = token_offs < num_sampled_tokens_per_req

    row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
    token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

    is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
    valid_count = tl.sum(is_valid_mask)

    if valid_count > 0:
        last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))
        last_valid_token = tl.sum(tl.where(token_offs == last_valid_index, token_ids, 0))
        tl.store(next_token_ids_ptr + req_idx, last_valid_token)
    else:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        tl.store(next_token_ids_ptr + req_idx, backup_token)

    tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def prepare_next_token_ids_padded(
    sampled_token_ids: torch.Tensor,
    vocab_size: int,
    discard_request_mask: torch.Tensor | None = None,
    backup_next_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_tokens = sampled_token_ids.shape
    device = sampled_token_ids.device
    if discard_request_mask is None:
        discard_request_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if backup_next_token_ids is None:
        backup_next_token_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)
    next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
    valid_sampled_tokens_count = torch.empty(batch_size, dtype=torch.int32, device=device)
    grid = (batch_size,)
    block_size_tokens = triton.next_power_of_2(num_tokens)
    eagle_prepare_next_token_padded_kernel[grid](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size,
        num_tokens,
        batch_size,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )
    return next_token_ids, valid_sampled_tokens_count


def prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    num_reqs = valid_sampled_tokens_count.numel()
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=valid_sampled_tokens_count.device)
    grid = (num_reqs,)
    eagle_prepare_inputs_padded_kernel[grid](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        token_indices_to_sample,
        num_reqs,
    )
    return token_indices_to_sample
