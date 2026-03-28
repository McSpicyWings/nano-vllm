import os
import pickle
from time import perf_counter
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.llama_eagle3 import LlamaEagle3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._exited = False
        self.spec_debug = (
            config.draft_model is not None
            and rank == 0
            and os.getenv("NANOVLLM_SPEC_DEBUG", "").lower() not in ("", "0", "false")
        )
        self.spec_stats = dict(
            spec_calls=0,
            proposed_tokens=0,
            accepted_tokens=0,
            accepted_draft_tokens=0,
            seed_time=0.0,
            draft_time=0.0,
            verify_time=0.0,
            accept_time=0.0,
        )
        self.eagle_aux_hidden_state_layer_ids: list[int] | None = None

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = self._build_model(hf_config)
        load_model(self.model, config.model)
        # Optional draft (speculative) model for single-step eagle-like flow.
        self.draft_model = None
        if config.draft_hf_config is not None:
            self.draft_model = self._build_model(config.draft_hf_config)
            load_model(self.draft_model, config.draft_model)
            if hasattr(self.draft_model, "tie_input_embeddings"):
                self.draft_model.tie_input_embeddings(self.model.model.embed_tokens.weight)
            if hasattr(self.model, "get_eagle3_aux_hidden_state_layers"):
                self.eagle_aux_hidden_state_layer_ids = list(self.model.get_eagle3_aux_hidden_state_layers())
            else:
                num_layers = config.hf_config.num_hidden_layers
                self.eagle_aux_hidden_state_layer_ids = [1, num_layers // 2, num_layers - 4]
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache(self.model, hf_config)
        if self.draft_model is not None:
            self.allocate_kv_cache(self.draft_model, config.draft_hf_config)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                # Rank 0 uses shared memory to broadcast RPC-style commands to sibling processes.
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    # Tear down caches, CUDA graphs, and process group resources.
    def exit(self):
        if self._exited:
            return
        self._exited = True
        self._log_spec_stats()
        if self.world_size > 1:
            try:
                self.shm.close()
            except Exception:
                pass
            try:
                dist.barrier()
            except Exception:
                pass
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except Exception:
                    pass
        if not self.enforce_eager:
            try:
                del self.graphs, self.graph_pool
            except Exception:
                pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    # 处理model和draft model
    def _build_model(self, hf_config):
        architectures = getattr(hf_config, "architectures", []) or []
        model_type = getattr(hf_config, "model_type", "") or ""
        arch = architectures[0] if architectures else ""
        if arch == "LlamaForCausalLMEagle3" or model_type == "llama":
            return LlamaEagle3ForCausalLM(hf_config)
        return Qwen3ForCausalLM(hf_config)

    def _spec_sync(self):
        if self.spec_debug:
            torch.cuda.synchronize()

    def _log_spec_stats(self):
        if not self.spec_debug:
            return
        stats = self.spec_stats
        if stats["spec_calls"] == 0:
            return
        draft_acceptance = (
            stats["accepted_draft_tokens"] / stats["proposed_tokens"]
            if stats["proposed_tokens"] > 0 else 0.0
        )
        avg_tokens = stats["accepted_tokens"] / stats["spec_calls"]
        avg_draft = stats["accepted_draft_tokens"] / stats["spec_calls"]
        avg_seed_time = stats["seed_time"] / stats["spec_calls"]
        avg_draft_time = stats["draft_time"] / stats["spec_calls"]
        avg_verify_time = stats["verify_time"] / stats["spec_calls"]
        avg_accept_time = stats["accept_time"] / stats["spec_calls"]
        print(
            "[spec_debug] "
            f"calls={stats['spec_calls']} proposed={stats['proposed_tokens']} "
            f"accepted_draft={stats['accepted_draft_tokens']} accepted_total={stats['accepted_tokens']} "
            f"draft_acceptance={draft_acceptance:.6f} avg_tokens_per_call={avg_tokens:.3f} "
            f"avg_draft_per_call={avg_draft:.3f} avg_time_ms="
            f"(seed={avg_seed_time * 1000:.2f}, draft={avg_draft_time * 1000:.2f}, verify={avg_verify_time * 1000:.2f}, accept={avg_accept_time * 1000:.2f})"
        )

    # Worker-side loop for ranks > 0: wait for commands and execute them.
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    # Read one command payload from shared memory (nonzero ranks).
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # Write a command payload into shared memory (rank 0).
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # Invoke a local method and optionally broadcast the call to other ranks.
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # One-off warmup to compile kernels and measure allocator peak before cache allocation.
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # Run a dummy pass so kernels are compiled and allocator peaks can be measured before reserving KV cache.
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # Reserve KV cache tensors based on available memory and wire them into the model layers.
    def allocate_kv_cache(self, model: Qwen3ForCausalLM, hf_config):
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # KV cache block大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 能划分出的kvcache block数量 每个block 大小可以认为是 block_bytes, 内部还可以根据 k/v layer block_size num_kv_head head_dim 来切割
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 这里逻辑有点奇怪
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        if model is self.model:
            self.kv_cache = kv_cache
        else:
            self.draft_kv_cache = kv_cache

    # Build padded block tables for a batch of sequences.
    # 将一个 batch 中的 sequence 逻辑块拼成2维tensor, 用-1做尾部的掩码, 把长度统一, 
    # token_block = block_tables[seq_i][kv_block_j]
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # Flatten prompt tokens into contiguous buffers and set prefill context metadata.
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # Prefix cached tokens require block tables so FlashAttention can look them up lazily.
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # Collect the latest token per sequence and set decode context metadata.
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # Only a single token per sequence is executed, so we just gather the tip of each block table.
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    # Prepare a single prefill call that verifies pending + draft tokens.
    def prepare_spec_verify(self, seqs: list[Sequence], draft_tokens: torch.Tensor, k_list: list[int]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for b, seq in enumerate(seqs):
            base_pos = len(seq) - 1
            q_len = k_list[b] + 1
            k_len = len(seq) + k_list[b]
            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + k_len)
            max_seqlen_q = max(max_seqlen_q, q_len)
            max_seqlen_k = max(max_seqlen_k, k_len)
            input_ids.append(seq.last_token)
            if q_len > 1:
                input_ids.extend(draft_tokens[b, : q_len - 1].tolist())
            positions.extend(range(base_pos, base_pos + q_len))
            base_slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            slot_mapping.extend(range(base_slot, base_slot + q_len))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            prefill_last_only=False,
        )
        return input_ids, positions, cu_seqlens_q

    # Assemble temperatures tensor on rank 0 for sampling.
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model(input_ids, positions)
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        # For small decode batches we can reuse captured CUDA graphs to skip kernel launch overhead.
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["outputs"][:bs]

    @torch.inference_mode()
    # Execute the model; optionally reuse captured CUDA graphs for small decode batches.
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        hidden_states = self.run_model_hidden(input_ids, positions, is_prefill)
        return self.model.compute_logits(hidden_states)

    @torch.inference_mode()
    def run_draft_seed_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        use_aux_hidden = bool(getattr(self.draft_model, "use_aux_hidden_state", False))
        if use_aux_hidden:
            hidden_states, aux_hidden_states = self.model(
                input_ids,
                positions,
                return_aux_hidden_states=True,
                aux_hidden_state_layer_ids=self.eagle_aux_hidden_state_layer_ids,
            )
            return aux_hidden_states if aux_hidden_states.numel() else hidden_states
        return self.run_model_hidden(input_ids, positions, is_prefill=False)

    @torch.inference_mode()
    def prefill_draft_cache(self, seqs: list[Sequence]):
        if self.draft_model is None:
            return
        input_ids = []
        positions = []
        cu_seqlens = [0]
        max_seqlen = 0
        slot_mapping = []
        for seq in seqs:
            if not seq.block_table:
                continue
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens.append(cu_seqlens[-1] + seqlen)
            max_seqlen = max(max_seqlen, seqlen)
            for block_idx, block_id in enumerate(seq.block_table):
                start = block_id * self.block_size
                if block_idx != len(seq.block_table) - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(range(start, end))
        if not input_ids:
            return
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        draft_hidden = None
        if bool(getattr(self.draft_model, "use_aux_hidden_state", False)):
            set_context(
                True,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                slot_mapping,
                None,
                None,
            )
            _, aux_hidden_states = self.model(
                input_ids,
                positions,
                return_aux_hidden_states=True,
                aux_hidden_state_layer_ids=self.eagle_aux_hidden_state_layer_ids,
            )
            reset_context()
            if aux_hidden_states.numel():
                draft_hidden = self.draft_model.combine_hidden_states(aux_hidden_states)
        set_context(
            True,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            slot_mapping,
            None,
            None,
        )
        _ = self.draft_model(input_ids, positions, hidden_states=draft_hidden)
        reset_context()

    @torch.inference_mode()
    def fill_draft_cache_from_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        draft_hidden: torch.Tensor,
    ) -> None:
        if self.draft_model is None:
            return
        _ = self.draft_model(input_ids, positions, hidden_states=draft_hidden)

    @torch.inference_mode()
    def propose_draft_tokens(
        self,
        seqs: list[Sequence],
        k_list: list[int],
        temperatures: torch.Tensor | None,
        last_target_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor]:
        assert self.draft_model is not None
        bs = len(seqs)
        max_k = max(k_list) if k_list else 0
        draft_tokens = torch.empty((bs, max_k), dtype=torch.int64, device="cuda")
        draft_step_logits = [] if self.rank == 0 else None
        if max_k == 0:
            return draft_tokens[:, :0], draft_step_logits, last_target_hidden

        active_k = torch.tensor(k_list, dtype=torch.int32, device="cuda")
        base_len = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        base_pos = torch.tensor([len(seq) - 1 for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        base_slot = torch.tensor(
            [seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1 for seq in seqs],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        cur_tokens = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64, device="cuda")
        cur_hidden = self.draft_model.combine_hidden_states(last_target_hidden)
        final_hidden = cur_hidden

        for step in range(max_k):
            active = active_k > step
            slot = torch.where(active, base_slot + step, torch.full_like(base_slot, -1))
            context_lens = torch.where(active, base_len + step, base_len)
            positions = base_pos + step
            set_context(False, slot_mapping=slot, context_lens=context_lens, block_tables=block_tables)
            hidden_out, next_hidden = self.draft_model.forward_with_hidden(cur_tokens, positions, cur_hidden)
            logits = self.draft_model.compute_logits(hidden_out)
            if self.rank == 0:
                draft_step_logits.append(logits)
                next_draft_tokens = self.sampler(logits, temperatures)
                next_target_tokens = self.draft_model.map_draft_to_target(next_draft_tokens)
                next_target_tokens = torch.where(active, next_target_tokens, cur_tokens)
            else:
                next_target_tokens = torch.empty((bs,), dtype=torch.int64, device="cuda")
            if self.world_size > 1:
                dist.broadcast(next_target_tokens, src=0)
            draft_tokens[:, step] = next_target_tokens
            cur_tokens = next_target_tokens
            cur_hidden = torch.where(active[:, None], next_hidden, cur_hidden)
            final_hidden = torch.where(active[:, None], next_hidden, final_hidden)
            reset_context()
        return draft_tokens, draft_step_logits, final_hidden

    def _sample_from_probs(self, probs: torch.Tensor) -> int:
        token = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return int(token.item())

    def _sample_residual(self, p_probs: torch.Tensor, q_probs: torch.Tensor) -> int:
        residual = (p_probs - q_probs).clamp_min_(0)
        if residual.sum() <= 0:
            residual = p_probs
        return self._sample_from_probs(residual)

    # Accept/reject draft tokens using target logits (exact speculative decoding).
    @torch.inference_mode()
    def accept_draft_tokens(
        self,
        seqs: list[Sequence],
        draft_tokens: torch.Tensor,
        draft_step_logits: list[torch.Tensor],
        target_logits: torch.Tensor,
        temperatures: torch.Tensor,
        k_list: list[int],
        cu_q: torch.Tensor,
    ) -> tuple[list[list[int]], torch.Tensor]:
        bs = len(seqs)
        target_vocab_size = target_logits.size(-1)
        target_ids = self.draft_model.get_draft_target_ids(target_logits.device)
        valid_target_ids = (target_ids >= 0) & (target_ids < target_vocab_size)
        q_target_probs_steps = []
        for step_logits in draft_step_logits:
            q_probs = torch.softmax(step_logits.float() / temperatures[:, None], dim=-1)
            q_target_probs = torch.zeros((bs, target_vocab_size), device=q_probs.device, dtype=q_probs.dtype)
            indices = target_ids[valid_target_ids].unsqueeze(0).expand(bs, -1)
            q_target_probs.scatter_add_(1, indices, q_probs[:, valid_target_ids])
            q_target_probs_steps.append(q_target_probs)

        accept_lens = torch.zeros(bs, dtype=torch.int32, device=target_logits.device)
        out_tokens_per_seq: list[list[int]] = []
        for b in range(bs):
            accepted_tokens: list[int] = []
            q_len = k_list[b] + 1
            start = int(cu_q[b].item())
            end = int(cu_q[b + 1].item())
            seq_logits = target_logits[start:end]
            temp = float(temperatures[b].item())
            rejected = False
            for step in range(k_list[b]):
                y = draft_tokens[b, step]
                p_probs = torch.softmax(seq_logits[step].float() / temp, dim=-1)
                q_probs = q_target_probs_steps[step][b]
                q_y = q_probs[y].clamp_min_(1e-20)
                a = torch.minimum(torch.tensor(1.0, device=p_probs.device), p_probs[y] / q_y)
                if torch.rand((), device=p_probs.device) <= a:
                    accepted_tokens.append(int(y.item()))
                    accept_lens[b] += 1
                else:
                    z = self._sample_residual(p_probs, q_probs)
                    accepted_tokens.append(z)
                    rejected = True
                    break
            if not rejected:
                extra_probs = torch.softmax(seq_logits[q_len - 1].float() / temp, dim=-1)
                accepted_tokens.append(self._sample_from_probs(extra_probs))
            out_tokens_per_seq.append(accepted_tokens)
        return out_tokens_per_seq, accept_lens

    @torch.inference_mode()
    def finalize_draft_cache(
        self,
        seqs: list[Sequence],
        draft_tokens: torch.Tensor,
        final_hidden: torch.Tensor,
        accept_lens: torch.Tensor,
        k_list: list[int],
    ) -> None:
        if self.draft_model is None or draft_tokens.numel() == 0:
            return
        fin_mask = torch.tensor(
            [(int(accept_lens[i].item()) == k_list[i]) and (k_list[i] > 0) for i in range(len(seqs))],
            dtype=torch.bool,
            device="cuda",
        )
        if not fin_mask.any():
            return
        block_tables = self.prepare_block_tables(seqs)
        base_len = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        base_pos = torch.tensor([len(seq) - 1 for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        base_slot = torch.tensor(
            [seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1 for seq in seqs],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        k_tensor = torch.tensor(k_list, dtype=torch.int64, device="cuda")
        last_draft_tokens = torch.tensor(
            [int(draft_tokens[i, k_list[i] - 1].item()) if k_list[i] > 0 else 0 for i in range(len(seqs))],
            dtype=torch.int64,
            device="cuda",
        )
        slot = torch.where(fin_mask, base_slot + k_tensor.to(torch.int32), torch.full_like(base_slot, -1))
        context_lens = torch.where(fin_mask, base_len + k_tensor.to(torch.int32), base_len)
        positions = base_pos + k_tensor
        set_context(False, slot_mapping=slot, context_lens=context_lens, block_tables=block_tables)
        _ = self.draft_model.forward_with_hidden(last_draft_tokens, positions, final_hidden)
        reset_context()

    # End-to-end run: prepare inputs, run model, sample tokens (on rank 0), and reset context.
    @torch.inference_mode()
    def run_baseline(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]] | None:
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        fused_prefill = False
        if is_prefill and self.draft_model is not None and all(seq.num_cached_tokens == 0 for seq in seqs):
            if bool(getattr(self.draft_model, "use_aux_hidden_state", False)):
                hidden_states, aux_hidden_states = self.model(
                    input_ids,
                    positions,
                    return_aux_hidden_states=True,
                    aux_hidden_state_layer_ids=self.eagle_aux_hidden_state_layer_ids,
                )
                draft_hidden = aux_hidden_states if aux_hidden_states.numel() else hidden_states
                draft_hidden = self.draft_model.combine_hidden_states(draft_hidden)
            else:
                hidden_states = self.model(input_ids, positions)
                draft_hidden = hidden_states
            logits = self.model.compute_logits(hidden_states)
            self.fill_draft_cache_from_prefill(input_ids, positions, draft_hidden)
            fused_prefill = True
        else:
            logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        if self.draft_model is not None and is_prefill and not fused_prefill:
            self.prefill_draft_cache(seqs)
        if token_ids is None:
            return None
        return [[t] for t in token_ids]

    @torch.inference_mode()
    def run_spec_decode(self, seqs: list[Sequence]) -> list[list[int]] | None:
        assert self.draft_model is not None
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        k_list = []
        for seq in seqs:
            remaining = seq.max_tokens - seq.num_completion_tokens
            k = min(getattr(self.config, "num_spec_tokens", 0), max(0, remaining - 1))
            k = min(k, self.block_size - seq.last_block_num_tokens)
            k_list.append(k)

        input_ids, positions = self.prepare_decode(seqs)
        if self.spec_debug:
            self._spec_sync()
            t0 = perf_counter()
        last_target_hidden = self.run_draft_seed_hidden(input_ids, positions)
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["seed_time"] += perf_counter() - t0
        reset_context()

        if self.spec_debug:
            self.spec_stats["spec_calls"] += 1
            self._spec_sync()
            t0 = perf_counter()
        draft_tokens, draft_step_logits, final_hidden = self.propose_draft_tokens(
            seqs,
            k_list,
            temperatures,
            last_target_hidden,
        )
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["draft_time"] += perf_counter() - t0
            self.spec_stats["proposed_tokens"] += sum(k_list)
        input_ids, positions, cu_q = self.prepare_spec_verify(seqs, draft_tokens, k_list)
        if self.spec_debug:
            self._spec_sync()
            t0 = perf_counter()
        target_logits = self.run_model(input_ids, positions, is_prefill=True)
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["verify_time"] += perf_counter() - t0
        reset_context()

        if self.rank != 0:
            accept_lens = torch.empty(len(seqs), dtype=torch.int32, device="cuda")
            if self.world_size > 1:
                dist.broadcast(accept_lens, src=0)
            self.finalize_draft_cache(seqs, draft_tokens, final_hidden, accept_lens, k_list)
            return None
        if self.spec_debug:
            self._spec_sync()
            t0 = perf_counter()
        token_ids, accept_lens = self.accept_draft_tokens(
            seqs,
            draft_tokens,
            draft_step_logits,
            target_logits,
            temperatures,
            k_list,
            cu_q,
        )
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["accept_time"] += perf_counter() - t0
            self.spec_stats["accepted_tokens"] += sum(len(toks) for toks in token_ids)
            self.spec_stats["accepted_draft_tokens"] += sum(max(len(toks) - 1, 0) for toks in token_ids)
        if self.world_size > 1:
            dist.broadcast(accept_lens, src=0)
        self.finalize_draft_cache(seqs, draft_tokens, final_hidden, accept_lens, k_list)
        return token_ids

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]] | None:
        if is_prefill or self.draft_model is None or self.config.num_spec_tokens <= 0:
            return self.run_baseline(seqs, is_prefill)
        return self.run_spec_decode(seqs)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
