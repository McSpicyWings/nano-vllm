import os
import ast
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
from nanovllm.utils.spec_decode_kernels import (
    prepare_inputs_padded,
    prepare_next_token_ids_padded,
)


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
            sequence_proposals=0,
            proposed_tokens=0,
            accepted_draft_tokens=0,
            emitted_tokens=0,
            seed_time=0.0,
            draft_time=0.0,
            verify_time=0.0,
            accept_time=0.0,
        )
        self.eagle_aux_hidden_state_layer_ids: list[int] | None = None
        self.seq_prev_hidden: dict[int, torch.Tensor] = {}
        self.spec_token_tree = self._build_spec_token_tree()
        self.spec_tree_leaf_paths = self._get_tree_leaf_paths(self.spec_token_tree)
        self.spec_tree_nodes_by_level = self._get_tree_nodes_by_level(self.spec_token_tree)
        self.spec_tree_children = self._get_tree_children(self.spec_token_tree)
        self.spec_tree_path_to_index = {path: idx for idx, path in enumerate(self.spec_token_tree)}
        self.spec_tree_depth = max((len(path) for path in self.spec_token_tree), default=0)
        self.spec_tree_attn_bias = self._build_tree_attn_bias(self.spec_token_tree)

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
        # Warmup sequences are synthetic and never pass through Scheduler cleanup.
        self.seq_prev_hidden.clear()
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

    def _build_spec_token_tree(self) -> list[tuple[int, ...]]:
        tree_str = getattr(self.config, "speculative_token_tree", None)
        if tree_str is None:
            return [(i + 1) * (0,) for i in range(max(getattr(self.config, "num_spec_tokens", 0), 0))]
        tree_choices = ast.literal_eval(tree_str) if isinstance(tree_str, str) else tree_str
        try:
            paths = [tuple(int(choice) for choice in path) for path in tree_choices]
        except (TypeError, ValueError) as exc:
            raise ValueError("speculative_token_tree must be a sequence of integer paths") from exc
        if not paths or any(not path for path in paths):
            raise ValueError("speculative_token_tree must contain non-empty paths")
        if any(choice < 0 for path in paths for choice in path):
            raise ValueError("speculative_token_tree choices must be non-negative")
        if len(set(paths)) != len(paths):
            raise ValueError("speculative_token_tree paths must be unique")
        path_set = set(paths)
        for path in paths:
            for depth in range(1, len(path)):
                if path[:depth] not in path_set:
                    raise ValueError(f"missing tree prefix {path[:depth]} for path {path}")
        leaves = [
            path for path in paths
            if not any(len(other) > len(path) and other[:len(path)] == path for other in paths)
        ]
        if len({len(path) for path in leaves}) != 1:
            raise ValueError("all speculative_token_tree leaves must have the same depth")
        max_depth = max(len(path) for path in paths)
        if max_depth > getattr(self.config, "num_spec_tokens", 0):
            raise ValueError("speculative_token_tree depth cannot exceed num_spec_tokens")
        return sorted(paths, key=lambda t: (len(t), t))

    def _get_tree_leaf_paths(self, tree_choices: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        if not tree_choices:
            return []
        tree_set = set(tree_choices)
        leaf_paths = []
        for path in tree_choices:
            if not any(len(other) > len(path) and other[: len(path)] == path for other in tree_set):
                leaf_paths.append(path)
        return sorted(leaf_paths, key=lambda t: (len(t), t))

    def _get_tree_nodes_by_level(self, tree_choices: list[tuple[int, ...]]) -> list[list[tuple[int, ...]]]:
        if not tree_choices:
            return []
        max_depth = max(len(path) for path in tree_choices)
        return [
            [path for path in tree_choices if len(path) == depth]
            for depth in range(1, max_depth + 1)
        ]

    def _get_tree_children(self, tree_choices: list[tuple[int, ...]]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        children: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for path in tree_choices:
            parent = path[:-1]
            children.setdefault(parent, []).append(path)
        for parent in children:
            children[parent].sort(key=lambda t: (len(t), t))
        return children

    def _build_tree_attn_bias(self, tree_choices: list[tuple[int, ...]]) -> torch.Tensor | None:
        if not tree_choices:
            return None
        tree_len = len(tree_choices) + 1
        tree_attn_bias = torch.full((tree_len, tree_len), float("-inf"), dtype=torch.float32)
        tree_attn_bias.diagonal().fill_(0)
        tree_attn_bias[:, 0] = 0
        path_to_index = {path: idx + 1 for idx, path in enumerate(tree_choices)}
        for idx, path in enumerate(tree_choices, start=1):
            ancestor_indices = [path_to_index[path[:depth]] for depth in range(1, len(path))]
            if ancestor_indices:
                tree_attn_bias[idx, ancestor_indices] = 0
        return tree_attn_bias[1:, 1:].contiguous()

    def _uses_tree_proposal(self) -> bool:
        return self.world_size == 1 and len(self.spec_tree_leaf_paths) > 1

    def _can_use_tree_batch(self, seqs: list[Sequence]) -> bool:
        if not self._uses_tree_proposal():
            return False
        if self.spec_tree_depth <= 0 or self.spec_tree_attn_bias is None:
            return False
        if any(len(path) != self.spec_tree_depth for path in self.spec_tree_leaf_paths):
            return False
        tree_slots = len(self.spec_token_tree)
        for seq in seqs:
            remaining = seq.max_tokens - seq.num_completion_tokens
            if remaining - 1 < self.spec_tree_depth:
                return False
            capacity = len(seq.block_table) * self.block_size - len(seq)
            if capacity < tree_slots:
                return False
        return True

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
        mean_acceptance_length = (
            stats["emitted_tokens"] / stats["sequence_proposals"]
            if stats["sequence_proposals"] > 0 else 0.0
        )
        avg_draft = (
            stats["accepted_draft_tokens"] / stats["sequence_proposals"]
            if stats["sequence_proposals"] > 0 else 0.0
        )
        avg_seed_time = stats["seed_time"] / stats["spec_calls"]
        avg_draft_time = stats["draft_time"] / stats["spec_calls"]
        avg_verify_time = stats["verify_time"] / stats["spec_calls"]
        avg_accept_time = stats["accept_time"] / stats["spec_calls"]
        print(
            "[spec_debug] "
            f"calls={stats['spec_calls']} sequence_proposals={stats['sequence_proposals']} "
            f"proposed={stats['proposed_tokens']} accepted_draft={stats['accepted_draft_tokens']} "
            f"emitted={stats['emitted_tokens']} draft_acceptance={draft_acceptance:.6f} "
            f"mean_acceptance_length={mean_acceptance_length:.3f} "
            f"mean_accepted_draft_length={avg_draft:.3f} avg_time_ms="
            f"(seed={avg_seed_time * 1000:.2f}, draft={avg_draft_time * 1000:.2f}, verify={avg_verify_time * 1000:.2f}, accept={avg_accept_time * 1000:.2f})"
        )

    def get_spec_stats(self, reset: bool = False) -> dict:
        stats = dict(self.spec_stats)
        stats["timing_enabled"] = bool(getattr(self, "spec_debug", False))
        stats["acceptance_rate"] = (
            stats["accepted_draft_tokens"] / stats["proposed_tokens"]
            if stats["proposed_tokens"] > 0 else 0.0
        )
        stats["mean_acceptance_length"] = (
            stats["emitted_tokens"] / stats["sequence_proposals"]
            if stats["sequence_proposals"] > 0 else 0.0
        )
        stats["mean_accepted_draft_length"] = (
            stats["accepted_draft_tokens"] / stats["sequence_proposals"]
            if stats["sequence_proposals"] > 0 else 0.0
        )
        if reset:
            for key in self.spec_stats:
                self.spec_stats[key] = 0 if key in {
                    "spec_calls",
                    "sequence_proposals",
                    "proposed_tokens",
                    "accepted_draft_tokens",
                    "emitted_tokens",
                } else 0.0
        return stats

    def release_sequences(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            self.seq_prev_hidden.pop(seq_id, None)

    def _cache_prev_hidden(self, seqs: list[Sequence], prev_hidden: torch.Tensor) -> None:
        for i, seq in enumerate(seqs):
            self.seq_prev_hidden[seq.seq_id] = prev_hidden[i].clone()

    def _get_cached_prev_hidden(self, seqs: list[Sequence]) -> torch.Tensor | None:
        hidden_rows = []
        for seq in seqs:
            hidden = self.seq_prev_hidden.get(seq.seq_id)
            if hidden is None:
                return None
            hidden_rows.append(hidden)
        return torch.stack(hidden_rows, dim=0) if hidden_rows else None

    def _get_last_query_indices(self, query_lens: list[int], device: torch.device) -> torch.Tensor:
        last_indices = []
        offset = 0
        for q_len in query_lens:
            offset += q_len
            last_indices.append(offset - 1)
        return torch.tensor(last_indices, dtype=torch.int64, device=device)

    def _cache_prev_hidden_from_verify(
        self,
        seqs: list[Sequence],
        prev_hidden_source: torch.Tensor,
        cu_q: torch.Tensor,
        accept_lens: torch.Tensor,
    ) -> None:
        prev_indices = cu_q[:-1].to(torch.int64) + accept_lens.to(torch.int64)
        prev_hidden = prev_hidden_source.index_select(0, prev_indices)
        self._cache_prev_hidden(seqs, prev_hidden)

    def _cache_prev_hidden_from_indices(
        self,
        seqs: list[Sequence],
        prev_hidden_source: torch.Tensor,
        prev_indices: torch.Tensor,
    ) -> None:
        prev_hidden = prev_hidden_source.index_select(0, prev_indices.to(torch.int64))
        self._cache_prev_hidden(seqs, prev_hidden)

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

    def _slot_for_position(self, seq: Sequence, position: int) -> int:
        logical_block = position // self.block_size
        if logical_block >= len(seq.block_table):
            raise RuntimeError(
                f"position {position} has no reserved KV block for sequence {seq.seq_id}"
            )
        return seq.block_table[logical_block] * self.block_size + position % self.block_size

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
            slot_mapping.append(self._slot_for_position(seq, len(seq) - 1))
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
            slot_mapping.extend(
                self._slot_for_position(seq, position)
                for position in range(base_pos, base_pos + q_len)
            )
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
    def run_target_decode_stage(
        self,
        seqs: list[Sequence],
        input_token_ids: list[int],
        stage: int,
        active_mask: list[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if active_mask is None:
            active_mask = [True] * len(seqs)
        input_ids = torch.tensor(input_token_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(
            [
                len(seq) - 1 + stage if active else len(seq) - 1
                for seq, active in zip(seqs, active_mask, strict=True)
            ],
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            [
                self._slot_for_position(seq, len(seq) - 1 + stage) if active else -1
                for seq, active in zip(seqs, active_mask, strict=True)
            ],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            [
                len(seq) + stage if active else len(seq)
                for seq, active in zip(seqs, active_mask, strict=True)
            ],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        use_aux_hidden = bool(getattr(self.draft_model, "use_aux_hidden_state", False))
        if use_aux_hidden:
            hidden_states, aux_hidden_states = self.model(
                input_ids,
                positions,
                return_aux_hidden_states=True,
                aux_hidden_state_layer_ids=self.eagle_aux_hidden_state_layer_ids,
            )
            prev_hidden = aux_hidden_states if aux_hidden_states.numel() else hidden_states
            # Ordinary decoding uses a CUDA graph when enabled. Recompute the
            # final hidden state through that same graph so strict greedy
            # verification sees the identical BF16 kernel and argmax.
            logits_hidden = (
                hidden_states
                if self.enforce_eager
                else self.run_model_hidden(input_ids, positions, is_prefill=False)
            )
        else:
            logits_hidden = self.run_model_hidden(input_ids, positions, is_prefill=False)
            prev_hidden = logits_hidden
        target_logits = self.model.compute_logits(logits_hidden)
        reset_context()
        return target_logits, prev_hidden

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
    def propose_linear_draft_tokens(
        self,
        seqs: list[Sequence],
        k_list: list[int],
        prev_token_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        assert self.draft_model is not None
        bs = len(seqs)
        max_k = max(k_list) if k_list else 0
        draft_tokens = torch.empty((bs, max_k), dtype=torch.int64, device="cuda")
        if max_k == 0:
            return draft_tokens[:, :0], self.draft_model.combine_hidden_states(prev_token_hidden), None, None

        active_k = torch.tensor(k_list, dtype=torch.int32, device="cuda")
        base_len = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        base_pos = torch.tensor([len(seq) - 1 for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        cur_tokens = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64, device="cuda")
        cur_hidden = self.draft_model.combine_hidden_states(prev_token_hidden)
        final_hidden = cur_hidden

        for step in range(max_k):
            active = active_k > step
            step_slots = torch.tensor(
                [self._slot_for_position(seq, len(seq) - 1 + step) for seq in seqs],
                dtype=torch.int32,
                pin_memory=True,
            ).cuda(non_blocking=True)
            slot = torch.where(active, step_slots, torch.full_like(step_slots, -1))
            context_lens = torch.where(active, base_len + step, base_len)
            positions = base_pos + step
            set_context(False, slot_mapping=slot, context_lens=context_lens, block_tables=block_tables)
            hidden_out, next_hidden = self.draft_model.forward_with_hidden(cur_tokens, positions, cur_hidden)
            logits = self.draft_model.compute_logits(hidden_out)
            if self.rank == 0:
                # vLLM's EAGLE proposer uses greedy draft tokens today; keep q(logits)
                # out of the hot path to avoid stochastic draft sampling overhead.
                next_draft_tokens = logits.argmax(dim=-1)
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
        return draft_tokens, final_hidden, None, None

    @torch.inference_mode()
    def propose_tree_draft_tokens(
        self,
        seqs: list[Sequence],
        k_list: list[int],
        prev_token_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        assert self.draft_model is not None
        bs = len(seqs)
        max_k = max(k_list) if k_list else 0
        if (
            max_k == 0
            or not self.spec_tree_leaf_paths
            or max_k != self.spec_tree_depth
            or any(k != max_k for k in k_list)
        ):
            return self.propose_linear_draft_tokens(seqs, k_list, prev_token_hidden)
        base_len = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        base_pos = torch.tensor([len(seq) - 1 for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        base_slot = torch.tensor(
            [self._slot_for_position(seq, len(seq) - 1) for seq in seqs],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        cur_tokens = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64, device="cuda")
        cur_hidden = self.draft_model.combine_hidden_states(prev_token_hidden)
        set_context(False, slot_mapping=base_slot, context_lens=base_len, block_tables=block_tables)
        root_hidden_out, root_next_hidden = self.draft_model.forward_with_hidden(cur_tokens, base_pos, cur_hidden)
        root_logits = self.draft_model.compute_logits(root_hidden_out)
        reset_context()

        node_tokens: dict[tuple[int, ...], torch.Tensor] = {}
        node_scores: dict[tuple[int, ...], torch.Tensor] = {}
        root_paths = self.spec_tree_nodes_by_level[0]
        num_root_children = len(root_paths)
        if num_root_children == 1:
            root_vals, root_ids = root_logits.max(dim=-1, keepdim=True)
        else:
            root_vals, root_ids = torch.topk(root_logits, k=num_root_children, dim=-1)
        root_tokens = self.draft_model.map_draft_to_target(root_ids)
        for idx, path in enumerate(root_paths):
            node_tokens[path] = root_tokens[:, idx]
            node_scores[path] = root_vals[:, idx].float()

        if self.spec_tree_depth == 1:
            best_root_idx = root_vals.argmax(dim=1)
            draft_tokens = root_tokens.gather(1, best_root_idx.view(bs, 1))
            return draft_tokens, root_next_hidden, root_tokens, best_root_idx

        tree_input_ids = root_tokens
        tree_positions = (base_pos[:, None] + 1).expand(bs, num_root_children)
        tree_hidden_inputs = root_next_hidden[:, None, :].expand(bs, num_root_children, -1)
        frontier_paths = root_paths
        frontier_count = len(frontier_paths)

        for level_idx in range(self.spec_tree_depth - 1):
            query_len = tree_input_ids.size(1)
            flat_input_ids = tree_input_ids.reshape(-1)
            flat_positions = tree_positions.reshape(-1)
            flat_hidden_inputs = tree_hidden_inputs.reshape(query_len * bs, -1)
            query_offsets = base_pos[:, None] + torch.arange(
                1, query_len + 1, device="cuda", dtype=torch.int64
            )
            block_numbers = query_offsets // self.block_size
            block_ids = block_tables.gather(1, block_numbers)
            slot_mapping = (block_ids * self.block_size + (query_offsets % self.block_size).to(block_ids.dtype)).reshape(-1)
            cu_seqlens_q = torch.arange(bs + 1, device="cuda", dtype=torch.int32) * query_len
            cu_seqlens_k = torch.zeros(bs + 1, device="cuda", dtype=torch.int32)
            cu_seqlens_k[1:] = (base_len + query_len).cumsum(dim=0)
            set_context(
                True,
                cu_seqlens_q,
                cu_seqlens_k,
                query_len,
                int((base_len + query_len).max().item()),
                slot_mapping.to(torch.int32),
                None,
                block_tables,
                self.spec_tree_attn_bias[:query_len, :query_len].to(device=flat_input_ids.device),
                prefill_last_only=False,
            )
            hidden_out, next_hidden = self.draft_model.forward_with_hidden(
                flat_input_ids,
                flat_positions,
                flat_hidden_inputs,
            )
            reset_context()

            hidden_out = hidden_out.view(bs, query_len, -1)
            next_hidden = next_hidden.view(bs, query_len, -1)
            frontier_hidden_out = hidden_out[:, -frontier_count:, :]
            frontier_next_hidden = next_hidden[:, -frontier_count:, :]
            frontier_logits = self.draft_model.compute_logits(
                frontier_hidden_out.reshape(bs * frontier_count, -1)
            ).view(bs, frontier_count, -1)

            next_paths: list[tuple[int, ...]] = []
            next_tokens_per_parent = []
            next_hidden_inputs_per_parent = []
            for frontier_idx, parent_path in enumerate(frontier_paths):
                child_paths = self.spec_tree_children.get(parent_path, [])
                if not child_paths:
                    continue
                num_children = len(child_paths)
                parent_logits = frontier_logits[:, frontier_idx, :]
                if num_children == 1:
                    child_vals, child_ids = parent_logits.max(dim=-1, keepdim=True)
                else:
                    child_vals, child_ids = torch.topk(parent_logits, k=num_children, dim=-1)
                child_tokens = self.draft_model.map_draft_to_target(child_ids)
                for child_idx, child_path in enumerate(child_paths):
                    node_tokens[child_path] = child_tokens[:, child_idx]
                    node_scores[child_path] = child_vals[:, child_idx].float()
                    next_paths.append(child_path)
                next_tokens_per_parent.append(child_tokens)
                next_hidden_inputs_per_parent.append(
                    frontier_next_hidden[:, frontier_idx : frontier_idx + 1, :].expand(-1, num_children, -1)
                )

            if not next_paths:
                break
            next_count = len(next_paths)
            next_depth = level_idx + 2
            next_tokens = torch.cat(next_tokens_per_parent, dim=1)
            next_positions = (base_pos[:, None] + next_depth).expand(bs, next_count)
            next_hidden_inputs = torch.cat(next_hidden_inputs_per_parent, dim=1)
            tree_input_ids = torch.cat([tree_input_ids, next_tokens], dim=1)
            tree_positions = torch.cat([tree_positions, next_positions], dim=1)
            tree_hidden_inputs = torch.cat([tree_hidden_inputs, next_hidden_inputs], dim=1)
            frontier_paths = next_paths
            frontier_count = next_count

        leaf_scores = []
        leaf_tokens = []
        for leaf_path in self.spec_tree_leaf_paths:
            prefixes = [leaf_path[: depth] for depth in range(1, len(leaf_path) + 1)]
            leaf_scores.append(sum(node_scores[prefix] for prefix in prefixes))
            leaf_tokens.append(torch.stack([node_tokens[prefix] for prefix in prefixes], dim=1))
        leaf_scores = torch.stack(leaf_scores, dim=1)
        leaf_tokens = torch.stack(leaf_tokens, dim=1)
        best_leaf_idx = leaf_scores.argmax(dim=1)
        root_path_to_index = {path: idx for idx, path in enumerate(root_paths)}
        leaf_root_indices = torch.tensor(
            [root_path_to_index[leaf_path[:1]] for leaf_path in self.spec_tree_leaf_paths],
            dtype=torch.int64,
            device=best_leaf_idx.device,
        )
        best_root_idx = leaf_root_indices.index_select(0, best_leaf_idx)
        gather_index = best_leaf_idx.view(bs, 1, 1).expand(bs, 1, max_k)
        draft_tokens = leaf_tokens.gather(1, gather_index).squeeze(1)
        return draft_tokens, root_next_hidden, tree_input_ids, best_root_idx

    @torch.inference_mode()
    def propose_draft_tokens(
        self,
        seqs: list[Sequence],
        k_list: list[int],
        prev_token_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self._can_use_tree_batch(seqs):
            return self.propose_tree_draft_tokens(seqs, k_list, prev_token_hidden)
        return self.propose_linear_draft_tokens(seqs, k_list, prev_token_hidden)

    def _sample_from_probs(self, probs: torch.Tensor) -> int:
        token = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return int(token.item())

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        token_ids: torch.Tensor | None = None,
    ) -> int:
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        sampled = self._sample_from_probs(probs)
        if token_ids is None:
            return sampled
        return int(token_ids[sampled].item())

    # Greedy verifier: accept draft tokens only while they match the target argmax.
    # On mismatch we fall back to the target sample for that step.
    @torch.inference_mode()
    def accept_draft_tokens(
        self,
        seqs: list[Sequence],
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        temperatures: torch.Tensor,
        k_list: list[int],
        cu_q: torch.Tensor,
        target_token_ids: torch.Tensor | None = None,
        use_padded_kernels: bool = False,
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
        bs = len(seqs)
        out_tokens_per_seq: list[list[int]] = []
        accept_lens = torch.zeros(bs, dtype=torch.int32, device=target_logits.device)
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
                greedy_idx = seq_logits[step].argmax(dim=-1)
                target_greedy = (
                    int(greedy_idx.item())
                    if target_token_ids is None
                    else int(target_token_ids[greedy_idx].item())
                )
                if target_greedy == int(y.item()):
                    accepted_tokens.append(int(y.item()))
                    accept_lens[b] += 1
                else:
                    accepted_tokens.append(target_greedy)
                    rejected = True
                    break
            if not rejected:
                extra_idx = seq_logits[q_len - 1].argmax(dim=-1)
                extra_token = (
                    int(extra_idx.item())
                    if target_token_ids is None
                    else int(target_token_ids[extra_idx].item())
                )
                accepted_tokens.append(extra_token)
            out_tokens_per_seq.append(accepted_tokens)
        if use_padded_kernels:
            max_sampled_tokens = max((len(toks) for toks in out_tokens_per_seq), default=1)
            sampled_token_ids = torch.full(
                (bs, max_sampled_tokens),
                -1,
                dtype=torch.int32,
                device=target_logits.device,
            )
            for b, toks in enumerate(out_tokens_per_seq):
                if toks:
                    sampled_token_ids[b, : len(toks)] = torch.tensor(toks, dtype=torch.int32, device=target_logits.device)
            vocab_size = (
                int(target_token_ids.max().item()) + 1
                if target_token_ids is not None else target_logits.size(-1)
            )
            _, valid_sampled_tokens_count = prepare_next_token_ids_padded(sampled_token_ids, vocab_size=vocab_size)
            accept_lens = (valid_sampled_tokens_count - 1).clamp_min_(0).to(torch.int32)
            cu_num_draft_tokens = torch.tensor(k_list, dtype=torch.int32, device=target_logits.device).cumsum(dim=0)
            prev_indices = prepare_inputs_padded(cu_num_draft_tokens, valid_sampled_tokens_count, cu_q)
        else:
            prev_indices = cu_q[:-1].to(torch.int64) + accept_lens.to(torch.int64)
        return out_tokens_per_seq, accept_lens, prev_indices

    @torch.inference_mode()
    def run_sequential_verify(
        self,
        seqs: list[Sequence],
        draft_tokens: torch.Tensor,
        k_list: list[int],
        tree_tokens: torch.Tensor | None = None,
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
        bs = len(seqs)
        device = draft_tokens.device
        out_tokens_per_seq: list[list[int]] = [[] for _ in range(bs)]
        accepted_hidden_rows: list[list[torch.Tensor]] = [[] for _ in range(bs)]
        accept_lens = torch.zeros(bs, dtype=torch.int32, device=device)
        next_prev_hidden = [None] * bs
        active = [True] * bs
        input_token_ids = [seq.last_token for seq in seqs]
        matched_paths: list[tuple[int, ...] | None] = [None] * bs
        max_k = max(k_list, default=0)

        # Keep every scheduler row in every target call. Besides avoiding Python
        # batch compaction, this deliberately matches ordinary greedy decode's
        # GEMM/CUDA-graph batch shape so near-tied BF16 logits cannot change the
        # fallback argmax merely because speculative verification was packed.
        for stage in range(max_k + 1):
            stage_active = [
                active[index] and stage <= k_list[index]
                for index in range(bs)
            ]
            if not any(stage_active):
                break
            target_logits, prev_hidden = self.run_target_decode_stage(
                seqs,
                input_token_ids,
                stage,
                stage_active,
            )
            next_input_token_ids = list(input_token_ids)
            for seq_idx in range(bs):
                if not stage_active[seq_idx]:
                    continue
                prev_hidden_row = prev_hidden[seq_idx].clone()
                accepted_hidden_rows[seq_idx].append(prev_hidden_row)
                target_greedy = int(target_logits[seq_idx].argmax(dim=-1).item())
                out_tokens_per_seq[seq_idx].append(target_greedy)
                next_input_token_ids[seq_idx] = target_greedy

                if stage == k_list[seq_idx]:
                    next_prev_hidden[seq_idx] = prev_hidden_row
                    active[seq_idx] = False
                    continue

                matched_path = None
                if tree_tokens is not None:
                    candidate_paths = (
                        self.spec_tree_nodes_by_level[0]
                        if stage == 0
                        else self.spec_tree_children.get(matched_paths[seq_idx], [])
                    )
                    for candidate_path in candidate_paths:
                        candidate_token = int(
                            tree_tokens[
                                seq_idx,
                                self.spec_tree_path_to_index[candidate_path],
                            ].item()
                        )
                        if candidate_token == target_greedy:
                            matched_path = candidate_path
                            break
                elif target_greedy == int(draft_tokens[seq_idx, stage].item()):
                    matched_path = (stage,)

                if matched_path is None:
                    next_prev_hidden[seq_idx] = prev_hidden_row
                    active[seq_idx] = False
                else:
                    matched_paths[seq_idx] = matched_path
                    accept_lens[seq_idx] += 1
            input_token_ids = next_input_token_ids

        if any(hidden is None for hidden in next_prev_hidden):
            raise RuntimeError("sequential verifier did not finalize every sequence")
        next_prev_hidden_tensor = torch.stack(next_prev_hidden, dim=0)
        return out_tokens_per_seq, accept_lens, next_prev_hidden_tensor, accepted_hidden_rows

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
        k_tensor = torch.tensor(k_list, dtype=torch.int64, device="cuda")
        last_draft_tokens = torch.tensor(
            [int(draft_tokens[i, k_list[i] - 1].item()) if k_list[i] > 0 else 0 for i in range(len(seqs))],
            dtype=torch.int64,
            device="cuda",
        )
        final_slots = torch.tensor(
            [
                self._slot_for_position(seq, len(seq) - 1 + k_list[i])
                for i, seq in enumerate(seqs)
            ],
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        slot = torch.where(fin_mask, final_slots, torch.full_like(final_slots, -1))
        context_lens = torch.where(fin_mask, base_len + k_tensor.to(torch.int32), base_len)
        positions = base_pos + k_tensor
        set_context(False, slot_mapping=slot, context_lens=context_lens, block_tables=block_tables)
        _ = self.draft_model.forward_with_hidden(last_draft_tokens, positions, final_hidden)
        reset_context()

    @torch.inference_mode()
    def replay_draft_cache(
        self,
        seqs: list[Sequence],
        token_ids: list[list[int]],
        prev_token_hidden: torch.Tensor,
        prev_hidden_source: torch.Tensor,
        cu_q: torch.Tensor,
        accepted_query_indices: list[list[int]] | None = None,
        accepted_hidden_rows: list[list[torch.Tensor]] | None = None,
    ) -> None:
        if self.draft_model is None:
            return
        input_ids = []
        positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        raw_hidden_inputs = []
        max_q = 0
        max_k = 0
        for b, seq in enumerate(seqs):
            cache_tokens = token_ids[b][:-1]
            q_len = len(cache_tokens) + 1
            k_len = len(seq) + len(cache_tokens)
            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + k_len)
            max_q = max(max_q, q_len)
            max_k = max(max_k, k_len)
            base_pos = len(seq) - 1
            seq_input_ids = [seq.last_token, *cache_tokens]
            input_ids.extend(seq_input_ids)
            positions.extend(range(base_pos, base_pos + q_len))
            slot_mapping.extend(
                self._slot_for_position(seq, position)
                for position in range(base_pos, base_pos + q_len)
            )
            raw_hidden_inputs.append(prev_token_hidden[b])
            if cache_tokens:
                if accepted_hidden_rows is not None:
                    if len(accepted_hidden_rows[b]) < len(cache_tokens):
                        raise RuntimeError("insufficient target hidden states for draft cache replay")
                    raw_hidden_inputs.extend(accepted_hidden_rows[b][:len(cache_tokens)])
                elif accepted_query_indices is None:
                    start = int(cu_q[b].item())
                    raw_hidden_inputs.extend(prev_hidden_source[start : start + len(cache_tokens)])
                else:
                    query_indices = torch.tensor(
                        accepted_query_indices[b],
                        dtype=torch.int64,
                        device=prev_hidden_source.device,
                    )
                    raw_hidden_inputs.extend(prev_hidden_source.index_select(0, query_indices))

        if not input_ids:
            return

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        hidden_inputs = torch.stack(raw_hidden_inputs, dim=0)
        draft_hidden = self.draft_model.combine_hidden_states(hidden_inputs)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_q,
            max_k,
            slot_mapping,
            None,
            block_tables,
            prefill_last_only=False,
        )
        _ = self.draft_model(input_ids, positions, hidden_states=draft_hidden)
        reset_context()

    # End-to-end run: prepare inputs, run model, sample tokens (on rank 0), and reset context.
    @torch.inference_mode()
    def run_baseline(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]] | None:
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        fused_prefill = False
        if is_prefill and self.draft_model is not None and all(seq.num_cached_tokens == 0 for seq in seqs):
            prev_hidden_source = None
            if bool(getattr(self.draft_model, "use_aux_hidden_state", False)):
                hidden_states, aux_hidden_states = self.model(
                    input_ids,
                    positions,
                    return_aux_hidden_states=True,
                    aux_hidden_state_layer_ids=self.eagle_aux_hidden_state_layer_ids,
                )
                prev_hidden_source = aux_hidden_states if aux_hidden_states.numel() else hidden_states
                draft_hidden = self.draft_model.combine_hidden_states(prev_hidden_source)
            else:
                hidden_states = self.model(input_ids, positions)
                prev_hidden_source = hidden_states
                draft_hidden = hidden_states
            logits = self.model.compute_logits(hidden_states)
            self.fill_draft_cache_from_prefill(input_ids, positions, draft_hidden)
            query_lens = [len(seq) - seq.num_cached_tokens for seq in seqs]
            last_query_indices = self._get_last_query_indices(query_lens, prev_hidden_source.device)
            prev_hidden = prev_hidden_source.index_select(0, last_query_indices)
            self._cache_prev_hidden(seqs, prev_hidden)
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
        if temperatures is not None and bool((temperatures > 0).any().item()):
            raise ValueError(
                "EAGLE-3 speculative decoding currently supports greedy sampling only "
                "(temperature=0); rejection sampling is not implemented."
            )
        use_tree_proposal = self._can_use_tree_batch(seqs)
        k_list = []
        for seq in seqs:
            remaining = seq.max_tokens - seq.num_completion_tokens
            if use_tree_proposal:
                k = min(self.spec_tree_depth, max(0, remaining - 1))
            else:
                k = min(getattr(self.config, "num_spec_tokens", 0), max(0, remaining - 1))
                k = min(k, self.block_size - seq.last_block_num_tokens)
            k_list.append(k)

        prev_token_hidden = self._get_cached_prev_hidden(seqs)
        if prev_token_hidden is None:
            input_ids, positions = self.prepare_decode(seqs)
            if self.spec_debug:
                self._spec_sync()
                t0 = perf_counter()
            prev_token_hidden = self.run_draft_seed_hidden(input_ids, positions)
            if self.spec_debug:
                self._spec_sync()
                self.spec_stats["seed_time"] += perf_counter() - t0
            reset_context()

        self.spec_stats["spec_calls"] += 1
        self.spec_stats["sequence_proposals"] += len(seqs)
        if self.spec_debug:
            self._spec_sync()
            t0 = perf_counter()
        draft_tokens, final_hidden, tree_tokens, _tree_root_choice = self.propose_draft_tokens(
            seqs,
            k_list,
            prev_token_hidden,
        )
        self.spec_stats["proposed_tokens"] += sum(k_list)
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["draft_time"] += perf_counter() - t0
        if self.spec_debug:
            self._spec_sync()
            t0 = perf_counter()
        token_ids, accept_lens, next_prev_hidden, accepted_hidden_rows = (
            self.run_sequential_verify(
                seqs,
                draft_tokens,
                k_list,
                tree_tokens if use_tree_proposal else None,
            )
        )
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["verify_time"] += perf_counter() - t0

        if self.rank != 0:
            if self.world_size > 1:
                dist.broadcast(accept_lens, src=0)
            self.finalize_draft_cache(seqs, draft_tokens, final_hidden, accept_lens, k_list)
            return None
        if self.spec_debug:
            self._spec_sync()
            t0 = perf_counter()
        accepted_draft_tokens = int(accept_lens.sum().item())
        emitted_tokens = sum(len(toks) for toks in token_ids)
        self.spec_stats["accepted_draft_tokens"] += accepted_draft_tokens
        self.spec_stats["emitted_tokens"] += emitted_tokens
        if self.world_size > 1:
            dist.broadcast(accept_lens, src=0)
        self._cache_prev_hidden(seqs, next_prev_hidden)
        if use_tree_proposal:
            cu_q = torch.arange(
                len(seqs) + 1,
                dtype=torch.int32,
                device=next_prev_hidden.device,
            )
            self.replay_draft_cache(
                seqs,
                token_ids,
                prev_token_hidden,
                next_prev_hidden,
                cu_q,
                accepted_hidden_rows=accepted_hidden_rows,
            )
        else:
            self.finalize_draft_cache(seqs, draft_tokens, final_hidden, accept_lens, k_list)
        if self.spec_debug:
            self._spec_sync()
            self.spec_stats["accept_time"] += perf_counter() - t0
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
