import pickle
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

    # Prepare decode inputs for a virtual step within a speculative draft cycle.
    def prepare_decode_step(self, seqs: list[Sequence], input_token_ids: torch.Tensor, step_offset: int):
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            pos = (len(seq) - 1) + step_offset
            positions.append(pos)
            context_lens.append(len(seq) + step_offset)
            block_idx = pos // self.block_size
            slot_in_block = pos % self.block_size
            slot_mapping.append(seq.block_table[block_idx] * self.block_size + slot_in_block)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_token_ids, positions

    # Prepare a single prefill call that verifies pending + draft tokens (k+1 queries each).
    def prepare_spec_verify(self, seqs: list[Sequence], draft_tokens: torch.Tensor, k: int):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for b, seq in enumerate(seqs):
            base_pos = len(seq) - 1
            q_len = k + 1
            k_len = len(seq) + k
            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + k_len)
            max_seqlen_q = max(max_seqlen_q, q_len)
            max_seqlen_k = max(max_seqlen_k, k_len)
            # j=0 pending token, j>=1 draft token j-1
            for j in range(q_len):
                pos = base_pos + j
                token = seq.last_token if j == 0 else draft_tokens[b, j - 1].item()
                input_ids.append(token)
                positions.append(pos)
                block_idx = pos // self.block_size
                slot_in_block = pos % self.block_size
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + slot_in_block)
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
        return input_ids, positions

    # Assemble temperatures tensor on rank 0 for sampling.
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    # Execute the model; optionally reuse captured CUDA graphs for small decode batches.
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
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
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_draft_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        assert self.draft_model is not None
        # Draft path uses eager for simplicity; decode batches are small in speculative flow.
        return self.draft_model.compute_logits(self.draft_model(input_ids, positions))

    # Draft propose pass: generate k draft tokens (one token per micro-step).
    @torch.inference_mode()
    def propose_draft_tokens(
        self,
        seqs: list[Sequence],
        k: int,
        temperatures: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        assert self.draft_model is not None
        bs = len(seqs)
        draft_tokens = torch.empty((bs, k), dtype=torch.int64, device="cuda")
        cur_tokens = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64, device="cuda")
        draft_step_logits = [] if self.rank == 0 else None

        for step in range(k):
            input_ids, positions = self.prepare_decode_step(seqs, cur_tokens, step)
            logits = self.run_draft_model(input_ids, positions, is_prefill=False)
            if self.rank == 0:
                draft_step_logits.append(logits)
                next_tok = self.sampler(logits, temperatures)
            else:
                next_tok = torch.empty((bs,), dtype=torch.int64, device="cuda")
            if self.world_size > 1:
                dist.broadcast(next_tok, src=0)
            draft_tokens[:, step] = next_tok
            cur_tokens = next_tok
        reset_context()
        return draft_tokens, draft_step_logits

    # Accept/reject draft tokens using target logits (exact speculative decoding).
    @torch.inference_mode()
    def accept_draft_tokens(
        self,
        seqs: list[Sequence],
        draft_tokens: torch.Tensor,
        draft_step_logits: list[torch.Tensor],
        target_logits: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> list[list[int]]:
        bs, k = draft_tokens.shape
        target_logits = target_logits.view(bs, k + 1, -1)
        scaled_target = target_logits.float() / temperatures[:, None, None]
        log_probs_p = torch.log_softmax(scaled_target, dim=-1)
        log_probs_q = [torch.log_softmax(step_logits.float() / temperatures[:, None], dim=-1) for step_logits in draft_step_logits]

        out_tokens_per_seq: list[list[int]] = []
        for b in range(bs):
            accepted_tokens: list[int] = []
            reject_step = None
            for step in range(k):
                y = draft_tokens[b, step]
                lp = log_probs_p[b, step, y]
                lq = log_probs_q[step][b, y]
                a = torch.exp(lp - lq).clamp(max=1.0)
                if torch.rand((), device=lp.device) <= a:
                    accepted_tokens.append(int(y.item()))
                else:
                    reject_step = step
                    break

            if reject_step is not None:
                logits_p_step = log_probs_p[b, reject_step]
                logits_q_step = log_probs_q[reject_step][b]
                dist_p = torch.distributions.Categorical(logits=logits_p_step)
                while True:
                    x = dist_p.sample()
                    if x >= logits_q_step.numel():
                        z = int(x.item())
                        break
                    lp_x = logits_p_step[x]
                    lq_x = logits_q_step[x]
                    alpha = torch.clamp(1 - torch.exp(lq_x - lp_x), min=0.0, max=1.0)
                    if torch.rand((), device=alpha.device) <= alpha:
                        z = int(x.item())
                        break
                out_tokens = accepted_tokens + [z]
            else:
                dist_p = torch.distributions.Categorical(logits=log_probs_p[b, k])
                z = int(dist_p.sample().item())
                out_tokens = accepted_tokens + [z]
            out_tokens_per_seq.append(out_tokens)
        return out_tokens_per_seq

    # End-to-end run: prepare inputs, run model, sample tokens (on rank 0), and reset context.
    def run_baseline(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]] | None:
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        if self.draft_model is not None and is_prefill:
            _ = self.run_draft_model(input_ids, positions, is_prefill)

        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        if token_ids is None:
            return None
        return [[t] for t in token_ids]

    def run_spec_decode(self, seqs: list[Sequence]) -> list[list[int]] | None:
        assert self.draft_model is not None
        k = getattr(self.config, "num_spec_tokens", 0)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        draft_tokens, draft_step_logits = self.propose_draft_tokens(seqs, k, temperatures)
        input_ids, positions = self.prepare_spec_verify(seqs, draft_tokens, k)
        target_logits = self.run_model(input_ids, positions, is_prefill=True)
        reset_context()

        if self.rank != 0:
            return None
        token_ids = self.accept_draft_tokens(seqs, draft_tokens, draft_step_logits, target_logits, temperatures)
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
