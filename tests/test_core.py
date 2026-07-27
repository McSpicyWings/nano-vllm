from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def test_sampling_params_allow_greedy() -> None:
    params = SamplingParams(temperature=0, max_tokens=1)
    assert params.temperature == 0
    with pytest.raises(AssertionError, match="non-negative"):
        SamplingParams(temperature=-0.1)


def test_sampler_uses_argmax_at_zero_temperature() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
    temperatures = torch.zeros(2)
    assert Sampler()(logits, temperatures).tolist() == [1, 0]


def test_full_vocab_target_fallback_is_not_limited_to_draft_vocab() -> None:
    runner = object.__new__(ModelRunner)
    draft_tokens = torch.tensor([[1, 2]])
    target_logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 9.0],
            [0.0, 1.0, 9.0, 3.0, 2.0],
            [9.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    token_ids, accept_lens, _ = runner.accept_draft_tokens(
        [SimpleNamespace()],
        draft_tokens,
        target_logits,
        torch.zeros(1),
        [2],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert token_ids == [[4]]
    assert accept_lens.tolist() == [0]


def test_greedy_acceptance_emits_bonus_after_full_match() -> None:
    runner = object.__new__(ModelRunner)
    token_ids, accept_lens, prev_indices = runner.accept_greedy_target_ids(
        torch.tensor([[1, 2]]),
        torch.tensor([1, 2, 4]),
        [2],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert token_ids == [[1, 2, 4]]
    assert accept_lens.tolist() == [2]
    assert prev_indices.tolist() == [2]


def test_spec_stats_use_sequence_level_denominator_and_reset() -> None:
    runner = object.__new__(ModelRunner)
    runner.spec_stats = {
        "spec_calls": 2,
        "sequence_proposals": 5,
        "proposed_tokens": 12,
        "accepted_draft_tokens": 4,
        "emitted_tokens": 9,
        "seed_time": 0.0,
        "draft_time": 0.0,
        "verify_time": 0.0,
        "accept_time": 0.0,
    }
    stats = runner.get_spec_stats(reset=True)
    assert stats["acceptance_rate"] == pytest.approx(4 / 12)
    assert stats["mean_acceptance_length"] == pytest.approx(9 / 5)
    assert stats["mean_accepted_draft_length"] == pytest.approx(4 / 5)
    assert runner.spec_stats["sequence_proposals"] == 0


def test_tree_parser_validates_prefixes_and_depth() -> None:
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        speculative_token_tree=str(
            [(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)]
        ),
        num_spec_tokens=3,
    )
    assert runner._build_spec_token_tree() == [
        (0,),
        (1,),
        (0, 0),
        (1, 0),
        (0, 0, 0),
        (1, 0, 0),
    ]

    runner.config.speculative_token_tree = str([(0,), (0, 0, 0)])
    with pytest.raises(ValueError, match="missing tree prefix"):
        runner._build_spec_token_tree()


def test_block_manager_reserve_and_rollback() -> None:
    manager = BlockManager(num_blocks=4, block_size=256)
    seq = Sequence([1] * 255, SamplingParams(temperature=0, max_tokens=8))
    manager.allocate(seq)
    assert len(seq.block_table) == 1
    assert manager.reserve(seq, 3) == 1
    assert len(seq.block_table) == 2
    manager.rollback_to(seq, 255)
    assert len(seq.block_table) == 1
    manager.deallocate(seq)
    assert len(manager.free_block_ids) == 4
