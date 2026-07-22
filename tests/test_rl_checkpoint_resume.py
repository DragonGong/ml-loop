from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torchtune.training.checkpointing._utils import safe_torch_load

import phi_agents.rl.train as train_module
from phi_agents.rl.rl_utils import GradientRMS
from phi_agents.rl.train import (
    RLOOTrainer,
    _capture_rank_training_state,
    _restore_rank_training_state,
)

if TYPE_CHECKING:
    from pathlib import Path


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_item, actual_item)
    else:
        assert actual == expected


def test_exact_training_state_roundtrip_restores_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    training_rng = np.random.default_rng(14)

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    gradient_rms = GradientRMS()
    gradient_rms.update(2.5)
    state = _capture_rank_training_state(
        optimizer=optimizer,
        lr_scheduler=scheduler,
        training_rng=training_rng,
        rank=0,
        world_size=1,
        gradient_rms=gradient_rms,
        invalid_steps_skipped=2,
        high_kl_events=3,
        n_outlier_grads=4,
    )
    state_path = tmp_path / "rank-00000.pt"
    torch.save(state, state_path)
    loaded = safe_torch_load(state_path, mmap=False)

    expected_random_values = (
        random.random(),
        float(np.random.random()),
        float(training_rng.random()),
        torch.rand(4),
    )
    expected_optimizer_state = loaded["optimizer"]
    expected_scheduler_state = loaded["lr_scheduler"]

    # Move every mutable component away from the checkpoint state.
    model(torch.full((2, 3), 2.0)).sum().backward()
    optimizer.step()
    scheduler.step()
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    training_rng = np.random.default_rng(104)
    gradient_rms = GradientRMS()

    counters = _restore_rank_training_state(
        loaded,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        training_rng=training_rng,
        rank=0,
        world_size=1,
        gradient_rms=gradient_rms,
    )

    _assert_nested_equal(expected_optimizer_state, optimizer.state_dict())
    _assert_nested_equal(expected_scheduler_state, scheduler.state_dict())
    assert (gradient_rms.mean, gradient_rms.var, gradient_rms.count) == (2.5, 0.0, 1)
    assert counters == {
        "invalid_steps_skipped": 2,
        "high_kl_events": 3,
        "n_outlier_grads": 4,
    }
    assert random.random() == expected_random_values[0]
    assert float(np.random.random()) == expected_random_values[1]
    assert float(training_rng.random()) == expected_random_values[2]
    torch.testing.assert_close(torch.rand(4), expected_random_values[3], rtol=0, atol=0)


def test_legacy_trainer_state_and_missing_rank_state_remain_loadable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint_dir = tmp_path / "checkpoint-7"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "iterations_completed": 7,
            "output_tokens_generated": 123,
            "rollouts_generated": 42,
            "profiler_state": {"legacy": True},
            "global_step": 99,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    observed_profiler_state: list[dict[str, Any]] = []
    monkeypatch.setattr(
        train_module,
        "profiler_load_state_dict",
        lambda state, _logger: observed_profiler_state.append(state),
    )

    trainer = object.__new__(RLOOTrainer)
    trainer._rank0_logger = logging.getLogger("test_rl_checkpoint_resume")
    trainer._rank = 0
    trainer._world_size = 1
    trainer._rng = np.random.default_rng(1)
    trainer._grad_rms = GradientRMS()
    trainer._invalid_steps_skipped = 0
    trainer._high_kl_events = 0
    trainer._n_outlier_grads = 0

    trainer._load_trainer_state(checkpoint_dir)
    assert trainer._iterations_completed == 7
    assert trainer._output_tokens_generated == 123
    assert trainer._rollouts_generated == 42
    assert trainer._global_step == 99
    assert observed_profiler_state == [{"legacy": True}]

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    assert not trainer._load_rank_training_state(checkpoint_dir, optimizer, scheduler)
