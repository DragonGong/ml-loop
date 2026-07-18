from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import phi_agents.rl.train as train_module
from phi_agents.rl.train import (
    IterationMetricAccumulator,
    RLOOTrainer,
    RolloutID,
    _log_training_failure,
    _parameter_update_l2_norm,
    _snapshot_trainable_parameters,
    sampled_token_entropy_stats,
    write_iteration_metric_report,
)


def _rollout(log_probs: list[float], is_output: list[bool]) -> SimpleNamespace:
    return SimpleNamespace(
        policy_token_info=SimpleNamespace(
            log_probs=log_probs,
            is_output=is_output,
            tokens=list(range(len(log_probs))),
        )
    )


def test_iteration_metric_report_uses_token_weighting_and_strict_json(tmp_path) -> None:
    accumulator = IterationMetricAccumulator(iteration=5, global_step_start=10)
    accumulator.record_gradient_observation(
        per_token_kl_sum=1.5,
        per_token_kl_observations=3,
        ppo_clipped_observations=1,
        ppo_total_observations=3,
    )
    accumulator.record_gradient_observation(
        per_token_kl_sum=0.5,
        per_token_kl_observations=1,
        ppo_clipped_observations=1,
        ppo_total_observations=1,
    )
    accumulator.record_optimizer_result(
        grad_norm=2.0,
        optimizer_stepped=True,
        parameter_update_l2_norm=0.25,
        learning_rate=5e-5,
    )
    accumulator.record_optimizer_result(
        grad_norm=float("inf"),
        optimizer_stepped=False,
        parameter_update_l2_norm=0.0,
        learning_rate=5e-5,
    )
    accumulator.high_kl_events = 1
    accumulator.learning_rate_before_scheduler = 5e-5
    accumulator.learning_rate_after_scheduler = 4e-5

    report = accumulator.report(
        status="completed",
        global_step_end=12,
        cumulative_high_kl_events=3,
    )
    path = write_iteration_metric_report(tmp_path, report)
    parsed = json.loads(path.read_text())

    assert parsed["per_token_kl"]["mean"] == pytest.approx(0.5)
    assert parsed["per_token_kl"]["token_observations"] == 4
    assert parsed["ppo_clip_fraction"]["fraction"] == pytest.approx(0.5)
    assert parsed["actual_optimizer_steps"] == 1
    assert parsed["attempted_gradient_steps"] == 2
    assert parsed["grad_norm_before_clipping"]["nonfinite_count"] == 1
    assert parsed["parameter_update_l2_norm"]["mean"] == pytest.approx(0.25)
    assert parsed["high_kl_events"] == {"iteration": 1, "cumulative": 3}
    assert not list(path.parent.glob("*.tmp"))


def test_sampled_token_entropy_is_output_token_surprisal() -> None:
    stats = sampled_token_entropy_stats(
        [
            _rollout([float("nan"), -1.0, -2.0], [False, True, True]),
            _rollout([float("nan"), float("nan")], [False, True]),
        ]
    )

    assert stats["mean_nats"] == pytest.approx(1.5)
    assert stats["finite_output_tokens"] == 2
    assert stats["nonfinite_output_tokens"] == 1


def test_advantage_filter_reports_output_tokens_without_changing_selection() -> None:
    trainer = object.__new__(RLOOTrainer)
    trainer._cfg = SimpleNamespace(  # type: ignore[attr-defined]
        params=SimpleNamespace(
            abs_adv_threshold=0.01,
            pos_adv_only=False,
            minibatch_size=1,
        )
    )
    trainer._world_size = 1  # type: ignore[attr-defined]
    trainer._rng = np.random.default_rng(7)  # type: ignore[attr-defined]
    rollouts = [
        _rollout([float("nan"), -1.0, -1.0], [False, True, True]),
        _rollout([float("nan"), -1.0, -1.0, -1.0], [False, True, True, True]),
    ]
    ids = [RolloutID(0, 0, 0, 0), RolloutID(0, 0, 0, 1)]

    subsets, filtered_fraction, _empirical_threshold, stats = trainer._filter_rollouts(  # type: ignore[arg-type]
        rollouts,
        np.array([0.0, 0.5]),
        ids,
    )

    assert filtered_fraction == pytest.approx(0.5)
    assert len(subsets[0][0]) == 1
    assert subsets[0][0][0] is rollouts[1]
    assert stats["candidate_output_tokens"] == 5
    assert stats["below_threshold_output_tokens"] == 2
    assert stats["actually_filtered_output_tokens"] == 2


def test_parameter_update_norm_measures_actual_optimizer_delta() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    snapshots = _snapshot_trainable_parameters(model)  # type: ignore[arg-type]

    optimizer.step()

    assert _parameter_update_l2_norm(snapshots) == pytest.approx(2**0.5 * 0.1)


def test_ppo_clip_debug_counts_token_observations() -> None:
    trainer = object.__new__(RLOOTrainer)
    trainer._cfg = SimpleNamespace(  # type: ignore[attr-defined]
        params=SimpleNamespace(ppo_epsilon=0.1)
    )

    _loss, debug = trainer._ppo_loss(
        torch.tensor(1.0),
        torch.tensor([1.0, 1.2, 0.5]),
    )

    assert debug.total_observations == 3
    assert debug.clipped_observations == 1
    assert debug.clipped_fraction == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("exception", "method", "event"),
    (
        (torch.cuda.OutOfMemoryError("allocation failed"), "critical", "cuda_oom"),
        (RuntimeError("boom"), "error", "training_failed"),
        (KeyboardInterrupt(), "error", "training_failed"),
    ),
)
def test_training_failure_alert_is_structured_and_idempotent(
    monkeypatch, exception: BaseException, method: str, event: str
) -> None:
    fake_logger = Mock()
    monkeypatch.setattr(train_module, "logger", fake_logger)

    _log_training_failure(exception)
    _log_training_failure(exception)

    log_method = getattr(fake_logger, method)
    log_method.assert_called_once()
    assert log_method.call_args.kwargs["extra"] == {"event": event}
    if str(exception):
        assert str(exception) not in log_method.call_args.args[0]
