from types import SimpleNamespace

import numpy as np
import pytest

from phi_agents.rl.rl_utils import (
    Baseline,
    RLAlgorithm,
    compute_advantage_estimates,
    compute_grpo_advantages,
    compute_loop_advantages,
)


def _rollout(ret: float) -> SimpleNamespace:
    return SimpleNamespace(ret=ret)


def test_loop_leave_one_out_advantages_match_existing_behavior() -> None:
    rollouts = [[_rollout(1.0), _rollout(0.0), _rollout(0.0)]]

    advantages = compute_loop_advantages(
        rollouts,
        baseline=Baseline.LOO,
        adv_normalization=False,
    )

    np.testing.assert_allclose(advantages, np.array([1.0, -0.5, -0.5]))


def test_grpo_group_normalized_advantages() -> None:
    rollouts = [[_rollout(0.0), _rollout(2.0)]]

    advantages = compute_grpo_advantages(rollouts)

    np.testing.assert_allclose(advantages, np.array([-1.0, 1.0]))


def test_grpo_equal_returns_produce_zero_advantages() -> None:
    rollouts = [[_rollout(0.5), _rollout(0.5), _rollout(0.5)]]

    advantages = compute_grpo_advantages(rollouts)

    np.testing.assert_allclose(advantages, np.zeros(3))


def test_grpo_preserves_flattened_scenario_order() -> None:
    rollouts = [
        [_rollout(0.0), _rollout(2.0)],
        [_rollout(4.0), _rollout(6.0)],
    ]

    advantages = compute_advantage_estimates(
        rollouts,
        algorithm=RLAlgorithm.GRPO,
        baseline=Baseline.LOO,
        adv_normalization=False,
    )

    np.testing.assert_allclose(advantages, np.array([-1.0, 1.0, -1.0, 1.0]))


def test_grouped_advantages_require_at_least_two_rollouts_per_scenario() -> None:
    rollouts = [[_rollout(1.0)]]

    with pytest.raises(ValueError, match="At least two rollouts per scenario"):
        compute_grpo_advantages(rollouts)
