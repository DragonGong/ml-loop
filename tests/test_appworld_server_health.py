import os

import pytest

from phi_agents.rl.appworld_scenario_runner import (
    AppWorldServerDeadError,
    AppWorldServerHealthConfig,
    AppWorldServerHealthSnapshot,
    AppWorldServerSoftDeadError,
    _process_tree_rss_gb,
)


def test_server_health_config_accepts_ordered_thresholds() -> None:
    cfg = AppWorldServerHealthConfig(
        enabled=True,
        recycle_rss_gb=25.0,
        draining_rss_gb=35.0,
        soft_dead_rss_gb=45.0,
        hard_dead_rss_gb=65.0,
        host_memory_hard_dead_percent=85.0,
        max_rollout_retries=2,
        max_soft_dead_per_iteration=8,
    )

    assert cfg.enabled
    assert cfg.recycle_rss_gb == 25.0
    assert cfg.draining_rss_gb == 35.0
    assert cfg.soft_dead_rss_gb == 45.0
    assert cfg.hard_dead_rss_gb == 65.0
    assert cfg.max_rollout_retries == 2
    assert cfg.max_soft_dead_per_iteration == 8


def test_server_health_config_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValueError):
        AppWorldServerHealthConfig(
            enabled=True,
            recycle_rss_gb=35.0,
            draining_rss_gb=25.0,
            soft_dead_rss_gb=45.0,
            hard_dead_rss_gb=65.0,
        )


def test_server_dead_error_carries_manifest_metadata() -> None:
    snapshot = AppWorldServerHealthSnapshot(
        status="hard_dead",
        rss_gb=77.6,
        host_memory_used_percent=60.0,
        server_pid=2489447,
        server_port=37161,
        task_id="d0b1f43_3",
    )

    exc = AppWorldServerDeadError(snapshot)

    assert exc.reason == "appworld_server_rss_65gb"
    assert exc.is_hard_dead
    assert exc.server_pid == 2489447
    assert exc.server_port == 37161
    assert exc.task_id == "d0b1f43_3"
    assert exc.rss_gb == 77.6


def test_server_soft_dead_error_carries_retry_metadata() -> None:
    snapshot = AppWorldServerHealthSnapshot(
        status="soft_dead",
        rss_gb=50.0,
        host_memory_used_percent=60.0,
        server_pid=2489447,
        server_port=37161,
        task_id="d0b1f43_3",
    )

    exc = AppWorldServerSoftDeadError(snapshot)

    assert exc.reason == "appworld_server_rss_45gb_soft_dead"
    assert exc.is_soft_dead
    assert exc.server_pid == 2489447
    assert exc.server_port == 37161
    assert exc.task_id == "d0b1f43_3"
    assert exc.rss_gb == 50.0


def test_process_tree_rss_handles_current_process() -> None:
    assert _process_tree_rss_gb(os.getpid()) > 0


def test_process_tree_rss_handles_missing_process() -> None:
    assert _process_tree_rss_gb(-1) == 0.0
