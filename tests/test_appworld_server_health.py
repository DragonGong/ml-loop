import os

import pytest

from phi_agents.rl.appworld_scenario_runner import (
    AppWorldServerDeadError,
    AppWorldServerHealthConfig,
    AppWorldServerHealthSnapshot,
    _process_tree_rss_gb,
)


def test_server_health_config_accepts_ordered_thresholds() -> None:
    cfg = AppWorldServerHealthConfig(
        enabled=True,
        warning_rss_gb=30.0,
        draining_rss_gb=45.0,
        dead_rss_gb=55.0,
    )

    assert cfg.enabled
    assert cfg.warning_rss_gb == 30.0
    assert cfg.draining_rss_gb == 45.0
    assert cfg.dead_rss_gb == 55.0


def test_server_health_config_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValueError):
        AppWorldServerHealthConfig(
            enabled=True,
            warning_rss_gb=45.0,
            draining_rss_gb=30.0,
            dead_rss_gb=55.0,
        )


def test_server_dead_error_carries_manifest_metadata() -> None:
    snapshot = AppWorldServerHealthSnapshot(
        status="dead",
        rss_gb=77.6,
        server_pid=2489447,
        server_port=37161,
        task_id="d0b1f43_3",
    )

    exc = AppWorldServerDeadError(snapshot)

    assert exc.reason == "appworld_server_rss_55gb"
    assert exc.server_pid == 2489447
    assert exc.server_port == 37161
    assert exc.task_id == "d0b1f43_3"
    assert exc.rss_gb == 77.6


def test_process_tree_rss_handles_current_process() -> None:
    assert _process_tree_rss_gb(os.getpid()) > 0


def test_process_tree_rss_handles_missing_process() -> None:
    assert _process_tree_rss_gb(-1) == 0.0
