from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.loop7b.analyze_appworld_behavior import analyze

if TYPE_CHECKING:
    from pathlib import Path


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _write_episode(
    appworld_root: Path,
    *,
    experiment_name: str,
    task_id: str,
    success: bool,
    execution_failures: int | None,
    chat_history: list[dict[str, str]],
) -> None:
    episode: dict[str, Any] = {
        "task": {"task_id": task_id},
        "num_prompt_messages": 1,
        "chat_history": [_message("system", "prompt"), *chat_history],
        "eval_result": {
            "success": success,
            "num_tests": 2,
            "passes": ["one", "two"] if success else ["one"],
            "num_interactions": len(chat_history) // 2,
        },
        "n_no_code_found": 0,
        "context_truncated": False,
    }
    if execution_failures is not None:
        episode["n_execution_failed"] = execution_failures
    path = (
        appworld_root
        / "experiments"
        / "outputs"
        / experiment_name
        / "tasks"
        / task_id
        / "logs"
        / "episode.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode), encoding="utf-8")


def test_analyze_adds_error_observation_and_rollout_recovery_metrics(
    tmp_path: Path,
) -> None:
    experiment_name = "behavior_metrics"
    repeated_action = "apis.spotify.play_song(song_id='abc')"
    _write_episode(
        tmp_path,
        experiment_name=experiment_name,
        task_id="recovered_success",
        success=True,
        execution_failures=2,
        chat_history=[
            _message("assistant", f"```python\n{repeated_action}\n```"),
            _message("user", "Execution failed. HTTP 401. NameError: missing variable"),
            _message("assistant", f"```python\n  {repeated_action}  \n```"),
            _message("user", "Execution failed. status=422; nameerror: still missing"),
            _message("assistant", f"```python\n{repeated_action}\n```"),
            _message("user", "Song started."),
        ],
    )
    _write_episode(
        tmp_path,
        experiment_name=experiment_name,
        task_id="clean_success",
        success=True,
        execution_failures=0,
        chat_history=[
            _message("assistant", "```python\napis.supervisor.complete_task()\n```"),
            _message("user", "Done."),
        ],
    )
    _write_episode(
        tmp_path,
        experiment_name=experiment_name,
        task_id="unrecovered_failure",
        success=False,
        execution_failures=None,
        chat_history=[
            _message(
                "assistant",
                "```python\napis.spotify.get_song(song_id='missing')\n```",
            ),
            _message(
                "user",
                "Execution failed. HTTP 401 (account reference 1401 is not a status).",
            ),
        ],
    )

    row = analyze(
        appworld_root=tmp_path,
        experiment_name=experiment_name,
        all_results_path=None,
        checkpoint_name="checkpoint-15",
        checkpoint_iteration=None,
        eval_result_path="",
        eval_log_path="",
        behavior_summary_path="behavior.json",
    )

    assert row["num_rollouts_analyzed"] == 3
    assert row["execution_failed_count"] == 3
    assert row["http_401_count"] == 2
    assert row["http_422_count"] == 1
    assert row["name_error_count"] == 2
    assert row["consecutive_repeated_failed_action_count"] == 1
    assert row["average_execution_errors_before_strict_success"] == 1.0
    assert row["error_rollout_count"] == 2
    assert row["error_rollout_strict_success_count"] == 1
    assert row["error_rollout_recovery_success_rate"] == 0.5

    # Existing endpoint-retry recovery remains a separate call-level metric.
    assert row["failed_api_calls"] == 3
    assert row["recovered_api_calls"] == 2
    assert row["error_recovery_success_rate"] == pytest.approx(2 / 3)


def test_analyze_zero_denominators_are_explicit_zero(tmp_path: Path) -> None:
    experiment_name = "no_success_or_error"
    _write_episode(
        tmp_path,
        experiment_name=experiment_name,
        task_id="plain_failure",
        success=False,
        execution_failures=0,
        chat_history=[
            _message("assistant", "```python\napis.spotify.search(query='x')\n```"),
            _message("user", "No match."),
        ],
    )

    row = analyze(
        appworld_root=tmp_path,
        experiment_name=experiment_name,
        all_results_path=None,
        checkpoint_name="base",
        checkpoint_iteration=None,
        eval_result_path="",
        eval_log_path="",
        behavior_summary_path="behavior.json",
    )

    assert row["average_execution_errors_before_strict_success"] == 0.0
    assert row["error_rollout_recovery_success_rate"] == 0.0
