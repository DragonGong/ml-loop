from __future__ import annotations

import json
from pathlib import Path

from scripts.loop7b.analyze_sft_loop15_trajectories import (
    _divergence,
    _report_pass_set,
    _task_category,
    _trace_from_episode,
)


def _message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def test_trace_extracts_error_recovery_and_completion(tmp_path: Path) -> None:
    episode_path = tmp_path / "tasks" / "task_1" / "logs" / "episode.json"
    episode_path.parent.mkdir(parents=True)
    episode = {
        "task": {
            "task_id": "task_1",
            "instruction": "Send all messages.",
        },
        "num_prompt_messages": 1,
        "chat_history": [
            _message("system", "prompt"),
            _message(
                "assistant",
                "```python\nprint(apis.api_docs.show_api_doc(app_name='phone', api_name='send_text_message'))\n```",
            ),
            _message("user", "doc"),
            _message(
                "assistant",
                "```python\napis.phone.send_text_message(access_token=bad, phone_number='1', message='x')\n```",
            ),
            _message(
                "user",
                'Execution failed. Response status code is 401: {"message":"Unauthorized"}',
            ),
            _message(
                "assistant",
                "```python\napis.phone.send_text_message(access_token=good, phone_number='1', message='x')\n```",
            ),
            _message("user", "Execution successful."),
            _message("assistant", "```python\napis.supervisor.complete_task()\n```"),
            _message("user", "Execution successful."),
        ],
        "eval_result": {
            "success": True,
            "difficulty": 1,
            "num_tests": 1,
            "passes": [{"requirement": "assert answers match."}],
            "failures": [],
        },
        "n_execution_failed": 1,
        "context_truncated": False,
    }
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    trace = _trace_from_episode(episode_path, "synthetic")

    assert trace["first_business_api"] == "phone.send_text_message"
    assert trace["first_error_type"] == "401"
    assert "before a successful login" in trace["first_error_cause"]
    assert trace["recovered_after_first_error"] is True
    assert trace["changed_parameters_after_first_error"] is True
    assert trace["complete_task_called"] is True
    assert trace["error_events"][0]["endpoint"] == "phone.send_text_message"


def test_task_category_and_api_divergence() -> None:
    traces = {
        "r0": {"success": False, "partial_pass": 0.2},
        "ckpt-6": {"success": False, "partial_pass": 0.4},
        "ckpt-9": {"success": True, "partial_pass": 1.0},
        "ckpt-12": {"success": False, "partial_pass": 0.3},
        "ckpt-15": {"success": False, "partial_pass": 0.2},
    }
    assert _task_category(traces) == "sft_failed_loop_succeeded"

    left = {"business_api_sequence": ["spotify.login", "spotify.show_liked_songs"]}
    right = {"business_api_sequence": ["spotify.login", "spotify.show_liked_albums"]}
    assert _divergence(left, right) == (
        1,
        "spotify.show_liked_songs",
        "spotify.show_liked_albums",
    )


def test_report_parser_keeps_same_reward_pass_set_identity(tmp_path: Path) -> None:
    report = tmp_path / "runner_1" / "tasks" / "abc_1" / "evaluation" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text(
        """Num Passed Tests : 2
Num Failed Tests : 1
Num Total  Tests : 3
----- Passes -----
>> Passed Requirement
assert answer matches.
>> Passed Requirement
assert one record was added.
----- Fails -----
>> Failed Requirement
assert amount matches.
""",
        encoding="utf-8",
    )

    parsed = _report_pass_set(report)

    assert parsed["task_id"] == "abc_1"
    assert parsed["passed_count"] == 2
    assert parsed["total_count"] == 3
    assert parsed["pass_set"] == (
        "assert answer matches.",
        "assert one record was added.",
    )
