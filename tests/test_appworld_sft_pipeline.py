from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from phi_agents.api_eval.deepseek_appworld import (
    DeepSeekAbsoluteTimeout,
    absolute_deadline,
    repair_prefix_steps,
)
from phi_agents.sft.dataset import (
    DatasetBuildConfig,
    QualityDecision,
    assess_trajectory,
    assistant_step_mask_reason,
    build_dataset,
    trajectory_to_sample,
)
from phi_agents.sft.teacher import (
    TeacherGenerationConfig,
    _read_manifest,
    _task_progress,
    _write_manifest,
    generation_plan,
)
from phi_agents.sft.trainer import (
    AssistantOnlyCollator,
    TokenizedSFTDataset,
    audit_sample_windows,
    tokenize_messages,
    window_sample,
)


def _trajectory(task_id: str = "6104387_1", difficulty: int = 3) -> dict[str, Any]:
    steps = [
        {
            "step_index": 1,
            "model_output": "Code:\n```python\nprint('ok')\n```",
            "action_code": "print('ok')",
            "environment_result": "ok",
            "observation_sent_to_model": None,
            "execution_failed": False,
            "no_code_found": False,
            "task_completed": False,
            "reasoning_content": "private teacher reasoning",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        {
            "step_index": 2,
            "model_output": "Code:\n```python\napis.supervisor.complete_task()\n```",
            "action_code": "apis.supervisor.complete_task()",
            "environment_result": "Marked the active task complete.",
            "observation_sent_to_model": "ok\nAs a reminder: solve the task",
            "execution_failed": False,
            "no_code_found": False,
            "task_completed": True,
            "reasoning_content": "more private reasoning",
            "usage": {"prompt_tokens": 20, "completion_tokens": 6},
        },
    ]
    return {
        "status": "completed",
        "task_id": task_id,
        "success": True,
        "cancelled": False,
        "error_type": "",
        "cost_cny": 0.02,
        "task": {"task_id": task_id, "instruction": "Do a real task."},
        "initial_prompt_messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Demonstration action"},
            {"role": "user", "content": "Original task instruction"},
        ],
        "steps": steps,
        "eval_result": {
            "success": True,
            "difficulty": difficulty,
            "num_tests": 2,
            "passes": [{"label": "one"}, {"label": "two"}],
            "failures": [],
        },
    }


class CharacterChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize and not add_generation_prompt
        return "".join(f"<{row['role']}>\n{row['content']}\n</{row['role']}>\n" for row in messages)

    def __call__(
        self, text: str, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, Any]:
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
        }


def test_generation_plan_is_balanced_and_train_only(tmp_path: Path) -> None:
    config = TeacherGenerationConfig(
        split="train_difficulty_1_2",
        output_dir=tmp_path,
        successes_per_task=14,
        max_attempts_per_task=28,
        cost_limit_cny=20.0,
    )
    plan = generation_plan(config)
    assert plan["tasks"] == 72
    assert plan["target_successes"] == 1008
    difficulty_3_plan = generation_plan(
        TeacherGenerationConfig(
            split="train_difficulty_3",
            output_dir=tmp_path / "d3",
            successes_per_task=14,
            max_attempts_per_task=28,
            cost_limit_cny=20.0,
        )
    )
    assert difficulty_3_plan["tasks"] == 18
    assert difficulty_3_plan["target_successes"] == 252
    with pytest.raises(ValueError, match="only permits"):
        generation_plan(
            TeacherGenerationConfig(
                split="dev",
                output_dir=tmp_path,
                successes_per_task=1,
                max_attempts_per_task=1,
                cost_limit_cny=1,
            )
        )


def test_repair_prefix_stops_before_first_critical_error() -> None:
    trajectory = _trajectory()
    trajectory["steps"].insert(
        1,
        {
            "action_code": "bad()",
            "model_output": "bad",
            "environment_result": "Execution failed.",
            "execution_failed": True,
            "no_code_found": False,
        },
    )
    assert [row["action_code"] for row in repair_prefix_steps(trajectory)] == ["print('ok')"]


def test_absolute_deadline_is_not_reset_by_activity() -> None:
    started = time.monotonic()
    with pytest.raises(DeepSeekAbsoluteTimeout):
        with absolute_deadline(0.05):
            while True:
                time.sleep(0.01)
    assert time.monotonic() - started < 0.5


def test_generation_manifest_resumes_task_progress(tmp_path: Path) -> None:
    config = TeacherGenerationConfig(
        split="train_difficulty_1_2",
        output_dir=tmp_path,
        successes_per_task=1,
        max_attempts_per_task=2,
        cost_limit_cny=1.0,
        limit=1,
    )
    path = tmp_path / "generation_manifest.json"
    manifest = _read_manifest(path, config)
    manifest["jobs"].append({"task_id": "82e2fac_1", "success": True})
    _write_manifest(path, manifest)
    resumed = _read_manifest(path, config)
    assert _task_progress(resumed, "82e2fac_1") == (1, 1)


def test_quality_filter_and_conversion_exclude_reasoning_and_evaluator(tmp_path: Path) -> None:
    trajectory = _trajectory()
    decision = assess_trajectory(trajectory, {"6104387_1"})
    assert decision.accepted
    sample = trajectory_to_sample(
        trajectory,
        tmp_path / "trajectory.json",
        decision,
        DatasetBuildConfig((tmp_path,), tmp_path),
    )
    text = json.dumps(sample)
    assert "private teacher reasoning" not in text
    assert '"passes"' not in text
    assert '"failures"' not in text
    assert all(
        message["loss"] is (message["role"] == "assistant" and index >= 3)
        for index, message in enumerate(sample["messages"])
    )
    assert sample["metadata"]["partial_pass"] == 1.0
    action_messages = [
        message
        for message in sample["messages"]
        if message.get("message_type") == "appworld_action"
    ]
    assert all(message["step_id"].startswith("6104387_1:traj-") for message in action_messages)
    assert len({message["step_id"] for message in action_messages}) == 2


def test_conversion_masks_execution_failure_and_keeps_recovery(tmp_path: Path) -> None:
    trajectory = _trajectory()
    trajectory["steps"].insert(
        1,
        {
            "step_index": 99,
            "action_code": "bad_call()",
            "model_output": "Code:\n```python\nbad_call()\n```",
            "environment_result": "Execution failed: bad call",
            "observation_sent_to_model": "ok",
            "execution_failed": True,
            "no_code_found": False,
            "task_completed": False,
            "finish_reason": "stop",
        },
    )
    trajectory["steps"][2]["observation_sent_to_model"] = (
        "Execution failed: bad call\nAs a reminder: solve the task"
    )
    sample = trajectory_to_sample(
        trajectory,
        tmp_path / "trajectory.json",
        QualityDecision(True, [], "recovered_success", "fingerprint"),
        DatasetBuildConfig((tmp_path,), tmp_path),
    )
    actions = [
        message
        for message in sample["messages"]
        if message.get("message_type") == "appworld_action"
    ]
    assert [message["loss"] for message in actions] == [True, False, True]
    assert actions[1]["mask_reason"] == "execution_failed"
    assert any(
        message.get("message_type") == "appworld_observation"
        and "Execution failed" in message["content"]
        for message in sample["messages"]
    )
    assert actions[2]["loss"] is True


def test_invalid_responses_before_valid_action_are_not_serialized(tmp_path: Path) -> None:
    trajectory = _trajectory()
    trajectory["steps"][0]["invalid_responses_before_action"] = [
        {"content": "THIS INVALID OUTPUT MUST NOT BECOME A TARGET", "usage": {}}
    ]
    decision = assess_trajectory(trajectory, {"6104387_1"})
    sample = trajectory_to_sample(
        trajectory,
        tmp_path / "trajectory.json",
        decision,
        DatasetBuildConfig((tmp_path,), tmp_path),
    )
    assert "THIS INVALID OUTPUT" not in json.dumps(sample)
    actions = [
        message
        for message in sample["messages"]
        if message.get("message_type") == "appworld_action"
    ]
    assert actions[0]["loss"] is True


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        ({"no_code_found": True}, "no_code"),
        ({"model_output": ""}, "empty_response"),
        ({"action_code": ""}, "empty_action"),
        ({"action_code": "if:"}, "unparseable_code"),
        ({"infrastructure_error": True}, "infrastructure_error"),
        ({"finish_reason": "length"}, "incomplete_generation:length"),
    ],
)
def test_structured_bad_action_mask_reasons(step: dict[str, Any], reason: str) -> None:
    base = {
        "model_output": "```python\nprint(1)\n```",
        "action_code": "print(1)",
        "execution_failed": False,
        "no_code_found": False,
        "finish_reason": "stop",
    }
    assert assistant_step_mask_reason({**base, **step}) == reason


def test_dataset_rejects_dev_and_splits_by_scenario(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    task_ids = ["6104387_1", "afc0fce_1"]
    for index, task_id in enumerate(task_ids):
        trajectory = _trajectory(task_id)
        trajectory["steps"][0]["action_code"] = f"print({index})"
        trajectory["steps"][0]["model_output"] = f"Code:\n```python\nprint({index})\n```"
        path = raw / task_id / f"attempt-{index}" / "trajectory.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(trajectory))
    dev = _trajectory("50e1ac9_1", difficulty=1)
    dev_path = raw / "50e1ac9_1" / "attempt-0" / "trajectory.json"
    dev_path.parent.mkdir(parents=True)
    dev_path.write_text(json.dumps(dev))

    output = tmp_path / "dataset"
    report = build_dataset(
        DatasetBuildConfig((raw,), output, mode="difficulty_1_2_3", validation_fraction=0.5)
    )
    assert report["accepted"] == 2
    assert report["rejection_reasons"]["task_not_in_allowed_train_split"] == 1
    train = [
        json.loads(line) for line in (output / "qwen_sft_train.jsonl").read_text().splitlines()
    ]
    validation = [
        json.loads(line) for line in (output / "qwen_sft_validation.jsonl").read_text().splitlines()
    ]
    train_scenarios = {row["metadata"]["scenario_id"] for row in train}
    validation_scenarios = {row["metadata"]["scenario_id"] for row in validation}
    assert train_scenarios.isdisjoint(validation_scenarios)
    assert all(not row["metadata"]["task_id"].startswith("50e1ac9") for row in train + validation)


def _window_messages(
    actions: list[tuple[str, str, bool, str | None]], *, task: str = "TASK"
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "SYS", "loss": False},
        {"role": "user", "content": task, "loss": False},
    ]
    for index, (action, observation, loss, mask_reason) in enumerate(actions):
        step_id = f"task:trajectory:{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": action,
                    "loss": loss,
                    "mask_reason": mask_reason,
                    "message_type": "appworld_action",
                    "step_id": step_id,
                    "step_index": index,
                },
                {
                    "role": "user",
                    "content": observation,
                    "loss": False,
                    "message_type": "appworld_observation",
                    "step_id": step_id,
                },
            ]
        )
    return messages


def _supervised_text(window: dict[str, Any]) -> str:
    return "".join(
        chr(token)
        for token, label in zip(window["input_ids"], window["labels"], strict=True)
        if label != -100
    )


def test_assistant_only_mask_and_unsplit_window_collator() -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages([("ACTION1", "OBS1", True, None), ("ACTION2", "OBS2", True, None)])
    tokenized = tokenize_messages(tokenizer, messages)
    assert _supervised_text(tokenized) == "ACTION1ACTION2"

    sample = {"messages": messages, "metadata": {"task_id": "x_1"}}
    windows = window_sample(tokenizer, sample, max_length=10_000)
    assert len(windows) == 1
    assert audit_sample_windows(sample, windows)["supervised_steps"] == 2
    batch = AssistantOnlyCollator(tokenizer.pad_token_id)([TokenizedSFTDataset(windows)[0]])
    assert torch.equal(batch["labels"][0], torch.tensor(windows[0]["labels"]))


def test_long_windows_keep_recent_history_without_duplicate_supervision() -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages([(f"ACTION{i}", f"OBS{i}", True, None) for i in range(4)])
    sample = {"messages": messages, "metadata": {"task_id": "task", "trajectory_id": "t"}}
    three_turn_length = len(tokenize_messages(tokenizer, messages[: 2 + 3 * 2])["input_ids"])
    windows = window_sample(tokenizer, sample, max_length=three_turn_length, turn_overlap=0)
    assert len(windows) == 2
    assert _supervised_text(windows[0]) == "ACTION0ACTION1ACTION2"
    assert _supervised_text(windows[1]) == "ACTION3"
    second_text = "\n".join(message["content"] for message in windows[1]["messages"])
    assert "OBS2" in second_text
    history_actions = [
        row
        for row in windows[1]["step_token_stats"]
        if str(row["window_role"]).startswith("history")
    ]
    assert history_actions
    assert all(row["supervised_tokens"] == 0 and not row["loss"] for row in history_actions)
    stats = audit_sample_windows(sample, windows)
    assert stats["supervised_steps"] == 4
    assert stats["history_repeated_steps"] >= 1


@pytest.mark.parametrize("turn_overlap", [0, 1, 2, 10])
def test_turn_overlap_only_changes_masked_history(turn_overlap: int) -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages([(f"A{i}", f"O{i}", True, None) for i in range(5)])
    sample = {"messages": messages, "metadata": {"task_id": "task", "trajectory_id": "t"}}
    max_length = len(tokenize_messages(tokenizer, messages[: 2 + 3 * 2])["input_ids"])
    windows = window_sample(tokenizer, sample, max_length=max_length, turn_overlap=turn_overlap)
    stats = audit_sample_windows(sample, windows)
    assert stats["supervised_steps"] == 5
    assert "".join(_supervised_text(window) for window in windows) == "A0A1A2A3A4"


def test_failed_and_no_code_actions_are_context_only_and_recovery_is_supervised() -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages(
        [
            ("BAD_ACTION", "Execution failed: bad", False, "execution_failed"),
            ("NO_CODE", "No code available", False, "no_code"),
            ("RECOVERY", "Execution successful", True, None),
        ]
    )
    sample = {"messages": messages, "metadata": {"task_id": "task", "trajectory_id": "t"}}
    windows = window_sample(tokenizer, sample, max_length=10_000)
    assert len(windows) == 1
    assert _supervised_text(windows[0]) == "RECOVERY"
    rendered = "\n".join(message["content"] for message in windows[0]["messages"])
    assert "BAD_ACTION" in rendered and "Execution failed" in rendered and "NO_CODE" in rendered
    by_id = {row["step_id"]: row for row in windows[0]["step_token_stats"]}
    assert by_id["task:trajectory:0"]["supervised_tokens"] == 0
    assert by_id["task:trajectory:1"]["supervised_tokens"] == 0
    assert by_id["task:trajectory:2"]["supervised_tokens"] == len("RECOVERY")
    stats = audit_sample_windows(sample, windows)
    assert stats["execution_failed_masked_steps"] == 1
    assert stats["no_code_masked_steps"] == 1


def test_supervision_audit_error_identifies_task_trajectory_step_and_windows() -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages([("ACTION", "OBS", True, None)])
    sample = {"messages": messages, "metadata": {"task_id": "task", "trajectory_id": "traj"}}
    window = window_sample(tokenizer, sample, max_length=10_000)[0]
    duplicate = {**window, "window_index": 1}
    with pytest.raises(ValueError) as error:
        audit_sample_windows(sample, [window, duplicate])
    text = str(error.value)
    assert "task_id=task" in text
    assert "trajectory_id=traj" in text
    assert "step_id=task:trajectory:0" in text
    assert "occurrences=2" in text and "supervision=2" in text
    assert "window" in text


def test_history_drops_oldest_turn_but_keeps_task_and_previous_observation() -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages(
        [
            ("A0" * 20, "OLD_OBS" * 20, True, None),
            ("A1" * 10, "RECENT_OBS", True, None),
            ("A2" * 20, "OBS2", True, None),
        ],
        task="IMPORTANT_TASK",
    )
    prompt_and_two = messages[: 2 + 2 * 2]
    max_length = len(tokenize_messages(tokenizer, prompt_and_two)["input_ids"])
    windows = window_sample(
        tokenizer,
        sample={"messages": messages, "metadata": {"task_id": "t"}},
        max_length=max_length,
    )
    assert len(windows) == 2
    later = "\n".join(message["content"] for message in windows[1]["messages"])
    assert "IMPORTANT_TASK" in later
    assert "RECENT_OBS" in later
    assert "OLD_OBS" not in later


def test_single_complete_turn_over_max_length_errors() -> None:
    tokenizer = CharacterChatTokenizer()
    messages = _window_messages([("A" * 200, "O" * 200, True, None)])
    with pytest.raises(ValueError, match="One complete prompt\\+turn exceeds"):
        window_sample(
            tokenizer,
            {"messages": messages, "metadata": {"task_id": "task"}},
            max_length=100,
        )
