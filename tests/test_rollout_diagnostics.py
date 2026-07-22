import ast
import json
import stat
from datetime import datetime
from pathlib import Path

import pytest

from phi_agents.appworld.interface import Failure, Pass, Supervisor, Task
from phi_agents.evals.appworld_evals import TaskEvalResult
from phi_agents.evals.appworld_rollout_data import (
    AppWorldRolloutData,
    AppWorldTrainingRollout,
)
from phi_agents.rl.rollout_diagnostics import (
    _redact_sensitive_text,
    compute_rollout_diagnostics,
    sanitized_trajectory_payload,
    save_rollout_diagnostics,
    save_sanitized_trajectories,
)
from phi_agents.rl.type_defs import PolicyMessage, PolicyTokenInfo, SystemMessage, UserMessage


def _assistant(content: str) -> PolicyMessage:
    return PolicyMessage(
        content,
        prompt_tokens=[7001],
        generated_tokens=[8001, 8002],
        generated_token_logprobs=[-0.1, -0.2],
        stopped_by_max_tokens_limit=False,
    )


def _rollout(
    *,
    task_id: str,
    ret: float,
    success: bool,
    passes: int,
    turns: list[tuple[str, str]],
    output_tokens: int = 2,
    execution_failed: int = 0,
    no_code: int = 0,
    truncated: bool = False,
) -> AppWorldTrainingRollout:
    messages = [SystemMessage("system prompt"), UserMessage("task instruction")]
    for action, observation in turns:
        messages.extend([_assistant(action), UserMessage(observation)])
    task = Task(
        task_id=task_id,
        datetime=datetime.fromisoformat("2026-07-18T00:00:00"),
        instruction="task instruction",
        supervisor=Supervisor("Private", "Person", "15555550123", "private@example.com"),
    )
    eval_result = TaskEvalResult(
        success=success,
        difficulty=2,
        num_tests=2,
        passes=[Pass(f"hidden pass requirement {index}", None) for index in range(passes)],
        failures=[Failure("hidden failure requirement", "hidden trace", None)],
        num_interactions=len(turns),
    )
    token_count = max(output_tokens, 1)
    token_info = PolicyTokenInfo(
        tokens=list(range(token_count)),
        log_probs=[-0.5] * token_count,
        is_output=[index < output_tokens for index in range(token_count)],
    )
    return AppWorldTrainingRollout(
        messages=messages,
        ret=ret,
        elapsed=1.25,
        cancelled=False,
        policy_token_info=token_info,
        appworld_rollout_data=AppWorldRolloutData(
            task=task,
            eval_result=eval_result,
            dataset_name="train_difficulty_1_2",
            num_prompt_messages=2,
            n_execution_failed=execution_failed,
            n_no_code_found=no_code,
            context_truncated=truncated,
        ),
    )


def test_grouped_return_advantage_and_api_diversity_metrics() -> None:
    common_turn = [("```python\napis.spotify.search_tracks(q='x')\n```", "ok")]
    groups = [
        [
            _rollout(task_id="success_1", ret=1.0, success=True, passes=2, turns=common_turn)
            for _ in range(3)
        ],
        [
            _rollout(task_id="failure_1", ret=0.0, success=False, passes=0, turns=common_turn)
            for _ in range(3)
        ],
        [
            _rollout(task_id="partial_1", ret=0.5, success=False, passes=1, turns=common_turn)
            for _ in range(3)
        ],
        [
            _rollout(
                task_id="variable_1",
                ret=ret,
                success=ret == 1.0,
                passes=int(ret * 2),
                turns=[(f"```python\napis.todoist.action_{index}()\n```", "ok")],
                output_tokens=index + 1,
            )
            for index, ret in enumerate((0.0, 0.5, 1.0))
        ],
    ]

    result = compute_rollout_diagnostics(groups)

    assert result["num_groups"] == 4
    assert result["num_rollouts"] == 12
    assert result["zero_return_std_group_rate"] == pytest.approx(0.75)
    assert result["zero_return_group_class_counts"] == {
        "all_success": 1,
        "all_failure": 1,
        "partial_same": 1,
        "non_zero_variance": 1,
    }
    assert result["mean_unique_return_count_per_group"] == pytest.approx(1.5)
    assert result["effective_advantage_rollout_count"] == 2
    assert result["effective_advantage_rollout_ratio"] == pytest.approx(2 / 12)
    assert result["effective_advantage_output_tokens"] == 4
    assert result["total_output_tokens"] == 24
    assert result["effective_advantage_token_ratio"] == pytest.approx(1 / 6)
    variable = result["groups"][3]
    assert variable["unique_api_sequence_count"] == 3
    assert variable["unique_business_api_sequence_count"] == 3
    assert variable["same_business_api_sequence_ratio"] == 0.0


def test_behavior_error_and_recovery_metrics() -> None:
    repeated = "```python\napis.spotify.bad_endpoint()\n```"
    rollout = _rollout(
        task_id="behavior_1",
        ret=1.0,
        success=True,
        passes=2,
        execution_failed=2,
        no_code=1,
        truncated=True,
        turns=[
            (repeated, "Execution failed. HTTP 401. NameError. invalid API"),
            (
                repeated,
                "Execution failed. status=422 AttributeError: apis.spotify has no attribute",
            ),
            (repeated, "success"),
            (
                "```python\napis.api_docs.show_api_doc(app_name='spotify', api_name='x')\n"
                "apis.api_docs.show_api_descriptions(app_name='spotify')\n```",
                "done",
            ),
        ],
    )

    result = compute_rollout_diagnostics([[rollout, rollout]])
    behavior = result["behavior"]

    assert behavior["execution_failed_count"] == 4
    assert behavior["no_code_found_count"] == 2
    assert behavior["http_401_count"] == 2
    assert behavior["http_422_count"] == 2
    assert behavior["name_error_count"] == 2
    assert behavior["invalid_api_hits"] == 6
    assert behavior["api_doc_calls"] == 2
    assert behavior["api_description_calls"] == 2
    assert behavior["consecutive_repeated_failed_action_count"] == 2
    assert behavior["failed_api_calls"] == 4
    assert behavior["recovered_api_calls"] == 4
    assert behavior["error_recovery_success_rate"] == 1.0
    assert behavior["error_rollout_recovery_success_rate"] == 1.0
    assert behavior["average_turn_count"] == 4.0
    assert behavior["context_truncation_ratio"] == 1.0
    assert behavior["average_execution_errors_before_strict_success"] == 2.0


def test_sanitized_trajectory_omits_privileged_and_token_data() -> None:
    rollout = _rollout(
        task_id="safe_1",
        ret=0.5,
        success=False,
        passes=1,
        turns=[
            (
                "```python\napis.spotify.search_tracks(q='x')\n```",
                "Authorization: Bearer super-secret\npassword='also-secret'\n"
                'metadata={"token": "third-secret", "cookie": "fourth-secret"}',
            )
        ],
    )

    payload = sanitized_trajectory_payload(rollout, iteration=7, scenario_idx=2, rollout_idx=5)
    serialized = json.dumps(payload)

    assert payload["metadata"]["partial_pass"] == 0.5
    assert payload["metadata"]["output_token_count"] == 2
    assert "hidden pass requirement" not in serialized
    assert "hidden failure requirement" not in serialized
    assert "hidden trace" not in serialized
    assert "private@example.com" not in serialized
    assert "generated_tokens" not in serialized
    assert "prompt_tokens" not in serialized
    assert "log_probs" not in serialized
    assert "super-secret" not in serialized
    assert "also-secret" not in serialized
    assert "third-secret" not in serialized
    assert "fourth-secret" not in serialized
    assert "[REDACTED]" in serialized


def test_redaction_preserves_python_delimiters_and_removes_credentials_and_pii() -> None:
    jwt = "eyJheaderpart123.payloadpart123.signaturepart123"
    source = (
        "fs_password = 'literal-password'\n"
        f"phone_token = '{jwt}'\n"
        "result = login(password=fs_password, access_token=phone_token)\n"
        "payload = {'password': 'nested-password', 'access_token': phone_token}\n"
    )

    safe = _redact_sensitive_text(source)

    ast.parse(safe)
    assert "literal-password" not in safe
    assert "nested-password" not in safe
    assert jwt not in safe
    assert "fs_password = '[REDACTED]'" in safe
    assert 'access_token="[REDACTED]")' in safe
    assert "'password': '[REDACTED]'" in safe

    prose = _redact_sensitive_text(
        "Found venmo password: plain-secret) My name is: Alice Smith. "
        "email=alice@example.com phone=+86 138-1234-5678"
    )
    assert prose.endswith("phone=[REDACTED_PHONE]")
    assert "plain-secret" not in prose
    assert "Alice Smith" not in prose
    assert "alice@example.com" not in prose


def test_trajectory_literal_registry_redacts_unlabelled_repeated_secret() -> None:
    repeated_secret = "same-secret-everywhere"
    rollout = _rollout(
        task_id="safe_registry_1",
        ret=0.0,
        success=False,
        passes=0,
        turns=[
            (
                "```python\nprint(apis.supervisor.show_account_passwords())\n```",
                f'{{"password": "{repeated_secret}", "first_name": "Alice"}}\n'
                f"{repeated_secret}\nAlice",
            )
        ],
    )

    payload = sanitized_trajectory_payload(rollout, iteration=1, scenario_idx=0, rollout_idx=0)
    serialized = json.dumps(payload)

    assert repeated_secret not in serialized
    assert "Alice" not in serialized
    assert payload["schema_version"] == "appworld-sanitized-trajectory-v2"


def test_independent_trajectory_paths_and_no_overwrite(tmp_path: Path) -> None:
    grouped = [
        [
            _rollout(
                task_id="same_task_1",
                ret=float(index),
                success=bool(index),
                passes=index * 2,
                turns=[("```python\napis.todoist.show_tasks()\n```", "ok")],
            )
            for index in range(2)
        ]
    ]

    paths = save_sanitized_trajectories(grouped, output_root=tmp_path / "trajectories", iteration=3)

    assert paths == [
        tmp_path / "trajectories/iteration-000003/scenario-0000/rollout-00/trajectory.json",
        tmp_path / "trajectories/iteration-000003/scenario-0000/rollout-01/trajectory.json",
    ]
    assert all(path.is_file() for path in paths)
    assert all(len(path.read_text(encoding="utf-8").splitlines()) == 1 for path in paths)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)
    assert json.loads(paths[0].read_text())["rollout_idx"] == 0
    assert json.loads(paths[1].read_text())["rollout_idx"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_sanitized_trajectories(grouped, output_root=tmp_path / "trajectories", iteration=3)


def test_real_rollout_indices_and_seeds_survive_out_of_order_completion(tmp_path: Path) -> None:
    scenario_zero = [
        _rollout(
            task_id="scenario_zero",
            ret=ret,
            success=bool(ret),
            passes=int(ret * 2),
            turns=[("```python\napis.todoist.show_tasks()\n```", "ok")],
        )
        for ret in (1.0, 0.0)
    ]
    scenario_one = [
        _rollout(
            task_id="scenario_one",
            ret=ret,
            success=bool(ret),
            passes=int(ret * 2),
            turns=[("```python\napis.spotify.search_tracks(q='x')\n```", "ok")],
        )
        for ret in (0.5, 0.0)
    ]
    # Lists are in completion order, not generation order. Both scenarios independently use
    # rollout indices 0 and 1.
    scenario_zero[0].rollout_idx = 1
    scenario_zero[0].generation_seed = 101
    scenario_zero[1].rollout_idx = 0
    scenario_zero[1].generation_seed = 100
    scenario_one[0].rollout_idx = 1
    scenario_one[0].generation_seed = 201
    scenario_one[1].rollout_idx = 0
    scenario_one[1].generation_seed = 200

    grouped = [scenario_zero, scenario_one]
    diagnostics = compute_rollout_diagnostics(grouped)
    paths = save_sanitized_trajectories(grouped, output_root=tmp_path / "ordered", iteration=4)

    assert [row["rollout_idx"] for row in diagnostics["groups"][0]["rollouts"]] == [0, 1]
    assert [row["generation_seed"] for row in diagnostics["groups"][0]["rollouts"]] == [
        100,
        101,
    ]
    assert diagnostics["groups"][0]["returns"] == [0.0, 1.0]
    assert len(paths) == 4
    first = json.loads(
        (tmp_path / "ordered/iteration-000004/scenario-0000/rollout-00/trajectory.json").read_text()
    )
    second_scenario = json.loads(
        (tmp_path / "ordered/iteration-000004/scenario-0001/rollout-00/trajectory.json").read_text()
    )
    assert first["metadata"]["generation_seed"] == 100
    assert first["metadata"]["return"] == 0.0
    assert second_scenario["metadata"]["generation_seed"] == 200


def test_rollout_index_validation_is_per_scenario() -> None:
    rollouts = [
        _rollout(
            task_id="invalid_1",
            ret=0.0,
            success=False,
            passes=0,
            turns=[("```python\napis.todoist.show_tasks()\n```", "ok")],
        )
        for _ in range(2)
    ]
    rollouts[0].rollout_idx = 0
    rollouts[1].rollout_idx = 0
    with pytest.raises(ValueError, match="duplicate rollout_idx"):
        compute_rollout_diagnostics([rollouts])

    rollouts[1].rollout_idx = 2
    with pytest.raises(ValueError, match="outside the expected range"):
        compute_rollout_diagnostics([rollouts])


def test_diagnostics_json_persistence_and_group_validation(tmp_path: Path) -> None:
    rollout = _rollout(
        task_id="task_1",
        ret=0.0,
        success=False,
        passes=0,
        turns=[("```python\napis.todoist.show_tasks()\n```", "ok")],
    )
    diagnostics = compute_rollout_diagnostics([[rollout, rollout]])
    output_path = tmp_path / "diagnostics.json"

    assert save_rollout_diagnostics(diagnostics, output_path=output_path) == output_path
    assert json.loads(output_path.read_text())["schema_version"] == diagnostics["schema_version"]
    with pytest.raises(ValueError, match="at least two"):
        compute_rollout_diagnostics([[rollout]])
