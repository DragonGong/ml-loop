#!/usr/bin/env python3
"""Build the offline SFT->LOOP 15-iteration comparison report.

This script never launches inference or training.  It preserves every expected
state/iteration even when artifacts are absent, validates evaluation
comparability, and writes equivalent JSON, CSV, and Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from scripts.loop7b.run_sft_loop15_evaluations import DEFAULT_ROLLOUT_SEEDS
except ModuleNotFoundError:
    # Support both ``python -m`` and direct execution from the repository root.
    from run_sft_loop15_evaluations import DEFAULT_ROLLOUT_SEEDS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


SCHEMA_VERSION = "appworld-sft-loop15-report-v1"
DEV_SPLIT = "dev_small64"
DIAGNOSTIC_SPLIT = "train_difficulty_1_2"
ITERATIONS = (0, 5, 10, 15)
CAPABILITY_METRICS = (
    "TGC",
    "SGC",
    "average_partial_pass_rate",
    "TGC_1",
    "TGC_2",
    "TGC_3",
    "SGC_1",
    "SGC_2",
    "SGC_3",
    "execution_failed_count",
    "no_code_found_count",
    "http_401_count",
    "http_422_count",
    "name_error_count",
    "invalid_api_hits",
    "consecutive_repeated_failed_action_count",
    "average_execution_errors_before_strict_success",
    "api_doc_calls_per_rollout",
    "api_description_calls_per_rollout",
    "api_doc_or_description_calls_per_rollout",
    "num_turns_avg",
    "context_truncation_ratio",
    "failed_api_calls",
    "recovered_api_calls",
    "failed_api_call_give_up_rate",
    "error_recovery_success_rate",
    "error_rollout_count",
    "error_rollout_strict_success_count",
    "error_rollout_recovery_success_rate",
)
CURVE_METRICS = ("TGC", "SGC", "average_partial_pass_rate")
DEV_SPECS = (
    ("r1", "r0", 0, "r0"),
    ("r1", "r1-5", 5, "checkpoint-5"),
    ("r1", "r1-10", 10, "checkpoint-10"),
    ("r1", "r1-15", 15, "checkpoint-15"),
    ("r2", "base", 0, "base"),
    ("r2", "r2-5", 5, "checkpoint-5"),
    ("r2", "r2-10", 10, "checkpoint-10"),
    ("r2", "r2-15", 15, "checkpoint-15"),
)
DIAGNOSTIC_SPECS = (
    ("r1", "r0", 0),
    ("r1", "r1-5", 5),
    ("r1", "r1-10", 10),
    ("r1", "r1-15", 15),
    ("r2", "base", 0),
    ("r2", "r2-5", 5),
    ("r2", "r2-10", 10),
    ("r2", "r2-15", 15),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _task_ids_sha256(task_ids: Iterable[str]) -> str:
    encoded = "".join(f"{task_id}\n" for task_id in task_ids).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_task_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _nested(payload: Any, *keys: str, default: Any = None) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _delta(after: Any, before: Any) -> float | None:
    after_number = _number(after)
    before_number = _number(before)
    if after_number is None or before_number is None:
        return None
    return float(after_number) - float(before_number)


def normalized_trapezoidal_auc(
    rows: Sequence[dict[str, Any]], metric: str
) -> tuple[float | None, list[int]]:
    """Return trapezoidal AUC divided by the 0--15 iteration interval."""
    by_iteration = {int(row["iteration"]): _number(row.get(metric)) for row in rows}
    missing = [iteration for iteration in ITERATIONS if by_iteration.get(iteration) is None]
    if missing:
        return None, missing
    area = 0.0
    for left, right in zip(ITERATIONS, ITERATIONS[1:], strict=False):
        left_value = float(by_iteration[left])
        right_value = float(by_iteration[right])
        area += (left_value + right_value) * 0.5 * (right - left)
    return area / (ITERATIONS[-1] - ITERATIONS[0]), []


def _issue(issues: list[dict[str, Any]], severity: str, scope: str, message: str) -> None:
    issues.append({"severity": severity, "scope": scope, "message": message})


def _safe_load(
    path: Path, *, issues: list[dict[str, Any]], scope: str, missing_ok: bool = True
) -> Any | None:
    if not path.is_file():
        _issue(
            issues,
            "missing" if missing_ok else "error",
            scope,
            f"artifact is absent: {path}",
        )
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        _issue(issues, "error", scope, f"artifact is not valid JSON: {path}: {exc}")
        return None


def _row_status(issues: Sequence[dict[str, Any]], scope: str, *, available: bool) -> str:
    severities = {item["severity"] for item in issues if item["scope"] == scope}
    if "error" in severities:
        return "fail"
    if not available or "missing" in severities:
        return "missing"
    if "warning" in severities:
        return "warning"
    return "pass"


def _base_capability_row(branch: str, state: str, iteration: int, source: Path) -> dict[str, Any]:
    return {
        "record_type": "capability",
        "branch": branch,
        "state": state,
        "iteration": iteration,
        "available": False,
        "historical_unpaired_seed": False,
        "comparability_status": "missing",
        "split": DEV_SPLIT,
        "task_count": None,
        "source": str(source),
        **{metric: None for metric in CAPABILITY_METRICS},
    }


def _copy_capability_metrics(row: dict[str, Any], payload: dict[str, Any]) -> None:
    for metric in CAPABILITY_METRICS:
        row[metric] = _number(payload.get(metric))


def _command_option(command: Sequence[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _validate_dev_provenance(
    *,
    state: str,
    output_dir: Path,
    issues: list[dict[str, Any]],
) -> None:
    scope = f"dev:{state}"
    launch = _safe_load(
        output_dir / "orchestrator_job.json", issues=issues, scope=scope, missing_ok=False
    )
    completion = _safe_load(
        output_dir / "orchestrator_complete.json", issues=issues, scope=scope, missing_ok=False
    )
    if not isinstance(launch, dict):
        return
    command = launch.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        _issue(issues, "error", scope, "orchestrator command provenance is invalid")
        return
    expected_options = {
        "--split": DEV_SPLIT,
        "--num-scenario-runners": "16",
        "--llm": "qwen_2_5_7b_lora16_eval",
        "--max-gpu-mem-utilization": "0.90",
        "--max-model-len": "16384",
        "--max-new-tokens": "1200",
    }
    for option, expected in expected_options.items():
        actual = _command_option(command, option)
        if actual != expected:
            _issue(issues, "error", scope, f"{option}={actual!r}; expected {expected!r}")
    required_flags = (
        "--once",
        "--no-wait-for-gpu-idle",
        "--no-run-base-if-missing",
        "--no-eager-mode",
        "--no-reuse-complete-inference",
    )
    for flag in required_flags:
        if flag not in command:
            _issue(issues, "error", scope, f"required evaluation flag is absent: {flag}")
    hydra_overrides = [
        command[index + 1] for index, item in enumerate(command[:-1]) if item == "--hydra-override"
    ]
    for expected in (
        "llm.temperature=0.1",
        "scenario_runner.appworld_config.env.max_interactions=50",
        "eval_seed=20260718",
        "llm.vllm_server.seed=20260718",
        "scenario_sampler.seed=20260718",
        "rollout_seeds=[2026071800]",
    ):
        if expected not in hydra_overrides:
            _issue(issues, "error", scope, f"required Hydra override is absent: {expected}")
    command_hash = _sha256_json(command)
    if launch.get("command_sha256") != command_hash:
        _issue(issues, "error", scope, "orchestrator launch command hash does not match")
    if not isinstance(completion, dict) or completion.get("command_sha256") != command_hash:
        _issue(issues, "error", scope, "completion marker does not match the launch command")


def _load_capability_rows(
    *,
    r0_metrics: Path,
    evaluation_root: Path,
    expected_dev_count: int | None,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, state, iteration, checkpoint_name in DEV_SPECS:
        if state == "r0":
            row = _base_capability_row(branch, state, iteration, r0_metrics)
            payload = _safe_load(r0_metrics, issues=issues, scope="dev:r0")
            if isinstance(payload, dict):
                row["available"] = True
                row["historical_unpaired_seed"] = True
                row["task_count"] = _number(payload.get("num_rollouts_analyzed"))
                _copy_capability_metrics(row, payload)
                if any(row[metric] is None for metric in CURVE_METRICS):
                    _issue(issues, "error", "dev:r0", "R0 primary metrics are incomplete")
                if payload.get("checkpoint_name") != "d12_100_x1":
                    _issue(issues, "error", "dev:r0", "R0 checkpoint label is not d12_100_x1")
                if expected_dev_count is not None and row["task_count"] != expected_dev_count:
                    _issue(issues, "error", "dev:r0", "R0 task count differs from dev_small64")
                experiment_name = str(payload.get("experiment_name") or "")
                if DEV_SPLIT not in experiment_name:
                    _issue(issues, "error", "dev:r0", "R0 experiment does not identify dev_small64")
                _issue(
                    issues,
                    "warning",
                    "dev:r0",
                    "historical_unpaired_seed=true: legacy R0 used eval_seed=0 and an unset "
                    "vLLM server seed, while new dev states use paired eval/scenario/request seeds",
                )
            row["comparability_status"] = _row_status(issues, "dev:r0", available=row["available"])
            rows.append(row)
            continue

        output_dir = evaluation_root / "dev_small64" / state
        summary_path = output_dir / "summary.json"
        row = _base_capability_row(branch, state, iteration, summary_path)
        payload = _safe_load(summary_path, issues=issues, scope=f"dev:{state}")
        if isinstance(payload, list):
            matches = [
                item
                for item in payload
                if isinstance(item, dict)
                and item.get("split") == DEV_SPLIT
                and item.get("checkpoint_name") == checkpoint_name
            ]
            if len(matches) != 1:
                _issue(
                    issues,
                    "error",
                    f"dev:{state}",
                    f"expected one {checkpoint_name!r} dev_small64 row; found {len(matches)}",
                )
            else:
                result = matches[0]
                row["available"] = True
                row["task_count"] = _number(result.get("episode_count"))
                _copy_capability_metrics(row, result)
                if any(row[metric] is None for metric in CURVE_METRICS):
                    _issue(
                        issues,
                        "error",
                        f"dev:{state}",
                        "primary capability metrics are incomplete",
                    )
                if expected_dev_count is not None and row["task_count"] != expected_dev_count:
                    _issue(
                        issues,
                        "error",
                        f"dev:{state}",
                        "episode count differs from the canonical dev_small64 task list",
                    )
                _validate_dev_provenance(state=state, output_dir=output_dir, issues=issues)
        elif payload is not None:
            _issue(issues, "error", f"dev:{state}", "summary JSON must be a list")
        row["comparability_status"] = _row_status(
            issues, f"dev:{state}", available=row["available"]
        )
        rows.append(row)
    return rows


def _load_fixed_task_contract(
    manifest_path: Path, issues: list[dict[str, Any]]
) -> tuple[list[str], str | None]:
    payload = _safe_load(
        manifest_path, issues=issues, scope="diagnostic:manifest", missing_ok=False
    )
    if not isinstance(payload, dict):
        return [], None
    expected_contract = (1, "appworld_loop_scenario_manifest", DIAGNOSTIC_SPLIT, 15, 24, 6)
    observed_contract = (
        payload.get("schema_version"),
        payload.get("kind"),
        payload.get("dataset_name"),
        payload.get("num_iterations"),
        payload.get("scenarios_per_iteration"),
        payload.get("rollouts_per_scenario"),
    )
    if observed_contract != expected_contract:
        _issue(
            issues,
            "error",
            "diagnostic:manifest",
            f"manifest contract differs: {observed_contract!r}",
        )
        return [], None
    iterations = payload.get("iterations")
    first = iterations[0] if isinstance(iterations, list) and iterations else None
    scenarios = first.get("scenarios") if isinstance(first, dict) else None
    if not isinstance(scenarios, list):
        _issue(issues, "error", "diagnostic:manifest", "first iteration scenarios are absent")
        return [], None
    task_ids = [item.get("task_id") for item in scenarios if isinstance(item, dict)]
    indices = [item.get("scenario_idx") for item in scenarios if isinstance(item, dict)]
    if (
        first.get("iteration") != 1
        or first.get("observed_rollouts") != 144
        or indices != list(range(24))
        or len(task_ids) != 24
        or len(set(task_ids)) != 24
        or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
    ):
        _issue(issues, "error", "diagnostic:manifest", "first iteration is not exact 24x6")
        return [], None
    return task_ids, _task_ids_sha256(task_ids)


def _diagnostic_row(branch: str, state: str, iteration: int, source: Path) -> dict[str, Any]:
    fields = {
        "zero_return_std_group_rate": None,
        "mean_unique_return_count_per_group": None,
        "mean_return_range_per_group": None,
        "mean_unique_api_sequence_count_per_group": None,
        "mean_unique_business_api_sequence_count_per_group": None,
        "same_business_api_sequence_ratio": None,
        "effective_advantage_rollout_ratio": None,
        "effective_advantage_token_ratio": None,
        "all_success_groups": None,
        "all_failure_groups": None,
        "partial_same_groups": None,
        "non_zero_variance_groups": None,
        "strict_success_rate": None,
        "average_partial_pass": None,
        "execution_failed_count": None,
        "no_code_found_count": None,
        "http_401_count": None,
        "http_422_count": None,
        "name_error_count": None,
        "invalid_api_hits": None,
        "consecutive_repeated_failed_action_count": None,
        "average_execution_errors_before_strict_success": None,
        "api_doc_calls_per_rollout": None,
        "api_description_calls_per_rollout": None,
        "average_turn_count": None,
        "context_truncation_ratio": None,
        "failed_api_calls": None,
        "recovered_api_calls": None,
        "error_recovery_success_rate": None,
        "error_rollout_count": None,
        "error_rollout_strict_success_count": None,
        "error_rollout_recovery_success_rate": None,
        "execution_failed_per_rollout": None,
        "no_code_found_per_rollout": None,
        "cancelled_count": None,
    }
    return {
        "record_type": "diagnostic",
        "branch": branch,
        "state": state,
        "iteration": iteration,
        "available": False,
        "comparability_status": "missing",
        "dataset_name": DIAGNOSTIC_SPLIT,
        "num_groups": None,
        "num_rollouts": None,
        "task_ids_sha256": None,
        "source": str(source),
        **fields,
    }


def _validate_diagnostic_config(
    resolved: Any,
    *,
    state: str,
    iteration: int,
    scope: str,
    expected_manifest: Path,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(resolved, dict):
        _issue(issues, "error", scope, "resolved diagnostic config must be a JSON object")
        return
    checks = (
        ("dataset_name", _nested(resolved, "scenario_sampler", "dataset_name"), DIAGNOSTIC_SPLIT),
        (
            "manifest_path",
            _nested(resolved, "scenario_sampler", "manifest_path"),
            str(expected_manifest),
        ),
        ("start_iteration", _nested(resolved, "scenario_sampler", "start_iteration"), 1),
        ("cycle", _nested(resolved, "scenario_sampler", "cycle"), False),
        ("max_parallel", _nested(resolved, "scenario_sampler", "max_parallel"), 1),
        ("temperature", _nested(resolved, "llm", "temperature"), 1.0),
        ("max_model_len", _nested(resolved, "llm", "vllm_server", "max_model_len"), 20000),
        ("max_new_tokens", _nested(resolved, "llm", "vllm_class", "max_new_tokens"), 1200),
        ("eager_mode", _nested(resolved, "llm", "vllm_server", "eager_mode"), False),
        ("max_gpu_mem_utilization", _nested(resolved, "llm", "max_gpu_mem_utilization"), 0.9),
        (
            "max_interactions",
            _nested(resolved, "scenario_runner", "appworld_config", "env", "max_interactions"),
            40,
        ),
        (
            "sparse_reward",
            _nested(resolved, "scenario_runner", "appworld_config", "env", "sparse_reward"),
            False,
        ),
        ("learning_max_seq_len", _nested(resolved, "rl", "learning_max_seq_len"), 16000),
        (
            "agent_max_seq_len",
            _nested(resolved, "scenario_runner", "appworld_config", "agent", "max_seq_len_tokens"),
            16000,
        ),
        ("num_scenario_runners", resolved.get("num_scenario_runners"), 12),
        ("num_scenarios", resolved.get("num_scenarios"), 24),
        ("rollouts_per_scenario", resolved.get("rollouts_per_scenario"), 6),
        (
            "rollout_seeds",
            resolved.get("rollout_seeds"),
            list(DEFAULT_ROLLOUT_SEEDS),
        ),
        ("diagnostic_iteration", resolved.get("diagnostic_iteration"), iteration),
        ("lora_rank", _nested(resolved, "llm", "lora_rank"), 32 if state == "r0" else 16),
    )
    for name, actual, expected in checks:
        if name == "manifest_path" and actual is not None:
            actual = str(_resolve(actual))
            expected = str(_resolve(expected))
        if actual != expected:
            _issue(issues, "error", scope, f"{name}={actual!r}; expected {expected!r}")


def _validate_policy_identity(
    *, state: str, iteration: int, run: dict[str, Any], scope: str, issues: list[dict[str, Any]]
) -> None:
    base_model = str(run.get("base_model_path") or "")
    adapter = run.get("adapter_path")
    if state.startswith("r1-"):
        if "Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16" not in base_model:
            _issue(issues, "error", scope, "R1 diagnostic did not use the merged SFT base")
        if not str(adapter or "").endswith(f"checkpoint-{iteration}"):
            _issue(issues, "error", scope, "R1 diagnostic adapter checkpoint is incorrect")
    elif state.startswith("r2-"):
        if not base_model.endswith("Qwen2.5-7B-Instruct"):
            _issue(issues, "error", scope, "R2 diagnostic did not use the original base")
        if not str(adapter or "").endswith(f"checkpoint-{iteration}"):
            _issue(issues, "error", scope, "R2 diagnostic adapter checkpoint is incorrect")
    elif state == "r0":
        if not base_model.endswith("Qwen2.5-7B-Instruct"):
            _issue(issues, "error", scope, "R0 diagnostic did not use the original base")
        if "d12_100_1epoch" not in str(adapter or ""):
            _issue(issues, "error", scope, "R0 diagnostic did not use the rank-32 SFT adapter")
    elif state == "base":
        if not base_model.endswith("Qwen2.5-7B-Instruct"):
            _issue(issues, "error", scope, "Base diagnostic used a non-original base")
        if adapter is not None:
            _issue(issues, "error", scope, "Base diagnostic unexpectedly loaded an adapter")


def _validate_diagnostic_provenance(
    output_dir: Path, *, scope: str, issues: list[dict[str, Any]]
) -> None:
    launch = _safe_load(
        output_dir / "orchestrator_job.json", issues=issues, scope=scope, missing_ok=False
    )
    completion = _safe_load(
        output_dir / "orchestrator_complete.json", issues=issues, scope=scope, missing_ok=False
    )
    if not isinstance(launch, dict):
        return
    command = launch.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        _issue(issues, "error", scope, "orchestrator command provenance is invalid")
        return
    command_hash = _sha256_json(command)
    if launch.get("command_sha256") != command_hash:
        _issue(issues, "error", scope, "orchestrator launch command hash does not match")
    if not isinstance(completion, dict) or completion.get("command_sha256") != command_hash:
        _issue(issues, "error", scope, "completion marker does not match the launch command")


def _load_diagnostic_rows(
    *,
    evaluation_root: Path,
    scenario_manifest: Path,
    expected_task_ids: Sequence[str],
    expected_task_hash: str | None,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, state, iteration in DIAGNOSTIC_SPECS:
        output_dir = evaluation_root / "fixed_train_diagnostic" / state
        diagnostics_path = output_dir / "rollout_diagnostics.json"
        scope = f"diagnostic:{state}"
        row = _diagnostic_row(branch, state, iteration, diagnostics_path)
        payload = _safe_load(diagnostics_path, issues=issues, scope=scope)
        if isinstance(payload, dict):
            row["available"] = True
            row["num_groups"] = _number(payload.get("num_groups"))
            row["num_rollouts"] = _number(payload.get("num_rollouts"))
            for field in (
                "zero_return_std_group_rate",
                "mean_unique_return_count_per_group",
                "mean_return_range_per_group",
                "mean_unique_api_sequence_count_per_group",
                "mean_unique_business_api_sequence_count_per_group",
                "same_business_api_sequence_ratio",
                "effective_advantage_rollout_ratio",
                "effective_advantage_token_ratio",
            ):
                row[field] = _number(payload.get(field))
            classes = payload.get("zero_return_group_class_counts") or {}
            for source_name, target_name in (
                ("all_success", "all_success_groups"),
                ("all_failure", "all_failure_groups"),
                ("partial_same", "partial_same_groups"),
                ("non_zero_variance", "non_zero_variance_groups"),
            ):
                row[target_name] = _number(classes.get(source_name))
            behavior = payload.get("behavior") or {}
            for field in (
                "strict_success_rate",
                "average_partial_pass",
                "execution_failed_count",
                "no_code_found_count",
                "http_401_count",
                "http_422_count",
                "name_error_count",
                "invalid_api_hits",
                "consecutive_repeated_failed_action_count",
                "average_execution_errors_before_strict_success",
                "api_doc_calls_per_rollout",
                "api_description_calls_per_rollout",
                "average_turn_count",
                "context_truncation_ratio",
                "failed_api_calls",
                "recovered_api_calls",
                "error_recovery_success_rate",
                "error_rollout_count",
                "error_rollout_strict_success_count",
                "error_rollout_recovery_success_rate",
                "execution_failed_per_rollout",
                "no_code_found_per_rollout",
                "cancelled_count",
            ):
                row[field] = _number(behavior.get(field))

            run = payload.get("run") or {}
            expected_run = {
                "dataset_name": DIAGNOSTIC_SPLIT,
                "temperature": 1.0,
                "max_interactions": 40,
                "num_scenarios": 24,
                "rollouts_per_scenario": 6,
                "rollout_seeds": list(DEFAULT_ROLLOUT_SEEDS),
                "num_scenario_runners": 12,
                "trajectory_count": 144,
            }
            for name, expected in expected_run.items():
                if run.get(name) != expected:
                    _issue(
                        issues,
                        "error",
                        scope,
                        f"diagnostic run {name}={run.get(name)!r}; expected {expected!r}",
                    )
            _validate_policy_identity(
                state=state, iteration=iteration, run=run, scope=scope, issues=issues
            )
            if row["num_groups"] != 24 or row["num_rollouts"] != 144:
                _issue(
                    issues,
                    "error",
                    scope,
                    "diagnostic summary is not exactly 24 groups/144 rollouts",
                )
            required_metrics = (
                "zero_return_std_group_rate",
                "mean_unique_return_count_per_group",
                "effective_advantage_rollout_ratio",
                "effective_advantage_token_ratio",
            )
            if any(row[field] is None for field in required_metrics):
                _issue(issues, "error", scope, "required diagnostic signal metrics are incomplete")
            groups = payload.get("groups")
            if not isinstance(groups, list):
                _issue(issues, "error", scope, "diagnostic groups are absent")
            else:
                rollout_rows = [
                    rollout
                    for group in groups
                    if isinstance(group, dict)
                    for rollout in (group.get("rollouts") or [])
                    if isinstance(rollout, dict)
                ]
                error_rollouts = [
                    rollout
                    for rollout in rollout_rows
                    if (_number(rollout.get("execution_failed_count")) or 0) > 0
                ]
                error_rollout_successes = sum(
                    bool(rollout.get("strict_success")) for rollout in error_rollouts
                )
                if row["error_rollout_count"] is None:
                    row["error_rollout_count"] = len(error_rollouts)
                if row["error_rollout_strict_success_count"] is None:
                    row["error_rollout_strict_success_count"] = error_rollout_successes
                if row["error_rollout_recovery_success_rate"] is None:
                    row["error_rollout_recovery_success_rate"] = (
                        error_rollout_successes / len(error_rollouts) if error_rollouts else None
                    )
                task_ids = [group.get("task_id") for group in groups if isinstance(group, dict)]
                row["task_ids_sha256"] = (
                    _task_ids_sha256(task_ids)
                    if all(isinstance(task_id, str) for task_id in task_ids)
                    else None
                )
                if list(task_ids) != list(expected_task_ids):
                    _issue(issues, "error", scope, "diagnostic task order differs from manifest")
                if expected_task_hash is not None and row["task_ids_sha256"] != expected_task_hash:
                    _issue(issues, "error", scope, "diagnostic task hash differs from manifest")
                for expected_idx, group in enumerate(groups):
                    if not isinstance(group, dict):
                        _issue(issues, "error", scope, "diagnostic group is not an object")
                        break
                    rollouts = group.get("rollouts")
                    seed_map = {
                        rollout.get("rollout_idx"): rollout.get("generation_seed")
                        for rollout in rollouts or []
                        if isinstance(rollout, dict)
                    }
                    if (
                        group.get("scenario_idx") != expected_idx
                        or group.get("rollout_count") != 6
                        or seed_map != dict(enumerate(DEFAULT_ROLLOUT_SEEDS))
                    ):
                        _issue(
                            issues,
                            "error",
                            scope,
                            f"scenario {expected_idx} indices or generation seeds differ",
                        )
                        break
            resolved = _safe_load(
                output_dir / "resolved_config.json",
                issues=issues,
                scope=scope,
                missing_ok=False,
            )
            _validate_diagnostic_config(
                resolved,
                state=state,
                iteration=iteration,
                scope=scope,
                expected_manifest=scenario_manifest,
                issues=issues,
            )
            _validate_diagnostic_provenance(output_dir, scope=scope, issues=issues)
        elif payload is not None:
            _issue(issues, "error", scope, "diagnostics JSON must be an object")
        row["comparability_status"] = _row_status(issues, scope, available=row["available"])
        rows.append(row)
    return rows


def _training_row(iteration: int, source: Path) -> dict[str, Any]:
    return {
        "record_type": "training",
        "branch": "r1",
        "state": f"r1-{iteration}",
        "iteration": iteration,
        "available": False,
        "status": "missing",
        "source": str(source),
        "actual_optimizer_steps": None,
        "attempted_gradient_steps": None,
        "per_token_kl": None,
        "ppo_clip_fraction": None,
        "grad_norm_mean": None,
        "grad_norm_max": None,
        "parameter_update_l2_norm_mean": None,
        "parameter_update_l2_norm_max": None,
        "sampling_entropy_mean_nats": None,
        "learning_rate_before_scheduler_step": None,
        "learning_rate_after_scheduler_step": None,
        "filtered_rollout_fraction": None,
        "below_threshold_rollouts": None,
        "below_threshold_output_tokens": None,
        "actually_filtered_output_tokens": None,
        "invalid_loss_steps": None,
        "high_kl_events_iteration": None,
        "high_kl_events_cumulative": None,
        "failure_event": None,
        "failure_type": None,
    }


def _load_training_rows(r1_run_dir: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for iteration in range(1, 16):
        path = r1_run_dir / "training_metrics" / f"iteration-{iteration:06d}.json"
        scope = f"training:{iteration}"
        row = _training_row(iteration, path)
        payload = _safe_load(path, issues=issues, scope=scope)
        if isinstance(payload, dict):
            row["available"] = True
            row["status"] = str(payload.get("status") or "unknown")
            if payload.get("schema_version") != "loop-iteration-training-metrics-v1":
                _issue(issues, "error", scope, "unexpected training metric schema")
            if payload.get("iteration") != iteration:
                _issue(issues, "error", scope, "training metric iteration does not match filename")
            row.update(
                actual_optimizer_steps=_number(payload.get("actual_optimizer_steps")),
                attempted_gradient_steps=_number(payload.get("attempted_gradient_steps")),
                per_token_kl=_number(_nested(payload, "per_token_kl", "mean")),
                ppo_clip_fraction=_number(_nested(payload, "ppo_clip_fraction", "fraction")),
                grad_norm_mean=_number(_nested(payload, "grad_norm_before_clipping", "mean")),
                grad_norm_max=_number(_nested(payload, "grad_norm_before_clipping", "max")),
                parameter_update_l2_norm_mean=_number(
                    _nested(payload, "parameter_update_l2_norm", "mean")
                ),
                parameter_update_l2_norm_max=_number(
                    _nested(payload, "parameter_update_l2_norm", "max")
                ),
                sampling_entropy_mean_nats=_number(
                    _nested(payload, "sampling_entropy", "mean_nats")
                ),
                learning_rate_before_scheduler_step=_number(
                    _nested(payload, "learning_rate", "before_iteration_scheduler_step")
                ),
                learning_rate_after_scheduler_step=_number(
                    _nested(payload, "learning_rate", "after_iteration_scheduler_step")
                ),
                filtered_rollout_fraction=_number(
                    _nested(
                        payload,
                        "abs_adv_threshold_filter",
                        "actually_filtered_rollout_fraction",
                    )
                ),
                below_threshold_rollouts=_number(
                    _nested(payload, "abs_adv_threshold_filter", "below_threshold_rollouts")
                ),
                below_threshold_output_tokens=_number(
                    _nested(payload, "abs_adv_threshold_filter", "below_threshold_output_tokens")
                ),
                actually_filtered_output_tokens=_number(
                    _nested(
                        payload,
                        "abs_adv_threshold_filter",
                        "actually_filtered_output_tokens",
                    )
                ),
                invalid_loss_steps=_number(payload.get("invalid_loss_steps")),
                high_kl_events_iteration=_number(_nested(payload, "high_kl_events", "iteration")),
                high_kl_events_cumulative=_number(_nested(payload, "high_kl_events", "cumulative")),
                failure_event=_nested(payload, "failure", "event"),
                failure_type=_nested(payload, "failure", "exception_type"),
            )
            if row["status"] != "completed":
                severity = "error" if row["status"] == "failed" else "warning"
                _issue(issues, severity, scope, f"training iteration status is {row['status']!r}")
        elif payload is not None:
            _issue(issues, "error", scope, "training metric JSON must be an object")
        rows.append(row)
    return rows


def _build_comparisons(capability_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["branch"], row["iteration"]): row for row in capability_rows}
    pointwise = []
    for iteration in ITERATIONS:
        r1 = by_key[("r1", iteration)]
        r2 = by_key[("r2", iteration)]
        pointwise.append(
            {
                "record_type": "pointwise_delta",
                "iteration": iteration,
                "r1_state": r1["state"],
                "r2_state": r2["state"],
                **{
                    f"delta_r1_minus_r2_{metric}": _delta(r1.get(metric), r2.get(metric))
                    for metric in CURVE_METRICS
                },
            }
        )

    auc_rows = []
    branch_auc: dict[str, dict[str, float | None]] = {}
    for branch in ("r1", "r2"):
        rows = [row for row in capability_rows if row["branch"] == branch]
        branch_auc[branch] = {}
        for metric in CURVE_METRICS:
            value, missing = normalized_trapezoidal_auc(rows, metric)
            branch_auc[branch][metric] = value
            auc_rows.append(
                {
                    "record_type": "normalized_auc",
                    "branch": branch,
                    "metric": metric,
                    "value": value,
                    "missing_iterations": missing,
                    "normalization_interval": 15,
                }
            )
    auc_delta = {
        metric: _delta(branch_auc["r1"][metric], branch_auc["r2"][metric])
        for metric in CURVE_METRICS
    }
    endpoint = {
        "r1_15_minus_r0": {
            metric: _delta(by_key[("r1", 15)].get(metric), by_key[("r1", 0)].get(metric))
            for metric in CURVE_METRICS
        },
        "r2_15_minus_base": {
            metric: _delta(by_key[("r2", 15)].get(metric), by_key[("r2", 0)].get(metric))
            for metric in CURVE_METRICS
        },
        "r1_15_minus_r2_15": {
            metric: _delta(by_key[("r1", 15)].get(metric), by_key[("r2", 15)].get(metric))
            for metric in CURVE_METRICS
        },
    }
    return {
        "pointwise_r1_minus_r2": pointwise,
        "normalized_auc": auc_rows,
        "normalized_auc_r1_minus_r2": auc_delta,
        "endpoint_deltas": endpoint,
    }


def _training_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    missing = [row for row in rows if not row["available"]]
    optimizer_steps = [row["actual_optimizer_steps"] for row in completed]
    high_kl = [row["high_kl_events_iteration"] for row in completed]
    return {
        "expected_iterations": 15,
        "completed_iterations": [row["iteration"] for row in completed],
        "failed_iterations": [row["iteration"] for row in failed],
        "missing_iterations": [row["iteration"] for row in missing],
        "total_optimizer_steps": (
            sum(int(value) for value in optimizer_steps)
            if completed and all(value is not None for value in optimizer_steps)
            else None
        ),
        "total_high_kl_events": (
            sum(int(value) for value in high_kl)
            if completed and all(value is not None for value in high_kl)
            else None
        ),
    }


def _comparability_summary(
    issues: Sequence[dict[str, Any]],
    capability_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        severity: sum(item["severity"] == severity for item in issues)
        for severity in ("error", "warning", "missing")
    }
    if counts["error"]:
        status = "fail"
    elif counts["missing"]:
        status = "incomplete"
    elif counts["warning"]:
        status = "warning"
    else:
        status = "pass"
    return {
        "status": status,
        "issue_counts": counts,
        "available_capability_states": sum(row["available"] for row in capability_rows),
        "expected_capability_states": len(capability_rows),
        "available_diagnostic_states": sum(row["available"] for row in diagnostic_rows),
        "expected_diagnostic_states": len(diagnostic_rows),
        "issues": list(issues),
    }


def build_report(
    *,
    repo_root: Path,
    r0_metrics: Path,
    evaluation_root: Path,
    r1_run_dir: Path,
    scenario_manifest: Path,
    dev_task_list: Path,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    dev_task_ids: list[str] = []
    if dev_task_list.is_file():
        dev_task_ids = _load_task_ids(dev_task_list)
        if not dev_task_ids or len(dev_task_ids) != len(set(dev_task_ids)):
            _issue(issues, "error", "dev:task_list", "dev task list is empty or duplicated")
    else:
        _issue(issues, "error", "dev:task_list", f"task list is absent: {dev_task_list}")
    expected_dev_count = len(dev_task_ids) if dev_task_ids else None
    capability_rows = _load_capability_rows(
        r0_metrics=r0_metrics,
        evaluation_root=evaluation_root,
        expected_dev_count=expected_dev_count,
        issues=issues,
    )
    fixed_task_ids, fixed_task_hash = _load_fixed_task_contract(scenario_manifest, issues)
    diagnostic_rows = _load_diagnostic_rows(
        evaluation_root=evaluation_root,
        scenario_manifest=scenario_manifest,
        expected_task_ids=fixed_task_ids,
        expected_task_hash=fixed_task_hash,
        issues=issues,
    )
    training_rows = _load_training_rows(r1_run_dir, issues)
    comparisons = _build_comparisons(capability_rows)
    comparability = _comparability_summary(issues, capability_rows, diagnostic_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "inputs": {
            "repo_root": str(repo_root),
            "r0_metrics": str(r0_metrics),
            "evaluation_root": str(evaluation_root),
            "r1_run_dir": str(r1_run_dir),
            "scenario_manifest": str(scenario_manifest),
            "dev_task_list": str(dev_task_list),
        },
        "contracts": {
            "dev": {
                "split": DEV_SPLIT,
                "task_count": expected_dev_count,
                "task_ids_sha256": _task_ids_sha256(dev_task_ids) if dev_task_ids else None,
                "iterations": list(ITERATIONS),
                "paired_eval_seed": 20260718,
                "paired_request_seed": DEFAULT_ROLLOUT_SEEDS[0],
                "r0_historical_unpaired_seed": True,
            },
            "diagnostic": {
                "split": DIAGNOSTIC_SPLIT,
                "task_count": len(fixed_task_ids) if fixed_task_ids else None,
                "task_ids_sha256": fixed_task_hash,
                "rollouts_per_scenario": 6,
                "rollout_seeds": list(DEFAULT_ROLLOUT_SEEDS),
            },
        },
        "comparability": comparability,
        "capability": capability_rows,
        "fixed_diagnostics": diagnostic_rows,
        "r1_training_metrics": training_rows,
        "r1_training_summary": _training_summary(training_rows),
        "comparisons": comparisons,
    }


def _csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(report["capability"])
    rows.extend(report["fixed_diagnostics"])
    rows.extend(report["r1_training_metrics"])
    rows.extend(report["comparisons"]["pointwise_r1_minus_r2"])
    rows.extend(report["comparisons"]["normalized_auc"])
    for name, deltas in report["comparisons"]["endpoint_deltas"].items():
        rows.append({"record_type": "endpoint_delta", "name": name, **deltas})
    rows.append(
        {
            "record_type": "normalized_auc_delta",
            "name": "r1_minus_r2",
            **report["comparisons"]["normalized_auc_r1_minus_r2"],
        }
    )
    rows.append({"record_type": "training_summary", **report["r1_training_summary"]})
    rows.append(
        {
            "record_type": "comparability_summary",
            "status": report["comparability"]["status"],
            **report["comparability"]["issue_counts"],
        }
    )
    rows.extend(
        {
            "record_type": "comparability_issue",
            **issue,
        }
        for issue in report["comparability"]["issues"]
    )
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return value


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = _csv_rows(report)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def _md(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "—"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    comparability = report["comparability"]
    lines = [
        "# SFT→LOOP 15-iteration comparison",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"Comparability status: **{comparability['status']}**. "
        f"Capability {comparability['available_capability_states']}/"
        f"{comparability['expected_capability_states']}; diagnostics "
        f"{comparability['available_diagnostic_states']}/"
        f"{comparability['expected_diagnostic_states']}.",
        "",
        "Missing values are shown as `—` and remain JSON `null` / CSV `NA`.",
        "",
        "## dev_small64 capability",
        "",
        "| Branch | State | Iter | Status | TGC | SGC | Partial | D1 TGC | D2 TGC | D3 TGC | Exec failed | Invalid API | Docs/rollout | Turns | Truncation |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["capability"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    str(row["iteration"]),
                    row["comparability_status"],
                    _md(row["TGC"], 2),
                    _md(row["SGC"], 2),
                    _md(row["average_partial_pass_rate"]),
                    _md(row["TGC_1"], 2),
                    _md(row["TGC_2"], 2),
                    _md(row["TGC_3"], 2),
                    _md(row["execution_failed_count"], 0),
                    _md(row["invalid_api_hits"], 0),
                    _md(row["api_doc_calls_per_rollout"]),
                    _md(row["num_turns_avg"]),
                    _md(row["context_truncation_ratio"]),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Difficulty breakdown",
            "",
            "| Branch | State | Iter | D1 TGC | D1 SGC | D2 TGC | D2 SGC | D3 TGC | D3 SGC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["capability"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    str(row["iteration"]),
                    _md(row["TGC_1"], 2),
                    _md(row["SGC_1"], 2),
                    _md(row["TGC_2"], 2),
                    _md(row["SGC_2"], 2),
                    _md(row["TGC_3"], 2),
                    _md(row["SGC_3"], 2),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Behavior and error breakdown",
            "",
            "### dev error counts",
            "",
            "| Branch | State | Iter | Exec failed | No-code | 401 | 422 | NameError | Invalid API | Repeated failed action | Errors before strict success |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["capability"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    str(row["iteration"]),
                    _md(row["execution_failed_count"], 0),
                    _md(row["no_code_found_count"], 0),
                    _md(row["http_401_count"], 0),
                    _md(row["http_422_count"], 0),
                    _md(row["name_error_count"], 0),
                    _md(row["invalid_api_hits"], 0),
                    _md(row["consecutive_repeated_failed_action_count"], 0),
                    _md(row["average_execution_errors_before_strict_success"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### dev recovery, exploration, and context",
            "",
            "| Branch | State | Failed API | Recovered API | API recovery | Error rollouts | Error-rollout successes | Error-rollout recovery | Give-up rate | Docs/rollout | Descriptions/rollout | Turns | Truncation |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["capability"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    _md(row["failed_api_calls"], 0),
                    _md(row["recovered_api_calls"], 0),
                    _md(row["error_recovery_success_rate"]),
                    _md(row["error_rollout_count"], 0),
                    _md(row["error_rollout_strict_success_count"], 0),
                    _md(row["error_rollout_recovery_success_rate"]),
                    _md(row["failed_api_call_give_up_rate"]),
                    _md(row["api_doc_calls_per_rollout"]),
                    _md(row["api_description_calls_per_rollout"]),
                    _md(row["num_turns_avg"]),
                    _md(row["context_truncation_ratio"]),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Normalized 0–15 trapezoidal AUC",
            "",
            "| Branch | Metric | AUC | Missing iterations |",
            "|---|---|---:|---|",
        )
    )
    for row in report["comparisons"]["normalized_auc"]:
        lines.append(
            f"| {row['branch'].upper()} | {row['metric']} | {_md(row['value'])} | "
            f"{_md(row['missing_iterations'])} |"
        )
    lines.extend(("", "AUC delta R1−R2:", ""))
    for metric, value in report["comparisons"]["normalized_auc_r1_minus_r2"].items():
        lines.append(f"- `{metric}`: {_md(value)}")

    lines.extend(
        (
            "",
            "## Capability deltas",
            "",
            "| Iteration | Δ TGC R1−R2 | Δ SGC R1−R2 | Δ partial R1−R2 |",
            "|---:|---:|---:|---:|",
        )
    )
    for row in report["comparisons"]["pointwise_r1_minus_r2"]:
        lines.append(
            f"| {row['iteration']} | {_md(row['delta_r1_minus_r2_TGC'])} | "
            f"{_md(row['delta_r1_minus_r2_SGC'])} | "
            f"{_md(row['delta_r1_minus_r2_average_partial_pass_rate'])} |"
        )
    lines.extend(("", "Endpoint deltas:", ""))
    for name, values in report["comparisons"]["endpoint_deltas"].items():
        lines.append(
            f"- `{name}`: TGC {_md(values['TGC'])}; SGC {_md(values['SGC'])}; "
            f"partial {_md(values['average_partial_pass_rate'])}"
        )

    lines.extend(
        (
            "",
            "## Fixed 24×6 train diagnostic",
            "",
            "| Branch | State | Iter | Status | Zero-std groups | All-failure | Unique returns | Effective rollout | Effective tokens | Unique API seq | Strict success | Docs/rollout | Errors |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["fixed_diagnostics"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    str(row["iteration"]),
                    row["comparability_status"],
                    _md(row["zero_return_std_group_rate"]),
                    _md(row["all_failure_groups"], 0),
                    _md(row["mean_unique_return_count_per_group"]),
                    _md(row["effective_advantage_rollout_ratio"]),
                    _md(row["effective_advantage_token_ratio"]),
                    _md(row["mean_unique_api_sequence_count_per_group"]),
                    _md(row["strict_success_rate"]),
                    _md(row["api_doc_calls_per_rollout"]),
                    _md(row["execution_failed_count"], 0),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "### fixed diagnostic error counts",
            "",
            "| Branch | State | Exec failed | No-code | 401 | 422 | NameError | Invalid API | Repeated failed action | Errors before strict success |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["fixed_diagnostics"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    _md(row["execution_failed_count"], 0),
                    _md(row["no_code_found_count"], 0),
                    _md(row["http_401_count"], 0),
                    _md(row["http_422_count"], 0),
                    _md(row["name_error_count"], 0),
                    _md(row["invalid_api_hits"], 0),
                    _md(row["consecutive_repeated_failed_action_count"], 0),
                    _md(row["average_execution_errors_before_strict_success"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### fixed diagnostic recovery and context",
            "",
            "| Branch | State | Failed API | Recovered API | API recovery | Error rollouts | Error-rollout successes | Error-rollout recovery | Docs/rollout | Descriptions/rollout | Turns | Truncation |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["fixed_diagnostics"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["branch"].upper(),
                    row["state"],
                    _md(row["failed_api_calls"], 0),
                    _md(row["recovered_api_calls"], 0),
                    _md(row["error_recovery_success_rate"]),
                    _md(row["error_rollout_count"], 0),
                    _md(row["error_rollout_strict_success_count"], 0),
                    _md(row["error_rollout_recovery_success_rate"]),
                    _md(row["api_doc_calls_per_rollout"]),
                    _md(row["api_description_calls_per_rollout"]),
                    _md(row["average_turn_count"]),
                    _md(row["context_truncation_ratio"]),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## R1 training stability",
            "",
            "Completed iterations: "
            f"{_md(report['r1_training_summary']['completed_iterations'])}; "
            "missing: "
            f"{_md(report['r1_training_summary']['missing_iterations'])}; "
            "total optimizer steps: "
            f"{_md(report['r1_training_summary']['total_optimizer_steps'], 0)}.",
            "",
            "| Iter | Status | Optimizer steps | KL/token | Clip fraction | Grad norm | Update norm | Entropy | Filtered rollout | High-KL | LR after |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["r1_training_metrics"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["iteration"]),
                    row["status"],
                    _md(row["actual_optimizer_steps"], 0),
                    _md(row["per_token_kl"]),
                    _md(row["ppo_clip_fraction"]),
                    _md(row["grad_norm_mean"]),
                    _md(row["parameter_update_l2_norm_mean"]),
                    _md(row["sampling_entropy_mean_nats"]),
                    _md(row["filtered_rollout_fraction"]),
                    _md(row["high_kl_events_iteration"], 0),
                    _md(row["learning_rate_after_scheduler_step"], 8),
                )
            )
            + " |"
        )

    lines.extend(("", "## Comparability findings", ""))
    if not comparability["issues"]:
        lines.append("- No comparability issues detected.")
    else:
        for issue in comparability["issues"]:
            lines.append(f"- **{issue['severity']}** `{issue['scope']}`: {issue['message']}")
    lines.append("")
    return "\n".join(lines)


def write_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "json": output_dir / "sft_loop15_report.json",
        "csv": output_dir / "sft_loop15_report.csv",
        "markdown": output_dir / "sft_loop15_report.md",
    }
    _write_json(paths["json"], report)
    write_csv(paths["csv"], report)
    _write_text_atomic(paths["markdown"], render_markdown(report))
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--r0-metrics",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "appworld_sft"
            / "compact_lora_20260713"
            / "evaluation"
            / "dev_small64"
            / "d12_100_x1"
            / "behavior.json"
        ),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=repo_root / "artifacts" / "sft_loop15" / "unified_evaluations",
    )
    parser.add_argument(
        "--r1-run-dir",
        type=Path,
        default=(
            repo_root
            / "experiments"
            / "qwen25_7b_d12_100x1_loop15_lora16"
            / "r1_loop15_seed20260718"
        ),
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=repo_root / "artifacts" / "sft_loop15" / "r2_first15_scenario_manifest.json",
    )
    parser.add_argument(
        "--dev-task-list",
        type=Path,
        default=repo_root / "data" / "appworld_splits" / "dev_small64.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "sft_loop15" / "report",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail after writing if any expected artifact is still missing",
    )
    parser.add_argument(
        "--strict-comparability",
        action="store_true",
        help="fail after writing if a task/config mismatch is detected",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = _resolve(args.repo_root)
    report = build_report(
        repo_root=repo_root,
        r0_metrics=_resolve(args.r0_metrics),
        evaluation_root=_resolve(args.evaluation_root),
        r1_run_dir=_resolve(args.r1_run_dir),
        scenario_manifest=_resolve(args.scenario_manifest),
        dev_task_list=_resolve(args.dev_task_list),
    )
    paths = write_report(_resolve(args.output_dir), report)
    print("report artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"comparability: {report['comparability']['status']}")
    if args.strict_comparability and report["comparability"]["issue_counts"]["error"]:
        return 2
    if args.require_complete and report["comparability"]["issue_counts"]["missing"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
