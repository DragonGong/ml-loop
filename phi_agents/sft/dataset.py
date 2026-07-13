from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phi_agents.appworld.interface import load_task_ids
from phi_agents.sft.teacher import DATA_VERSION

SFT_DATA_VERSION = f"{DATA_VERSION}-supervision-v2"

API_CALL_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
COMPLETE_TASK_RE = re.compile(r"\bapis\.supervisor\.complete_task\s*\(")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"DEEPSEEK_API_KEY\s*[:=]"),
)
HIDDEN_TEXT_PATTERNS = (
    re.compile(r"hidden\s+(?:test|requirement|evaluator)", re.IGNORECASE),
    re.compile(r"target\s+database\s+state", re.IGNORECASE),
)
INFRASTRUCTURE_ERROR_TYPES = {
    "appworld_connection_error",
    "appworld_execution_error",
    "deepseek_request_error",
    "service_error",
    "timeout",
    "cancelled",
}
INFRASTRUCTURE_ERROR_FLAGS = (
    "infrastructure_error",
    "service_error",
    "connection_error",
    "timed_out",
    "timeout",
    "cancelled",
)


@dataclass(frozen=True)
class DatasetBuildConfig:
    input_roots: tuple[Path, ...]
    output_dir: Path
    mode: str = "difficulty_1_2"
    validation_fraction: float = 0.15
    near_duplicate_threshold: float = 0.92
    data_version: str = SFT_DATA_VERSION
    mask_known_bad_actions: bool = True

    @property
    def allowed_splits(self) -> tuple[str, ...]:
        if self.mode == "difficulty_1_2":
            return ("train_difficulty_1_2",)
        if self.mode == "difficulty_3":
            return ("train_difficulty_3",)
        if self.mode == "difficulty_1_2_3":
            return ("train_difficulty_1_2", "train_difficulty_3")
        raise ValueError(f"Unsupported dataset mode: {self.mode}")


@dataclass
class QualityDecision:
    accepted: bool
    reasons: list[str]
    success_type: str | None
    action_fingerprint: str
    near_duplicate_of: str | None = None


def scenario_id(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def _normalized_action(action: str) -> str:
    return " ".join(action.split())


def action_fingerprint(trajectory: dict[str, Any]) -> str:
    actions = [
        _normalized_action(str(step.get("action_code") or ""))
        for step in trajectory.get("steps") or []
    ]
    return hashlib.sha256("\n---\n".join(actions).encode()).hexdigest()


def _action_shingles(trajectory: dict[str, Any], size: int = 3) -> set[str]:
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_.]*|\d+|[^\s]",
        "\n".join(str(step.get("action_code") or "") for step in trajectory.get("steps") or []),
    )
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[idx : idx + size]) for idx in range(len(tokens) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _has_repetition(steps: list[dict[str, Any]]) -> bool:
    actions = [_normalized_action(str(step.get("action_code") or "")) for step in steps]
    counts = Counter(action for action in actions if action)
    if any(count >= 4 for count in counts.values()):
        return True
    for idx in range(len(actions) - 2):
        if actions[idx] and len(set(actions[idx : idx + 3])) == 1:
            return True
    doc_only_streak = 0
    seen_doc_actions: set[str] = set()
    for action in actions:
        calls = API_CALL_RE.findall(action)
        is_doc_only = bool(calls) and all(app_name == "api_docs" for app_name, _ in calls)
        if is_doc_only:
            doc_only_streak += 1
            if action in seen_doc_actions or doc_only_streak > 6:
                return True
            seen_doc_actions.add(action)
        else:
            doc_only_streak = 0
    return False


def _code_is_parseable(action: str) -> bool:
    if not action.strip():
        return False
    try:
        ast.parse(action)
    except SyntaxError:
        return False
    return True


def stable_trajectory_id(trajectory: dict[str, Any]) -> str:
    """Return a path-independent, reproducible identifier for a teacher trajectory."""
    for key in ("trajectory_id", "rollout_id", "run_id", "attempt_id"):
        value = trajectory.get(key)
        if value not in (None, ""):
            return str(value)
    identity = {
        "task_id": trajectory.get("task_id"),
        "experiment_name": trajectory.get("experiment_name"),
        "started_at": trajectory.get("started_at"),
        "finished_at": trajectory.get("finished_at"),
        "steps": [
            {
                "step_index": step.get("step_index"),
                "response_id": step.get("response_id"),
                "model_output": step.get("model_output"),
                "action_code": step.get("action_code"),
                "environment_result": step.get("environment_result"),
            }
            for step in trajectory.get("steps") or []
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"traj-{hashlib.sha256(encoded.encode()).hexdigest()[:20]}"


def assistant_step_mask_reason(step: dict[str, Any]) -> str | None:
    """Classify structured, reliably identifiable assistant-step failures."""
    if bool(step.get("execution_failed")):
        return "execution_failed"
    if bool(step.get("no_code_found")):
        return "no_code"
    error_type = str(step.get("error_type") or "").strip().lower()
    if error_type in INFRASTRUCTURE_ERROR_TYPES or any(
        bool(step.get(flag)) for flag in INFRASTRUCTURE_ERROR_FLAGS
    ):
        return "infrastructure_error"
    model_output = str(step.get("model_output") or "")
    if not model_output.strip():
        return "empty_response"
    action_code = str(step.get("action_code") or "")
    if not action_code.strip():
        return "empty_action"
    if not _code_is_parseable(action_code):
        return "unparseable_code"
    finish_reason = str(step.get("finish_reason") or "").strip().lower()
    if finish_reason and finish_reason not in {"stop", "tool_calls"}:
        return f"incomplete_generation:{finish_reason}"
    return None


def _visible_text(trajectory: dict[str, Any]) -> str:
    initial = trajectory.get("initial_prompt_messages") or []
    pieces = [str(message.get("content") or "") for message in initial]
    for step in trajectory.get("steps") or []:
        pieces.extend(
            [str(step.get("model_output") or ""), str(step.get("environment_result") or "")]
        )
    return "\n".join(pieces)


def _contains_secret(text: str) -> bool:
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return True
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    return bool(api_key and api_key in text)


def assess_trajectory(trajectory: dict[str, Any], allowed_task_ids: set[str]) -> QualityDecision:
    reasons: list[str] = []
    task_id = str(trajectory.get("task_id") or "")
    steps = list(trajectory.get("steps") or [])
    eval_result = trajectory.get("eval_result") or {}
    text = _visible_text(trajectory)
    if task_id not in allowed_task_ids:
        reasons.append("task_not_in_allowed_train_split")
    if trajectory.get("status") != "completed" or trajectory.get("success") is not True:
        reasons.append("not_strict_success")
    if trajectory.get("cancelled") or trajectory.get("error_type") in {
        "max_interactions",
        "appworld_connection_error",
        "appworld_execution_error",
        "deepseek_request_error",
        "cost_limit_reached",
    }:
        reasons.append("incomplete_or_service_failure")
    if not steps or not COMPLETE_TASK_RE.search(str(steps[-1].get("action_code") or "")):
        reasons.append("missing_final_complete_task")
    if steps and not steps[-1].get("task_completed"):
        reasons.append("environment_not_marked_complete")
    if any(step.get("environment_result") is None for step in steps):
        reasons.append("missing_real_observation")
    no_code_count = sum(bool(step.get("no_code_found")) for step in steps)
    if no_code_count > 2:
        reasons.append("excessive_no_code")
    if any(
        step.get("action_code") and not _code_is_parseable(str(step.get("action_code")))
        for step in steps
    ):
        reasons.append("unparseable_code")
    if _has_repetition(steps):
        reasons.append("degenerate_repetition")
    if _contains_secret(text):
        reasons.append("secret_detected")
    if any(pattern.search(text) for pattern in HIDDEN_TEXT_PATTERNS):
        reasons.append("hidden_evaluator_text_detected")
    if any(step.get("reasoning_content") not in (None, "") for step in steps):
        # Reasoning may exist in legacy raw files, but it is deliberately excluded from SFT text.
        pass
    if eval_result.get("success") is not True:
        reasons.append("evaluator_disagrees")

    had_error = any(
        step.get("execution_failed")
        or step.get("no_code_found")
        or step.get("invalid_responses_before_action")
        for step in steps
    )
    was_repaired = bool(
        trajectory.get("repair_from_trajectory")
        or (trajectory.get("teacher_generation") or {}).get("repair_source")
    )
    return QualityDecision(
        accepted=not reasons,
        reasons=reasons,
        success_type="recovered_success" if had_error or was_repaired else "clean_success",
        action_fingerprint=action_fingerprint(trajectory),
    )


def trajectory_to_sample(
    trajectory: dict[str, Any],
    source_path: Path,
    decision: QualityDecision,
    config: DatasetBuildConfig,
) -> dict[str, Any]:
    initial = trajectory.get("initial_prompt_messages") or []
    messages = [
        {"role": message["role"], "content": str(message.get("content") or ""), "loss": False}
        for message in initial
    ]
    steps = trajectory.get("steps") or []
    task_id = str(trajectory["task_id"])
    trajectory_id = stable_trajectory_id(trajectory)
    mask_reason_counts: Counter[str] = Counter()
    for index, step in enumerate(steps):
        step_index = step.get("step_index")
        if step_index in (None, ""):
            step_index = index
        step_id = f"{task_id}:{trajectory_id}:{step_index}"
        mask_reason = assistant_step_mask_reason(step)
        if not config.mask_known_bad_actions and mask_reason != "no_code":
            mask_reason = None
        if mask_reason is not None:
            mask_reason_counts[mask_reason] += 1
        messages.append(
            {
                "role": "assistant",
                "content": str(step.get("model_output") or ""),
                "loss": mask_reason is None,
                "message_type": "appworld_action",
                "step_id": step_id,
                "step_index": step_index,
                "mask_reason": mask_reason,
            }
        )
        next_observation = (
            steps[index + 1].get("observation_sent_to_model") if index + 1 < len(steps) else None
        )
        messages.append(
            {
                "role": "user",
                "content": str(next_observation or step.get("environment_result") or ""),
                "loss": False,
                "message_type": "appworld_observation",
                "step_id": step_id,
                "observation_for_step_id": step_id,
            }
        )

    eval_result = trajectory.get("eval_result") or {}
    num_tests = int(eval_result.get("num_tests") or 0)
    passes = eval_result.get("passes") or []
    usage_rows: list[dict[str, Any]] = []
    for step in steps:
        usage_rows.extend(
            record.get("usage") or {}
            for record in step.get("invalid_responses_before_action") or []
        )
        usage_rows.append(step.get("usage") or {})
    api_calls = sum(len(API_CALL_RE.findall(str(step.get("action_code") or ""))) for step in steps)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage_rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in usage_rows)
    metadata = {
        "task_id": task_id,
        "trajectory_id": trajectory_id,
        "scenario_id": scenario_id(task_id),
        "difficulty": int(eval_result["difficulty"]),
        "success_type": decision.success_type,
        "interaction_count": len(steps),
        "api_call_count": api_calls,
        "execution_failed_count": sum(bool(step.get("execution_failed")) for step in steps),
        "no_code_count": (
            sum(bool(step.get("no_code_found")) for step in steps)
            + sum(len(step.get("invalid_responses_before_action") or []) for step in steps)
        ),
        "partial_pass": len(passes) / num_tests if num_tests else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "teacher_cost_cny": float(trajectory.get("cost_cny") or 0.0),
        "generation_config": {
            "teacher_generation": trajectory.get("teacher_generation"),
            "model_config": trajectory.get("model_config"),
        },
        "data_version": config.data_version,
        "source_trajectory": str(source_path.resolve()),
        "action_fingerprint": decision.action_fingerprint,
        "assistant_step_count": len(steps),
        "supervised_step_count": len(steps) - sum(mask_reason_counts.values()),
        "step_mask_reason_counts": dict(mask_reason_counts),
    }
    return {"messages": messages, "metadata": metadata}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _without_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_reasoning(child)
            for key, child in value.items()
            if key not in {"reasoning", "reasoning_content", "thinking"}
        }
    if isinstance(value, list):
        return [_without_reasoning(child) for child in value]
    return value


def _output_contains_secret(paths: list[Path]) -> bool:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    for path in paths:
        text = path.read_text()
        if (api_key and api_key in text) or any(
            pattern.search(text) for pattern in SECRET_PATTERNS
        ):
            return True
    return False


def _split_scenarios(samples: list[dict[str, Any]], fraction: float) -> tuple[list[Any], list[Any]]:
    scenarios = sorted({sample["metadata"]["scenario_id"] for sample in samples})
    if len(scenarios) < 2:
        return samples, []
    count = max(1, min(len(scenarios) - 1, round(len(scenarios) * fraction)))
    ordered = sorted(scenarios, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    validation = set(ordered[:count])
    return (
        [sample for sample in samples if sample["metadata"]["scenario_id"] not in validation],
        [sample for sample in samples if sample["metadata"]["scenario_id"] in validation],
    )


def build_dataset(config: DatasetBuildConfig) -> dict[str, Any]:
    if not config.mask_known_bad_actions:
        warnings.warn(
            "mask_known_bad_actions=False restores unsafe legacy supervision and may train "
            "execution-failed or malformed actions.",
            stacklevel=2,
        )
    allowed_task_ids = set()
    for split in config.allowed_splits:
        allowed_task_ids.update(load_task_ids(split))
    paths = sorted(
        {path.resolve() for root in config.input_roots for path in root.rglob("trajectory.json")}
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    input_cost_cny = 0.0
    shingles_by_task: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)

    for path in paths:
        trajectory = json.loads(path.read_text())
        input_cost_cny += float(trajectory.get("cost_cny") or 0.0)
        decision = assess_trajectory(trajectory, allowed_task_ids)
        task_id = str(trajectory.get("task_id") or "unknown")
        if decision.accepted:
            current = _action_shingles(trajectory)
            for prior_path, prior in shingles_by_task[task_id]:
                if _jaccard(current, prior) >= config.near_duplicate_threshold:
                    decision.accepted = False
                    decision.reasons.append("near_duplicate")
                    decision.near_duplicate_of = prior_path
                    break
            if decision.accepted:
                shingles_by_task[task_id].append((str(path), current))
                accepted.append(trajectory_to_sample(trajectory, path, decision, config))

        bucket = "filtered_success" if decision.accepted else "failures"
        destination = config.output_dir / bucket / task_id / path.parent.name / "trajectory.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if decision.accepted:
            destination.write_text(
                json.dumps(_without_reasoning(trajectory), indent=2, sort_keys=True) + "\n"
            )
        else:
            shutil.copy2(path, destination)
        decisions.append({"source": str(path), "destination": str(destination), **asdict(decision)})

    train, validation = _split_scenarios(accepted, config.validation_fraction)
    _write_jsonl(config.output_dir / "qwen_sft_train.jsonl", train)
    _write_jsonl(config.output_dir / "qwen_sft_validation.jsonl", validation)
    _write_jsonl(config.output_dir / "quality_decisions.jsonl", decisions)
    counts = Counter(sample["metadata"]["success_type"] for sample in accepted)
    rejection_reasons = Counter(reason for row in decisions for reason in row["reasons"])
    output_jsonls = [
        config.output_dir / "qwen_sft_train.jsonl",
        config.output_dir / "qwen_sft_validation.jsonl",
        config.output_dir / "quality_decisions.jsonl",
    ]
    report = {
        "data_version": config.data_version,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "mask_known_bad_actions": config.mask_known_bad_actions,
        "allowed_splits": config.allowed_splits,
        "input_trajectories": len(paths),
        "accepted": len(accepted),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_scenarios": len({row["metadata"]["scenario_id"] for row in train}),
        "validation_scenarios": len({row["metadata"]["scenario_id"] for row in validation}),
        "success_types": dict(counts),
        "rejection_reasons": dict(rejection_reasons),
        "teacher_cost_cny_accepted": sum(row["metadata"]["teacher_cost_cny"] for row in accepted),
        "teacher_cost_cny_input": input_cost_cny,
        "teacher_cost_cny_per_accepted": (input_cost_cny / len(accepted) if accepted else 0.0),
        "api_key_present_in_output": _output_contains_secret(output_jsonls),
        "dev_or_test_tasks_present": any(
            row["metadata"]["task_id"] not in allowed_task_ids for row in train + validation
        ),
        "assistant_steps": sum(row["metadata"]["assistant_step_count"] for row in accepted),
        "supervised_steps": sum(row["metadata"]["supervised_step_count"] for row in accepted),
        "masked_step_reasons": dict(
            sum(
                (Counter(row["metadata"]["step_mask_reason_counts"]) for row in accepted),
                Counter(),
            )
        ),
    }
    if report["api_key_present_in_output"] or report["dev_or_test_tasks_present"]:
        raise RuntimeError("Post-build data safety validation failed")
    (config.output_dir / "data_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    report_lines = ["# AppWorld SFT data quality report", ""]
    report_lines.extend(f"- {key}: {value}" for key, value in report.items())
    (config.output_dir / "quality_report.md").write_text("\n".join(report_lines) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter teacher trajectories and build Qwen SFT JSONL."
    )
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("difficulty_1_2", "difficulty_3", "difficulty_1_2_3"),
        default="difficulty_1_2",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument(
        "--unsafe-train-known-bad-actions",
        action="store_true",
        help="Restore legacy execution-failure supervision (unsafe; emits a warning).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetBuildConfig(
        input_roots=tuple(args.input_root),
        output_dir=args.output_dir,
        mode=args.mode,
        validation_fraction=args.validation_fraction,
        near_duplicate_threshold=args.near_duplicate_threshold,
        mask_known_bad_actions=not args.unsafe_train_known_bad_actions,
    )
    print(json.dumps(build_dataset(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
