from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phi_agents.api_eval.deepseek_appworld import (
    DEFAULT_COST_LIMIT_CNY,
    AppWorldApiEvalSettings,
    DeepSeekApiClient,
    DeepSeekProfileOverrides,
    build_profile,
    run_single_task_once,
    save_json,
)
from phi_agents.appworld.interface import AppWorldInterface, load_task_ids

DATA_VERSION = "appworld-deepseek-sft-v1"
ALLOWED_SPLITS = ("train_difficulty_1_2", "train_difficulty_3")
MAX_NO_CODE_FOR_SFT = 2


@dataclass(frozen=True)
class TeacherGenerationConfig:
    split: str
    output_dir: Path
    successes_per_task: int
    max_attempts_per_task: int
    cost_limit_cny: float | None
    max_interactions: int = 50
    task_retries: int = 2
    repair_failures: bool = True
    profile: str = "deepseek-v4-flash-thinking"
    task_ids: tuple[str, ...] = ()
    limit: int | None = None

    def validate(self) -> None:
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Teacher SFT generation only permits {ALLOWED_SPLITS}; got {self.split!r}."
            )
        if self.successes_per_task < 1:
            raise ValueError("successes_per_task must be positive")
        if self.max_attempts_per_task < self.successes_per_task:
            raise ValueError("max_attempts_per_task must cover successes_per_task")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative")
        split_ids = set(load_task_ids(self.split))
        unknown = sorted(set(self.task_ids) - split_ids)
        if unknown:
            raise ValueError(f"Requested task IDs are outside {self.split}: {unknown}")


def _read_manifest(path: Path, config: TeacherGenerationConfig) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text())
        prior = data.get("config") or {}
        immutable = ("split", "successes_per_task", "profile", "task_ids", "limit")
        current = _config_dict(config)
        changed = [key for key in immutable if prior.get(key) != current.get(key)]
        if changed:
            raise ValueError(f"Cannot resume with changed manifest fields: {changed}")
        return data
    return {
        "data_version": DATA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "config": _config_dict(config),
        "jobs": [],
        "cumulative_cost_cny": 0.0,
    }


def _config_dict(config: TeacherGenerationConfig) -> dict[str, Any]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    data["task_ids"] = list(config.task_ids)
    return data


def _selected_task_ids(config: TeacherGenerationConfig) -> list[str]:
    task_ids = list(config.task_ids) if config.task_ids else load_task_ids(config.split)
    return task_ids[: config.limit] if config.limit is not None else task_ids


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(temporary, manifest)
    temporary.replace(path)


def _task_progress(manifest: dict[str, Any], task_id: str) -> tuple[int, int]:
    jobs = [job for job in manifest["jobs"] if job["task_id"] == task_id]
    accepted = 0
    for job in jobs:
        accepted_for_sft = job.get("accepted_for_sft")
        if accepted_for_sft is None and job.get("success") is True:
            trajectory_path = job.get("trajectory_path")
            if trajectory_path and Path(trajectory_path).exists():
                trajectory = json.loads(Path(trajectory_path).read_text())
                no_code_count = sum(
                    bool(step.get("no_code_found")) for step in trajectory.get("steps") or []
                )
                accepted_for_sft = no_code_count <= MAX_NO_CODE_FOR_SFT
            else:
                accepted_for_sft = True
        accepted += accepted_for_sft is True
    return accepted, len(jobs)


def _last_failed_path(manifest: dict[str, Any], task_id: str) -> Path | None:
    for job in reversed(manifest["jobs"]):
        if job["task_id"] == task_id and job.get("success") is not True:
            path = Path(job["trajectory_path"])
            if path.exists():
                return path
    return None


def generation_plan(config: TeacherGenerationConfig) -> dict[str, Any]:
    config.validate()
    task_ids = _selected_task_ids(config)
    return {
        "split": config.split,
        "tasks": len(task_ids),
        "target_successes": len(task_ids) * config.successes_per_task,
        "maximum_attempts": len(task_ids) * config.max_attempts_per_task,
        "estimated_cost_cny": {
            "low": round(len(task_ids) * config.successes_per_task * 0.025, 2),
            "high": round(len(task_ids) * config.max_attempts_per_task * 0.04, 2),
            "basis": "observed DeepSeek dev run plus retry allowance",
        },
    }


def generate_teacher_trajectories(config: TeacherGenerationConfig) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "generation_manifest.json"
    manifest = _read_manifest(manifest_path, config)
    profile = build_profile(config.profile, DeepSeekProfileOverrides())
    if profile.name != "deepseek-v4-flash-thinking":
        raise ValueError("The SFT teacher must use deepseek-v4-flash-thinking by default.")

    client = DeepSeekApiClient(profile, config.cost_limit_cny)
    client.total_cost_cny = float(manifest.get("cumulative_cost_cny") or 0.0)
    if config.cost_limit_cny is not None and client.total_cost_cny >= config.cost_limit_cny:
        client.cost_limit_reached = True

    task_ids = _selected_task_ids(config)
    base_settings = AppWorldApiEvalSettings(
        model_profile=profile,
        output_dir=config.output_dir,
        experiment_name=f"teacher_{config.split}_{config.output_dir.name}",
        split_name=config.split,
        split_path=Path("data/appworld_splits") / f"{config.split}.txt",
        max_interactions=config.max_interactions,
        task_retries=config.task_retries,
        cost_limit_cny=config.cost_limit_cny,
        persist_reasoning_content=False,
    )

    with AppWorldInterface(stdout_to_devnull=True, timeout_seconds=100) as world:
        made_progress = True
        while made_progress and not client.cost_limit_reached:
            made_progress = False
            for task_id in task_ids:
                successes, attempts = _task_progress(manifest, task_id)
                if successes >= config.successes_per_task or attempts >= config.max_attempts_per_task:
                    continue
                made_progress = True
                attempt_id = f"attempt-{attempts:04d}"
                output_path = (
                    config.output_dir / "raw" / task_id / attempt_id / "trajectory.json"
                )
                repair_source = (
                    _last_failed_path(manifest, task_id) if config.repair_failures else None
                )
                settings = replace(
                    base_settings,
                    experiment_name=(
                        f"teacher_{config.split}_{task_id}_{attempt_id}_{config.output_dir.name}"
                    ),
                    repair_from_trajectory=repair_source,
                )
                for retry_index in range(config.task_retries + 1):
                    if retry_index:
                        world.ensure_server()
                    result = run_single_task_once(
                        world=world,
                        task_id=task_id,
                        settings=settings,
                        client=client,
                        output_path=output_path,
                    )
                    if result.status == "completed" or result.error_type in {
                        "cost_limit_reached",
                        "deepseek_request_error",
                    }:
                        break
                trajectory = json.loads(output_path.read_text())
                usage_rows: list[dict[str, Any]] = []
                for step in trajectory.get("steps") or []:
                    usage_rows.extend(
                        record.get("usage") or {}
                        for record in step.get("invalid_responses_before_action") or []
                    )
                    usage_rows.append(step.get("usage") or {})
                no_code_count = sum(
                    bool(step.get("no_code_found")) for step in trajectory.get("steps") or []
                )
                execution_failed_count = sum(
                    bool(step.get("execution_failed")) for step in trajectory.get("steps") or []
                )
                invalid_response_count = sum(
                    len(step.get("invalid_responses_before_action") or [])
                    for step in trajectory.get("steps") or []
                )
                accepted_for_sft = result.success is True and no_code_count <= MAX_NO_CODE_FOR_SFT
                trajectory["teacher_generation"] = {
                    "data_version": DATA_VERSION,
                    "split": config.split,
                    "attempt": attempts,
                    "repair_source": str(repair_source) if repair_source else None,
                }
                save_json(output_path, trajectory)
                manifest["jobs"].append(
                    {
                        "task_id": task_id,
                        "attempt": attempts,
                        "status": result.status,
                        "success": result.success,
                        "accepted_for_sft": accepted_for_sft,
                        "error_type": result.error_type,
                        "cost_cny": result.cost_cny,
                        "duration_seconds": float(trajectory.get("duration_seconds") or 0.0),
                        "interaction_count": int(trajectory.get("steps_count") or 0),
                        "no_code_count": no_code_count,
                        "execution_failed_count": execution_failed_count,
                        "invalid_response_count": invalid_response_count,
                        "prompt_tokens": sum(
                            int(usage.get("prompt_tokens") or 0) for usage in usage_rows
                        ),
                        "completion_tokens": sum(
                            int(usage.get("completion_tokens") or 0) for usage in usage_rows
                        ),
                        "trajectory_path": str(output_path.resolve()),
                        "repair_source": str(repair_source) if repair_source else None,
                        "task_retry_count": retry_index,
                    }
                )
                manifest["cumulative_cost_cny"] = client.total_cost_cny
                _write_manifest(manifest_path, manifest)
                if client.cost_limit_reached:
                    break

    manifest["plan"] = generation_plan(config)
    _write_manifest(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced AppWorld teacher trajectories.")
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="train_difficulty_1_2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--successes-per-task", type=int, default=14)
    parser.add_argument("--max-attempts-per-task", type=int, default=28)
    parser.add_argument("--cost-limit-cny", type=float, default=DEFAULT_COST_LIMIT_CNY)
    parser.add_argument("--max-interactions", type=int, default=50)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-repair-failures", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TeacherGenerationConfig(
        split=args.split,
        output_dir=args.output_dir,
        successes_per_task=args.successes_per_task,
        max_attempts_per_task=args.max_attempts_per_task,
        cost_limit_cny=None if args.cost_limit_cny <= 0 else args.cost_limit_cny,
        max_interactions=args.max_interactions,
        repair_failures=not args.no_repair_failures,
        task_ids=tuple(args.task_id),
        limit=args.limit,
    )
    result = generation_plan(config) if args.plan_only else generate_teacher_trajectories(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
