#!/usr/bin/env python3
"""Run the unified R1/R2 dev and fixed-rollout evaluations, never training.

The orchestrator deliberately exposes no split override.  Capability evaluation
is fixed to ``dev_small64`` and diagnostics are fixed to the first 24 training
scenarios recorded in the historical R2 manifest.  It only launches
``scripts.loop7b.eval_watch`` or ``scripts.appworld.run_inference``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


DEV_SPLIT = "dev_small64"
DIAGNOSTIC_SPLIT = "train_difficulty_1_2"
DEV_STATES = ("r1-5", "r1-10", "r1-15", "base", "r2-5", "r2-10", "r2-15")
DIAGNOSTIC_STATES = (
    "r0",
    "r1-5",
    "r1-10",
    "r1-15",
    "base",
    "r2-5",
    "r2-10",
    "r2-15",
)
ALL_STATES = tuple(dict.fromkeys((*DIAGNOSTIC_STATES, *DEV_STATES)))
ALLOWED_MODULES = {
    "scripts.loop7b.eval_watch",
    "scripts.appworld.run_inference",
}
# SeededTrainableLLM advances a rollout's seed once per interaction.  Keep the
# six streams farther apart than the 40-turn diagnostic horizon so different
# rollout indices never reuse a per-turn seed.
DEFAULT_ROLLOUT_SEEDS = tuple(2026071800 + 100_000 * index for index in range(6))
MANIFEST_KIND = "appworld_sft_loop15_evaluation_run"
MANIFEST_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _default_python() -> Path:
    conda_python = Path.home() / "miniconda3" / "envs" / "ml-loop-py312" / "bin" / "python"
    return conda_python if conda_python.exists() else Path(sys.executable)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_rollout_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rollout seeds must be comma-separated integers") from exc
    if len(seeds) != 6 or len(set(seeds)) != 6 or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "exactly six distinct non-negative rollout seeds are required"
        )
    return seeds


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    appworld_root: Path
    output_root: Path
    python_bin: Path
    appworld_env_bin: Path
    tmpdir: Path
    original_base: Path
    merged_sft_base: Path
    sft_adapter: Path
    r1_run_dir: Path
    r2_run_dir: Path
    scenario_manifest: Path


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    state: str
    command: tuple[str, ...]
    output_dir: Path
    base_model: Path
    adapter_path: Path | None
    checkpoint_name: str | None
    diagnostic_iteration: int | None
    num_scenario_runners: int
    context_limit: int
    rollout_seeds: tuple[int, ...] = ()
    scenario_manifest: Path | None = None

    @property
    def command_sha256(self) -> str:
        return _sha256_json(list(self.command))

    @property
    def completion_path(self) -> Path:
        return self.output_dir / "orchestrator_complete.json"

    @property
    def launch_path(self) -> Path:
        return self.output_dir / "orchestrator_job.json"


def _state_iteration(state: str) -> int:
    if "-" not in state:
        return 0
    return int(state.rsplit("-", 1)[1])


def _state_model_and_adapter(state: str, paths: Paths) -> tuple[Path, Path | None, str]:
    if state == "r0":
        return paths.original_base, paths.sft_adapter, "qwen_2_5_7b_lora32_eval"
    if state == "base":
        return paths.original_base, None, "qwen_2_5_7b_lora16_eval"
    iteration = _state_iteration(state)
    if state.startswith("r1-"):
        return (
            paths.merged_sft_base,
            paths.r1_run_dir / f"checkpoint-{iteration}",
            "qwen_2_5_7b_lora16_eval",
        )
    if state.startswith("r2-"):
        return (
            paths.original_base,
            paths.r2_run_dir / f"checkpoint-{iteration}",
            "qwen_2_5_7b_lora16_eval",
        )
    raise ValueError(f"unknown model state: {state}")


def _assert_command_safe(command: Sequence[str]) -> None:
    """Refuse training entrypoints and all formal AppWorld test splits."""
    try:
        module = command[command.index("-m") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("evaluation command must launch an allow-listed Python module") from exc
    if module not in ALLOWED_MODULES:
        raise ValueError(f"refusing non-evaluation module: {module}")

    lowered = " ".join(command).lower()
    forbidden = ("test_normal", "test_challenge", "--allow-test-split")
    if any(marker in lowered for marker in forbidden):
        raise ValueError("formal AppWorld test splits are forbidden by this experiment")


def _dev_command(
    *,
    state: str,
    paths: Paths,
    cuda_visible_devices: str,
    runners: int,
    seed: int,
) -> Job:
    base_model, adapter_path, llm = _state_model_and_adapter(state, paths)
    iteration = _state_iteration(state)
    output_dir = paths.output_root / "dev_small64" / state
    mode = "base" if state == "base" else "checkpoint"
    run_name = "sft_loop15_r1" if state.startswith("r1-") else "base_loop_r2_historical"
    checkpoint_name = "base" if state == "base" else f"checkpoint-{iteration}"

    command = [
        str(paths.python_bin),
        "-m",
        "scripts.loop7b.eval_watch",
        "--mode",
        mode,
        "--repo-root",
        str(paths.repo_root),
        "--appworld-root",
        str(paths.appworld_root),
        "--run-name",
        run_name,
        "--summary-dir",
        str(output_dir),
        "--split",
        DEV_SPLIT,
        "--once",
        "--no-wait-for-gpu-idle",
        "--no-run-base-if-missing",
        "--repeat",
        "1",
        "--python-bin",
        str(paths.python_bin),
        "--appworld-env-bin",
        str(paths.appworld_env_bin),
        "--tmpdir",
        str(paths.tmpdir),
        "--cuda-visible-devices",
        cuda_visible_devices,
        "--num-scenario-runners",
        str(runners),
        "--llm",
        llm,
        "--max-gpu-mem-utilization",
        "0.90",
        "--no-eager-mode",
        "--max-model-len",
        "16384",
        "--max-new-tokens",
        "1200",
        "--no-reuse-complete-inference",
        "--hydra-override",
        f"llm.base_model_path={base_model}",
        "--hydra-override",
        "llm.temperature=0.1",
        "--hydra-override",
        "scenario_runner.appworld_config.env.max_interactions=50",
        "--hydra-override",
        f"eval_seed={seed}",
        "--hydra-override",
        f"llm.vllm_server.seed={seed}",
        "--hydra-override",
        f"scenario_sampler.seed={seed}",
        "--hydra-override",
        f"rollout_seeds=[{DEFAULT_ROLLOUT_SEEDS[0]}]",
    ]
    if adapter_path is not None:
        command.extend(("--checkpoint-path", str(adapter_path)))
    _assert_command_safe(command)
    return Job(
        job_id=f"dev-{state}",
        stage="dev",
        state=state,
        command=tuple(command),
        output_dir=output_dir,
        base_model=base_model,
        adapter_path=adapter_path,
        checkpoint_name=checkpoint_name,
        diagnostic_iteration=None,
        num_scenario_runners=runners,
        context_limit=16384,
    )


def _diagnostic_command(
    *,
    state: str,
    paths: Paths,
    cuda_visible_devices: str,
    runners: int,
    seed: int,
    rollout_seeds: tuple[int, ...],
    start_port: int,
) -> Job:
    base_model, adapter_path, llm = _state_model_and_adapter(state, paths)
    iteration = _state_iteration(state)
    output_dir = paths.output_root / "fixed_train_diagnostic" / state
    seeds_override = "[" + ",".join(str(item) for item in rollout_seeds) + "]"
    adapter_override = "null" if adapter_path is None else str(adapter_path)
    experiment_name = f"loop15_fixed_diagnostic_{state.replace('-', '_')}"
    command = [
        str(paths.python_bin),
        "-m",
        "scripts.appworld.run_inference",
        "rl/scenario_sampler@scenario_sampler=appworld_manifest",
        f"scenario_sampler.manifest_path={paths.scenario_manifest}",
        f"scenario_sampler.dataset_name={DIAGNOSTIC_SPLIT}",
        "scenario_sampler.start_iteration=1",
        "scenario_sampler.cycle=false",
        "scenario_sampler.max_parallel=1",
        f"experiment_name={experiment_name}",
        f"llm={llm}",
        f"llm.base_model_path={base_model}",
        f"llm.adapter_path={adapter_override}",
        "llm.temperature=1.0",
        "llm.max_gpu_mem_utilization=0.90",
        "llm.vllm_server.gpus_per_vllm_server=1",
        "llm.vllm_server.max_model_len=20000",
        "+rl.learning_max_seq_len=16000",
        "llm.vllm_class.max_new_tokens=1200",
        "llm.vllm_server.eager_mode=false",
        f"llm.vllm_server.seed={seed}",
        "scenario_runner.appworld_config.env.max_interactions=40",
        "scenario_runner.appworld_config.env.sparse_reward=false",
        f"eval_seed={seed}",
        f"num_scenario_runners={runners}",
        "num_scenarios=24",
        "rollouts_per_scenario=6",
        f"rollout_seeds={seeds_override}",
        f"diagnostic_output_dir={output_dir}",
        f"diagnostic_iteration={iteration}",
        f"start_port={start_port}",
    ]
    _assert_command_safe(command)
    return Job(
        job_id=f"diagnostic-{state}",
        stage="diagnostic",
        state=state,
        command=tuple(command),
        output_dir=output_dir,
        base_model=base_model,
        adapter_path=adapter_path,
        checkpoint_name=None,
        diagnostic_iteration=iteration,
        num_scenario_runners=runners,
        context_limit=16000,
        rollout_seeds=rollout_seeds,
        scenario_manifest=paths.scenario_manifest,
    )


def build_plan(args: argparse.Namespace, paths: Paths) -> list[Job]:
    requested_states = set(args.state or ALL_STATES)
    jobs: list[Job] = []
    if args.stage in {"all", "dev"}:
        jobs.extend(
            _dev_command(
                state=state,
                paths=paths,
                cuda_visible_devices=args.cuda_visible_devices,
                runners=args.dev_runners,
                seed=args.seed,
            )
            for state in DEV_STATES
            if state in requested_states
        )
    if args.stage in {"all", "diagnostic"}:
        jobs.extend(
            _diagnostic_command(
                state=state,
                paths=paths,
                cuda_visible_devices=args.cuda_visible_devices,
                runners=args.diagnostic_runners,
                seed=args.seed,
                rollout_seeds=args.rollout_seeds,
                start_port=args.start_port,
            )
            for state in DIAGNOSTIC_STATES
            if state in requested_states
        )
    if not jobs:
        raise ValueError("the selected stage/state combination contains no evaluation jobs")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("evaluation plan contains duplicate job IDs")
    return jobs


def _validate_model_dir(path: Path) -> None:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"base model is missing or incomplete: {path}")


def _validate_adapter_dir(path: Path) -> None:
    lora_dir = path / "lora"
    required = (lora_dir / "adapter_config.json", lora_dir / "adapter_model.safetensors")
    missing = [item for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"adapter checkpoint is incomplete at {path}; missing: "
            + ", ".join(str(item) for item in missing)
        )


def _validate_scenario_manifest(path: Path) -> tuple[str, ...]:
    payload = _read_json(path)
    contract = (
        payload.get("schema_version"),
        payload.get("kind"),
        payload.get("dataset_name"),
        payload.get("num_iterations"),
        payload.get("scenarios_per_iteration"),
        payload.get("rollouts_per_scenario"),
    )
    expected = (
        1,
        "appworld_loop_scenario_manifest",
        DIAGNOSTIC_SPLIT,
        15,
        24,
        6,
    )
    if contract != expected:
        raise ValueError(f"unexpected fixed-scenario manifest contract: {contract!r}")
    iterations = payload.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        raise ValueError("fixed-scenario manifest has no iterations")
    first = iterations[0]
    scenarios = first.get("scenarios") if isinstance(first, dict) else None
    task_ids = [item.get("task_id") for item in scenarios or [] if isinstance(item, dict)]
    scenario_indices = [
        item.get("scenario_idx") for item in scenarios or [] if isinstance(item, dict)
    ]
    if (
        first.get("iteration") != 1
        or first.get("observed_rollouts") != 144
        or len(task_ids) != 24
        or len(set(task_ids)) != 24
        or scenario_indices != list(range(24))
        or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
    ):
        raise ValueError("the first manifest iteration is not exactly 24 scenarios x 6 rollouts")
    return tuple(task_ids)


def _validate_job_inputs(job: Job, paths: Paths) -> None:
    _assert_command_safe(job.command)
    _validate_model_dir(job.base_model)
    if job.adapter_path is not None:
        _validate_adapter_dir(job.adapter_path)
    if job.stage == "dev":
        repo_split = paths.repo_root / "data" / "appworld_splits" / f"{DEV_SPLIT}.txt"
        appworld_split = paths.appworld_root / "data" / "datasets" / f"{DEV_SPLIT}.txt"
        if not repo_split.is_file() or not appworld_split.is_file():
            raise FileNotFoundError("dev_small64 must exist in both repo and AppWorld data roots")
        if repo_split.read_bytes() != appworld_split.read_bytes():
            raise ValueError("repo and AppWorld dev_small64 task lists differ")
    else:
        _validate_scenario_manifest(paths.scenario_manifest)


def _dev_output_validation(job: Job, expected_episodes: int = 57) -> tuple[bool, str]:
    summary_path = job.output_dir / "summary.json"
    if not summary_path.is_file():
        return False, f"missing {summary_path.name}"
    try:
        rows = _read_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid summary JSON: {exc}"
    if not isinstance(rows, list):
        return False, "summary JSON is not a list"
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("split") == DEV_SPLIT
        and row.get("checkpoint_name") == job.checkpoint_name
    ]
    if not matches:
        return False, "expected checkpoint row is absent"
    row = matches[-1]
    required_metrics = ("TGC", "SGC", "average_partial_pass_rate")
    if any(row.get(metric) is None for metric in required_metrics):
        return False, "primary metrics are incomplete"
    if row.get("episode_count") != expected_episodes:
        return False, f"episode_count is not {expected_episodes}"
    if row.get("num_rollouts_analyzed") not in (None, expected_episodes):
        return False, f"num_rollouts_analyzed is not {expected_episodes}"
    behavior_files = list((job.output_dir / "runs").glob("*/behavior_summary.json"))
    if not behavior_files:
        return False, "behavior summary artifact is absent"
    return True, "complete"


def _diagnostic_output_validation(job: Job) -> tuple[bool, str]:
    diagnostics_path = job.output_dir / "rollout_diagnostics.json"
    resolved_config_path = job.output_dir / "resolved_config.json"
    if not diagnostics_path.is_file() or not resolved_config_path.is_file():
        return False, "diagnostics or resolved config is absent"
    try:
        diagnostics = _read_json(diagnostics_path)
        resolved = _read_json(resolved_config_path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid diagnostic JSON: {exc}"
    if job.scenario_manifest is None:
        return False, "diagnostic job has no fixed-scenario manifest"
    try:
        expected_task_ids = _validate_scenario_manifest(job.scenario_manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, f"fixed-scenario manifest is invalid: {exc}"
    run = diagnostics.get("run", {}) if isinstance(diagnostics, dict) else {}
    if diagnostics.get("num_groups") != 24 or diagnostics.get("num_rollouts") != 144:
        return False, "diagnostics do not contain 24 groups and 144 rollouts"
    expected_run = {
        "dataset_name": DIAGNOSTIC_SPLIT,
        "temperature": 1.0,
        "max_interactions": 40,
        "num_scenarios": 24,
        "rollouts_per_scenario": 6,
        "rollout_seeds": list(job.rollout_seeds),
        "num_scenario_runners": job.num_scenario_runners,
        "trajectory_count": 144,
        "base_model_path": str(job.base_model),
        "adapter_path": None if job.adapter_path is None else str(job.adapter_path),
    }
    for key, expected in expected_run.items():
        if run.get(key) != expected:
            return False, f"diagnostic run field {key!r} does not match"
    resolved_contract = {
        "num_scenario_runners": resolved.get("num_scenario_runners"),
        "rollouts_per_scenario": resolved.get("rollouts_per_scenario"),
        "rollout_seeds": resolved.get("rollout_seeds"),
        "base_model_path": resolved.get("llm", {}).get("base_model_path"),
        "adapter_path": resolved.get("llm", {}).get("adapter_path"),
        "temperature": resolved.get("llm", {}).get("temperature"),
        "vllm_max_model_len": resolved.get("llm", {}).get("vllm_server", {}).get("max_model_len"),
        "max_new_tokens": resolved.get("llm", {}).get("vllm_class", {}).get("max_new_tokens"),
        "max_interactions": resolved.get("scenario_runner", {})
        .get("appworld_config", {})
        .get("env", {})
        .get("max_interactions"),
        "agent_context_limit": resolved.get("scenario_runner", {})
        .get("appworld_config", {})
        .get("agent", {})
        .get("max_seq_len_tokens"),
    }
    expected_resolved_contract = {
        "num_scenario_runners": job.num_scenario_runners,
        "rollouts_per_scenario": 6,
        "rollout_seeds": list(job.rollout_seeds),
        "base_model_path": str(job.base_model),
        "adapter_path": None if job.adapter_path is None else str(job.adapter_path),
        "temperature": 1.0,
        "vllm_max_model_len": 20000,
        "max_new_tokens": 1200,
        "max_interactions": 40,
        "agent_context_limit": job.context_limit,
    }
    if resolved_contract != expected_resolved_contract:
        return False, (
            "resolved config contract differs: "
            f"observed={resolved_contract!r}, expected={expected_resolved_contract!r}"
        )

    groups = diagnostics.get("groups")
    if not isinstance(groups, list) or len(groups) != 24:
        return False, "diagnostic groups are absent or incomplete"
    groups_by_index = {
        group.get("scenario_idx"): group for group in groups if isinstance(group, dict)
    }
    if set(groups_by_index) != set(range(24)):
        return False, "diagnostic scenario indices are incomplete or duplicated"
    expected_seed_mapping = dict(enumerate(job.rollout_seeds))
    for scenario_idx, expected_task_id in enumerate(expected_task_ids):
        group = groups_by_index[scenario_idx]
        if group.get("task_id") != expected_task_id:
            return False, f"diagnostic task order differs at scenario {scenario_idx}"
        group_rollouts = group.get("rollouts")
        if not isinstance(group_rollouts, list) or len(group_rollouts) != 6:
            return False, f"diagnostic scenario {scenario_idx} does not have six rollouts"
        observed_seed_mapping = {
            rollout.get("rollout_idx"): rollout.get("generation_seed")
            for rollout in group_rollouts
            if isinstance(rollout, dict)
        }
        if observed_seed_mapping != expected_seed_mapping:
            return False, f"diagnostic group seed mapping differs at scenario {scenario_idx}"
    trajectory_paths = list(
        (job.output_dir / "trajectories").glob(
            f"iteration-{job.diagnostic_iteration:06d}/scenario-*/rollout-*/trajectory.json"
        )
    )
    if len(trajectory_paths) != 144:
        return False, f"expected 144 trajectory files, found {len(trajectory_paths)}"
    seeds_by_scenario: dict[int, dict[int, int]] = {}
    try:
        for path in trajectory_paths:
            trajectory = _read_json(path)
            scenario_idx = trajectory.get("scenario_idx")
            rollout_idx = trajectory.get("rollout_idx")
            task_id = trajectory.get("task_id")
            generation_seed = trajectory.get("metadata", {}).get("generation_seed")
            if not all(
                isinstance(value, int) for value in (scenario_idx, rollout_idx, generation_seed)
            ):
                return False, f"trajectory indices/seed are invalid: {path}"
            scenario_rollouts = seeds_by_scenario.setdefault(scenario_idx, {})
            if rollout_idx in scenario_rollouts:
                return False, f"duplicate rollout index in scenario {scenario_idx}"
            if not 0 <= scenario_idx < 24 or task_id != expected_task_ids[scenario_idx]:
                return False, f"trajectory task order differs: {path}"
            scenario_rollouts[rollout_idx] = generation_seed
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid trajectory JSON: {exc}"
    if set(seeds_by_scenario) != set(range(24)) or any(
        rollout_seeds != expected_seed_mapping for rollout_seeds in seeds_by_scenario.values()
    ):
        return False, "per-scenario rollout indices or generation seeds do not match"
    return True, "complete"


def validate_job_output(job: Job) -> tuple[bool, str]:
    if job.stage == "dev":
        return _dev_output_validation(job)
    return _diagnostic_output_validation(job)


def _completion_matches(job: Job) -> bool:
    if not job.completion_path.is_file():
        return False
    try:
        payload = _read_json(job.completion_path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("command_sha256") == job.command_sha256


def _launch_matches(job: Job) -> bool:
    if not job.launch_path.is_file():
        return False
    try:
        payload = _read_json(job.launch_path)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("command_sha256") == job.command_sha256


def _write_launch(job: Job) -> None:
    _write_json(
        job.launch_path,
        {
            "schema_version": 1,
            "kind": "appworld_sft_loop15_evaluation_job",
            "launched_at": _utc_now(),
            "job_id": job.job_id,
            "command": list(job.command),
            "command_sha256": job.command_sha256,
        },
    )


def _write_completion(job: Job) -> None:
    _write_json(
        job.completion_path,
        {
            "schema_version": 1,
            "kind": "appworld_sft_loop15_evaluation_job_complete",
            "completed_at": _utc_now(),
            "job_id": job.job_id,
            "command_sha256": job.command_sha256,
        },
    )


def _job_manifest_row(job: Job, *, status: str) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "state": job.state,
        "status": status,
        "command": list(job.command),
        "command_sha256": job.command_sha256,
        "output_dir": str(job.output_dir),
        "base_model": str(job.base_model),
        "adapter_path": None if job.adapter_path is None else str(job.adapter_path),
        "checkpoint_name": job.checkpoint_name,
        "diagnostic_iteration": job.diagnostic_iteration,
        "num_scenario_runners": job.num_scenario_runners,
        "context_limit": job.context_limit,
        "rollout_seeds": list(job.rollout_seeds),
        "scenario_manifest": (
            None if job.scenario_manifest is None else str(job.scenario_manifest)
        ),
    }


def _base_manifest(args: argparse.Namespace, paths: Paths, jobs: Sequence[Job]) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "created_at": now,
        "updated_at": now,
        "dry_run": bool(args.dry_run),
        "safety": {
            "launches_training": False,
            "allowed_python_modules": sorted(ALLOWED_MODULES),
            "capability_split": DEV_SPLIT,
            "diagnostic_split": DIAGNOSTIC_SPLIT,
            "formal_test_splits_forbidden": True,
        },
        "comparability_notes": {
            "new_dev_states_use_paired_request_seed": True,
            "r0_dev_is_reused_historical_result": True,
            "r0_historical_unpaired_seed": True,
        },
        "settings": {
            "repo_root": str(paths.repo_root),
            "appworld_root": str(paths.appworld_root),
            "output_root": str(paths.output_root),
            "python_bin": str(paths.python_bin),
            "cuda_visible_devices": args.cuda_visible_devices,
            "dev_runners": args.dev_runners,
            "diagnostic_runners": args.diagnostic_runners,
            "seed": args.seed,
            "rollout_seeds": list(args.rollout_seeds),
            "start_port": args.start_port,
            "stage": args.stage,
            "selected_states": list(args.state or ALL_STATES),
            "scenario_manifest": str(paths.scenario_manifest),
        },
        "jobs": [_job_manifest_row(job, status="planned") for job in jobs],
    }


def _write_commands(
    path: Path, paths: Paths, args: argparse.Namespace, jobs: Sequence[Job]
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(paths.repo_root))}",
        f"export APPWORLD_ROOT={shlex.quote(str(paths.appworld_root))}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)}",
        f"export TMPDIR={shlex.quote(str(paths.tmpdir))}",
        f"export RAY_TMPDIR={shlex.quote(str(paths.tmpdir))}",
        f'export PATH={shlex.quote(str(paths.appworld_env_bin))}:"$PATH"',
        "",
    ]
    for job in jobs:
        lines.extend((f"# {job.job_id}", shlex.join(job.command), ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _prepare_environment(args: argparse.Namespace, paths: Paths) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "APPWORLD_ROOT": str(paths.appworld_root),
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "TMPDIR": str(paths.tmpdir),
            "RAY_TMPDIR": str(paths.tmpdir),
            "PATH": f"{paths.appworld_env_bin}{os.pathsep}{env.get('PATH', '')}",
        }
    )
    return env


def _manifest_job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    return next(row for row in manifest["jobs"] if row["job_id"] == job_id)


def _persist_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _write_json(path, manifest)


def _quarantine_incomplete_output(job: Job, output_root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = output_root / "quarantine" / f"{job.job_id}-{timestamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.replace(destination)
    return destination


def run(args: argparse.Namespace) -> int:
    paths = Paths(
        repo_root=_path(args.repo_root),
        appworld_root=_path(args.appworld_root),
        output_root=_path(args.output_root),
        python_bin=_path(args.python_bin),
        appworld_env_bin=_path(args.appworld_env_bin),
        tmpdir=_path(args.tmpdir),
        original_base=_path(args.original_base),
        merged_sft_base=_path(args.merged_sft_base),
        sft_adapter=_path(args.sft_adapter),
        r1_run_dir=_path(args.r1_run_dir),
        r2_run_dir=_path(args.r2_run_dir),
        scenario_manifest=_path(args.scenario_manifest),
    )
    if args.dev_runners <= 0 or args.diagnostic_runners <= 0:
        raise ValueError("runner counts must be positive")
    if args.start_port <= 0 or args.start_port > 65535:
        raise ValueError("start port must be in 1..65535")
    jobs = build_plan(args, paths)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.tmpdir.mkdir(parents=True, exist_ok=True)
    command_path = paths.output_root / "commands.sh"
    manifest_path = paths.output_root / "run_manifest.json"
    _write_commands(command_path, paths, args, jobs)
    manifest = _base_manifest(args, paths, jobs)
    manifest["command_file"] = str(command_path)
    _persist_manifest(manifest_path, manifest)

    print(f"command list: {command_path}")
    print(f"run manifest: {manifest_path}")
    if args.dry_run:
        for job in jobs:
            print(f"[dry-run] {job.job_id}: {shlex.join(job.command)}")
            _manifest_job(manifest, job.job_id)["status"] = "dry_run"
        _persist_manifest(manifest_path, manifest)
        return 0

    env = _prepare_environment(args, paths)
    for job in jobs:
        row = _manifest_job(manifest, job.job_id)
        complete, reason = validate_job_output(job)
        if complete and _completion_matches(job):
            row.update(status="skipped_complete", completion_reason=reason)
            _persist_manifest(manifest_path, manifest)
            print(f"[skip] {job.job_id}: verified complete")
            continue
        if complete and _launch_matches(job):
            _write_completion(job)
            row.update(
                status="skipped_complete",
                completion_reason="adopted output from matching interrupted orchestrator run",
            )
            _persist_manifest(manifest_path, manifest)
            print(f"[skip] {job.job_id}: adopted verified complete output")
            continue
        if complete:
            raise RuntimeError(
                f"{job.job_id} output is complete but has no matching command provenance; "
                f"use a different --output-root or move {job.output_dir} aside"
            )

        if job.output_dir.exists() and any(job.output_dir.iterdir()):
            if not _launch_matches(job):
                raise RuntimeError(
                    f"{job.job_id} has incomplete output without matching command provenance; "
                    f"refusing to overwrite {job.output_dir}"
                )
            quarantined = _quarantine_incomplete_output(job, paths.output_root)
            row["quarantined_incomplete_output"] = str(quarantined)
            _persist_manifest(manifest_path, manifest)
            print(f"[quarantine] {job.job_id}: {quarantined}")

        _validate_job_inputs(job, paths)
        row.update(status="running", started_at=_utc_now(), pre_run_output_status=reason)
        _persist_manifest(manifest_path, manifest)
        _write_launch(job)
        print(f"[run] {job.job_id}: {shlex.join(job.command)}", flush=True)
        try:
            subprocess.run(
                list(job.command),
                cwd=paths.repo_root,
                env=env,
                check=True,
            )
            complete, reason = validate_job_output(job)
            if not complete:
                raise RuntimeError(f"job returned successfully but output is incomplete: {reason}")
            _write_completion(job)
            row.update(status="completed", completed_at=_utc_now(), completion_reason=reason)
            _persist_manifest(manifest_path, manifest)
        except BaseException as exc:
            row.update(status="failed", failed_at=_utc_now(), error=f"{type(exc).__name__}: {exc}")
            _persist_manifest(manifest_path, manifest)
            raise
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "dev", "diagnostic"), default="all")
    parser.add_argument("--state", action="append", choices=ALL_STATES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--appworld-root",
        type=Path,
        default=Path(os.environ.get("APPWORLD_ROOT", "/home/yunlong/dragongong/appworld-data")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts" / "sft_loop15" / "unified_evaluations",
    )
    parser.add_argument("--python-bin", type=Path, default=_default_python())
    parser.add_argument("--appworld-env-bin", type=Path, default=repo_root / "appworld-env" / "bin")
    parser.add_argument("--tmpdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--dev-runners", type=int, default=16)
    parser.add_argument("--diagnostic-runners", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--rollout-seeds",
        type=_parse_rollout_seeds,
        default=DEFAULT_ROLLOUT_SEEDS,
        help="exactly six comma-separated seeds shared by every diagnostic state",
    )
    parser.add_argument("--start-port", type=int, default=5555)
    parser.add_argument(
        "--original-base",
        type=Path,
        default=repo_root / ".model_cache" / "Qwen" / "Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--merged-sft-base",
        type=Path,
        default=(
            repo_root / ".model_cache" / "Qwen" / "Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16"
        ),
    )
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "appworld_sft"
            / "compact_lora_20260713"
            / "runs"
            / "d12_100_1epoch"
        ),
        help=(
            "checkpoint root containing lora/; run_inference resolves the actual "
            "adapter with file_utils.lora_path"
        ),
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
        "--r2-run-dir",
        type=Path,
        default=(
            repo_root / "experiments" / "qwen25_7b_loop_200x24x6_lora16" / "2026-06-08_10-12-49"
        ),
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=repo_root / "artifacts" / "sft_loop15" / "r2_first15_scenario_manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
