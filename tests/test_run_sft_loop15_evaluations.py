from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING

import pytest

from scripts.loop7b import run_sft_loop15_evaluations as orchestration

if TYPE_CHECKING:
    from pathlib import Path


def _paths(tmp_path: Path) -> orchestration.Paths:
    return orchestration.Paths(
        repo_root=tmp_path / "repo",
        appworld_root=tmp_path / "appworld",
        output_root=tmp_path / "output",
        python_bin=tmp_path / "missing-conda-python",
        appworld_env_bin=tmp_path / "appworld-env" / "bin",
        tmpdir=tmp_path / "tmp",
        original_base=tmp_path / "original-base",
        merged_sft_base=tmp_path / "merged-sft-base",
        sft_adapter=tmp_path / "missing-sft-adapter",
        r1_run_dir=tmp_path / "missing-r1-run",
        r2_run_dir=tmp_path / "missing-r2-run",
        scenario_manifest=tmp_path / "missing-scenario-manifest.json",
    )


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "stage": "all",
        "state": None,
        "cuda_visible_devices": "0,1,2,3",
        "dev_runners": 16,
        "diagnostic_runners": 12,
        "seed": 20260718,
        "rollout_seeds": orchestration.DEFAULT_ROLLOUT_SEEDS,
        "start_port": 5555,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _make_complete_dev_output(job: orchestration.Job) -> None:
    _write_json(
        job.output_dir / "summary.json",
        [
            {
                "split": orchestration.DEV_SPLIT,
                "checkpoint_name": job.checkpoint_name,
                "TGC": 1.0,
                "SGC": 2.0,
                "average_partial_pass_rate": 0.3,
                "episode_count": 57,
                "num_rollouts_analyzed": 57,
            }
        ],
    )
    _write_json(job.output_dir / "runs" / "eval" / "behavior_summary.json", {})


def _make_complete_diagnostic_output(job: orchestration.Job) -> None:
    assert job.scenario_manifest is not None
    task_ids = [f"task_{index:02d}" for index in range(24)]
    _write_json(
        job.scenario_manifest,
        {
            "schema_version": 1,
            "kind": "appworld_loop_scenario_manifest",
            "dataset_name": orchestration.DIAGNOSTIC_SPLIT,
            "num_iterations": 15,
            "scenarios_per_iteration": 24,
            "rollouts_per_scenario": 6,
            "iterations": [
                {
                    "iteration": 1,
                    "observed_rollouts": 144,
                    "scenarios": [
                        {"scenario_idx": index, "task_id": task_id}
                        for index, task_id in enumerate(task_ids)
                    ],
                }
            ],
        },
    )
    _write_json(
        job.output_dir / "rollout_diagnostics.json",
        {
            "num_groups": 24,
            "num_rollouts": 144,
            "groups": [
                {
                    "scenario_idx": scenario_idx,
                    "task_id": task_ids[scenario_idx],
                    "rollouts": [
                        {
                            "rollout_idx": rollout_idx,
                            "generation_seed": job.rollout_seeds[rollout_idx],
                        }
                        for rollout_idx in range(6)
                    ],
                }
                for scenario_idx in range(24)
            ],
            "run": {
                "dataset_name": orchestration.DIAGNOSTIC_SPLIT,
                "temperature": 1.0,
                "max_interactions": 40,
                "num_scenarios": 24,
                "rollouts_per_scenario": 6,
                "rollout_seeds": list(job.rollout_seeds),
                "num_scenario_runners": job.num_scenario_runners,
                "trajectory_count": 144,
                "base_model_path": str(job.base_model),
                "adapter_path": (None if job.adapter_path is None else str(job.adapter_path)),
            },
        },
    )
    _write_json(
        job.output_dir / "resolved_config.json",
        {
            "num_scenario_runners": job.num_scenario_runners,
            "rollouts_per_scenario": 6,
            "rollout_seeds": list(job.rollout_seeds),
            "llm": {
                "base_model_path": str(job.base_model),
                "adapter_path": (None if job.adapter_path is None else str(job.adapter_path)),
                "temperature": 1.0,
                "vllm_server": {"max_model_len": 20000},
                "vllm_class": {"max_new_tokens": 1200},
            },
            "scenario_runner": {
                "appworld_config": {
                    "env": {"max_interactions": 40},
                    "agent": {"max_seq_len_tokens": job.context_limit},
                }
            },
        },
    )
    for scenario_idx in range(24):
        for rollout_idx in range(6):
            path = (
                job.output_dir
                / "trajectories"
                / f"iteration-{job.diagnostic_iteration:06d}"
                / f"scenario-{scenario_idx:04d}"
                / f"rollout-{rollout_idx:02d}"
                / "trajectory.json"
            )
            _write_json(
                path,
                {
                    "scenario_idx": scenario_idx,
                    "rollout_idx": rollout_idx,
                    "task_id": task_ids[scenario_idx],
                    "metadata": {"generation_seed": job.rollout_seeds[rollout_idx]},
                },
            )


def test_full_dry_plan_has_fixed_order_and_does_not_require_r1_checkpoints(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    jobs = orchestration.build_plan(_args(), paths)

    assert [job.job_id for job in jobs] == [
        "dev-r1-5",
        "dev-r1-10",
        "dev-r1-15",
        "dev-base",
        "dev-r2-5",
        "dev-r2-10",
        "dev-r2-15",
        "diagnostic-r0",
        "diagnostic-r1-5",
        "diagnostic-r1-10",
        "diagnostic-r1-15",
        "diagnostic-base",
        "diagnostic-r2-5",
        "diagnostic-r2-10",
        "diagnostic-r2-15",
    ]
    assert not paths.r1_run_dir.exists()

    dev_r1 = jobs[0]
    assert dev_r1.base_model == paths.merged_sft_base
    assert dev_r1.adapter_path == paths.r1_run_dir / "checkpoint-5"
    assert "--num-scenario-runners" in dev_r1.command
    assert "16" in dev_r1.command
    assert "--no-eager-mode" in dev_r1.command
    assert "--max-model-len" in dev_r1.command
    assert "16384" in dev_r1.command
    assert "llm.temperature=0.1" in dev_r1.command
    assert "scenario_runner.appworld_config.env.max_interactions=50" in dev_r1.command
    assert "scenario_sampler.seed=20260718" in dev_r1.command
    assert "rollout_seeds=[2026071800]" in dev_r1.command

    diagnostic_r0 = jobs[7]
    assert diagnostic_r0.base_model == paths.original_base
    assert diagnostic_r0.adapter_path == paths.sft_adapter
    assert "llm=qwen_2_5_7b_lora32_eval" in diagnostic_r0.command
    assert "rl/scenario_sampler@scenario_sampler=appworld_manifest" in diagnostic_r0.command
    assert "llm.temperature=1.0" in diagnostic_r0.command
    assert "llm.vllm_server.max_model_len=20000" in diagnostic_r0.command
    assert "+rl.learning_max_seq_len=16000" in diagnostic_r0.command
    assert "scenario_runner.appworld_config.env.max_interactions=40" in diagnostic_r0.command
    assert "num_scenario_runners=12" in diagnostic_r0.command
    assert "rollouts_per_scenario=6" in diagnostic_r0.command
    assert (
        "rollout_seeds=[2026071800,2026171800,2026271800,2026371800,2026471800,2026571800]"
        in diagnostic_r0.command
    )

    diagnostic_base = jobs[11]
    assert diagnostic_base.base_model == paths.original_base
    assert diagnostic_base.adapter_path is None
    assert "llm.adapter_path=null" in diagnostic_base.command
    assert all("test_normal" not in " ".join(job.command) for job in jobs)
    assert all("test_challenge" not in " ".join(job.command) for job in jobs)


def test_default_sft_adapter_is_checkpoint_root_with_lora_subdirectory() -> None:
    args = orchestration.parse_args([])

    assert args.sft_adapter.name == "d12_100_1epoch"
    assert args.sft_adapter.joinpath("lora", "adapter_config.json").is_file()
    assert args.sft_adapter.joinpath("lora", "adapter_model.safetensors").is_file()


def test_stage_and_state_selection_are_independent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    all_r1_5 = orchestration.build_plan(_args(state=["r1-5"]), paths)
    assert [job.job_id for job in all_r1_5] == ["dev-r1-5", "diagnostic-r1-5"]

    diagnostic_base = orchestration.build_plan(_args(stage="diagnostic", state=["base"]), paths)
    assert [job.job_id for job in diagnostic_base] == ["diagnostic-base"]

    with pytest.raises(ValueError, match="contains no evaluation jobs"):
        orchestration.build_plan(_args(stage="dev", state=["r0"]), paths)


def test_safety_gate_rejects_test_splits_and_training_modules() -> None:
    with pytest.raises(ValueError, match="test splits"):
        orchestration._assert_command_safe(
            [
                "python",
                "-m",
                "scripts.appworld.run_inference",
                "scenario_sampler.dataset_name=test_normal",
            ]
        )
    with pytest.raises(ValueError, match="non-evaluation module"):
        orchestration._assert_command_safe(
            ["python", "-m", "phi_agents.rl.train", "rl.params.total_iterations=15"]
        )


def test_output_validators_require_complete_parseable_artifacts(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    dev_job = orchestration.build_plan(_args(stage="dev", state=["base"]), paths)[0]
    diagnostic_job = orchestration.build_plan(_args(stage="diagnostic", state=["r2-5"]), paths)[0]

    assert orchestration.validate_job_output(dev_job)[0] is False
    _make_complete_dev_output(dev_job)
    assert orchestration.validate_job_output(dev_job) == (True, "complete")

    assert orchestration.validate_job_output(diagnostic_job)[0] is False
    _make_complete_diagnostic_output(diagnostic_job)
    assert orchestration.validate_job_output(diagnostic_job) == (True, "complete")

    one_trajectory = next((diagnostic_job.output_dir / "trajectories").rglob("trajectory.json"))
    one_trajectory.unlink()
    complete, reason = orchestration.validate_job_output(diagnostic_job)
    assert complete is False
    assert "found 143" in reason


def test_dry_run_writes_commands_and_manifest_without_launching_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "dry-run"

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run launched a subprocess")

    monkeypatch.setattr(orchestration.subprocess, "run", fail_if_called)
    return_code = orchestration.main(
        [
            "--dry-run",
            "--stage",
            "all",
            "--state",
            "r1-5",
            "--output-root",
            str(output_root),
            "--r1-run-dir",
            str(tmp_path / "does-not-exist"),
        ]
    )

    assert return_code == 0
    command_text = (output_root / "commands.sh").read_text(encoding="utf-8")
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert "scripts.loop7b.eval_watch" in command_text
    assert "scripts.appworld.run_inference" in command_text
    assert "test_normal" not in command_text
    assert "test_challenge" not in command_text
    assert manifest["safety"]["launches_training"] is False
    assert [row["status"] for row in manifest["jobs"]] == ["dry_run", "dry_run"]


def test_verified_job_is_skipped_on_resume_without_input_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "resume"
    argv = [
        "--stage",
        "dev",
        "--state",
        "base",
        "--output-root",
        str(output_root),
        "--original-base",
        str(tmp_path / "missing-base"),
    ]
    args = orchestration.parse_args(argv)
    paths = orchestration.Paths(
        repo_root=orchestration._path(args.repo_root),
        appworld_root=orchestration._path(args.appworld_root),
        output_root=orchestration._path(args.output_root),
        python_bin=orchestration._path(args.python_bin),
        appworld_env_bin=orchestration._path(args.appworld_env_bin),
        tmpdir=orchestration._path(args.tmpdir),
        original_base=orchestration._path(args.original_base),
        merged_sft_base=orchestration._path(args.merged_sft_base),
        sft_adapter=orchestration._path(args.sft_adapter),
        r1_run_dir=orchestration._path(args.r1_run_dir),
        r2_run_dir=orchestration._path(args.r2_run_dir),
        scenario_manifest=orchestration._path(args.scenario_manifest),
    )
    job = orchestration.build_plan(args, paths)[0]
    _make_complete_dev_output(job)
    orchestration._write_completion(job)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a completed job was launched again")

    monkeypatch.setattr(orchestration.subprocess, "run", fail_if_called)
    assert orchestration.main(argv) == 0
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["jobs"][0]["status"] == "skipped_complete"
