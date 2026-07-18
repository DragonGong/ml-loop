from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.loop7b import build_sft_loop15_report as report_builder

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _metric_payload(tgc: float, *, checkpoint_name: str) -> dict[str, Any]:
    return {
        "checkpoint_name": checkpoint_name,
        "TGC": tgc,
        "SGC": tgc / 2,
        "average_partial_pass_rate": tgc / 100,
        "TGC_1": tgc + 1,
        "TGC_2": tgc,
        "TGC_3": tgc - 1,
        "SGC_1": tgc / 2 + 1,
        "SGC_2": tgc / 2,
        "SGC_3": tgc / 2 - 1,
        "execution_failed_count": int(100 - tgc),
        "no_code_found_count": 1,
        "http_401_count": 3,
        "http_422_count": 4,
        "name_error_count": 5,
        "invalid_api_hits": 2,
        "consecutive_repeated_failed_action_count": 6,
        "average_execution_errors_before_strict_success": 0.75,
        "api_doc_calls_per_rollout": 3.0,
        "api_description_calls_per_rollout": 1.0,
        "api_doc_or_description_calls_per_rollout": 4.0,
        "num_turns_avg": 20.0,
        "context_truncation_ratio": 0.1,
        "failed_api_calls": 12,
        "recovered_api_calls": 3,
        "failed_api_call_give_up_rate": 0.75,
        "error_recovery_success_rate": 0.2,
        "error_rollout_count": 8,
        "error_rollout_strict_success_count": 2,
        "error_rollout_recovery_success_rate": 0.15,
    }


def _write_provenance(output_dir: Path, command: list[str], job_id: str) -> None:
    command_hash = report_builder._sha256_json(command)
    _write_json(
        output_dir / "orchestrator_job.json",
        {
            "job_id": job_id,
            "command": command,
            "command_sha256": command_hash,
        },
    )
    _write_json(
        output_dir / "orchestrator_complete.json",
        {"job_id": job_id, "command_sha256": command_hash},
    )


def _dev_command() -> list[str]:
    return [
        "python",
        "-m",
        "scripts.loop7b.eval_watch",
        "--split",
        "dev_small64",
        "--num-scenario-runners",
        "16",
        "--llm",
        "qwen_2_5_7b_lora16_eval",
        "--max-gpu-mem-utilization",
        "0.90",
        "--max-model-len",
        "16384",
        "--max-new-tokens",
        "1200",
        "--once",
        "--no-wait-for-gpu-idle",
        "--no-run-base-if-missing",
        "--no-eager-mode",
        "--no-reuse-complete-inference",
        "--hydra-override",
        "llm.temperature=0.1",
        "--hydra-override",
        "scenario_runner.appworld_config.env.max_interactions=50",
        "--hydra-override",
        "eval_seed=20260718",
        "--hydra-override",
        "llm.vllm_server.seed=20260718",
        "--hydra-override",
        "scenario_sampler.seed=20260718",
        "--hydra-override",
        "rollout_seeds=[2026071800]",
    ]


def _scenario_manifest(path: Path) -> list[str]:
    task_ids = [f"train_task_{index}" for index in range(24)]
    iterations = []
    for iteration in range(1, 16):
        iterations.append(
            {
                "iteration": iteration,
                "observed_rollouts": 144,
                "scenarios": [
                    {"scenario_idx": index, "task_id": task_id}
                    for index, task_id in enumerate(task_ids)
                ],
            }
        )
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": "appworld_loop_scenario_manifest",
            "dataset_name": "train_difficulty_1_2",
            "num_iterations": 15,
            "scenarios_per_iteration": 24,
            "rollouts_per_scenario": 6,
            "iterations": iterations,
        },
    )
    return task_ids


def _resolved_config(manifest_path: Path, state: str, iteration: int) -> dict[str, Any]:
    return {
        "scenario_sampler": {
            "dataset_name": "train_difficulty_1_2",
            "manifest_path": str(manifest_path.resolve()),
            "start_iteration": 1,
            "cycle": False,
            "max_parallel": 1,
        },
        "llm": {
            "temperature": 1.0,
            "max_gpu_mem_utilization": 0.9,
            "lora_rank": 32 if state == "r0" else 16,
            "vllm_server": {"max_model_len": 20000, "eager_mode": False},
            "vllm_class": {"max_new_tokens": 1200},
        },
        "scenario_runner": {
            "appworld_config": {
                "env": {"max_interactions": 40, "sparse_reward": False},
                "agent": {"max_seq_len_tokens": 16000},
            }
        },
        "rl": {"learning_max_seq_len": 16000},
        "num_scenario_runners": 12,
        "num_scenarios": 24,
        "rollouts_per_scenario": 6,
        "rollout_seeds": list(report_builder.DEFAULT_ROLLOUT_SEEDS),
        "diagnostic_iteration": iteration,
    }


def _diagnostics(task_ids: list[str], state: str, iteration: int) -> dict[str, Any]:
    groups = []
    for scenario_idx, task_id in enumerate(task_ids):
        groups.append(
            {
                "scenario_idx": scenario_idx,
                "task_id": task_id,
                "rollout_count": 6,
                "rollouts": [
                    {
                        "rollout_idx": rollout_idx,
                        "generation_seed": report_builder.DEFAULT_ROLLOUT_SEEDS[rollout_idx],
                    }
                    for rollout_idx in range(6)
                ],
            }
        )
    return {
        "num_groups": 24,
        "num_rollouts": 144,
        "zero_return_std_group_rate": 0.5,
        "mean_unique_return_count_per_group": 2.0,
        "mean_return_range_per_group": 0.4,
        "mean_unique_api_sequence_count_per_group": 3.0,
        "mean_unique_business_api_sequence_count_per_group": 2.5,
        "same_business_api_sequence_ratio": 0.2,
        "effective_advantage_rollout_ratio": 0.4,
        "effective_advantage_token_ratio": 0.45,
        "zero_return_group_class_counts": {
            "all_success": 2,
            "all_failure": 8,
            "partial_same": 2,
            "non_zero_variance": 12,
        },
        "behavior": {
            "strict_success_rate": 0.2,
            "average_partial_pass": 0.4,
            "execution_failed_count": 20,
            "no_code_found_count": 1,
            "http_401_count": 2,
            "http_422_count": 3,
            "name_error_count": 4,
            "invalid_api_hits": 2,
            "consecutive_repeated_failed_action_count": 5,
            "average_execution_errors_before_strict_success": 0.5,
            "api_doc_calls_per_rollout": 3.0,
            "api_description_calls_per_rollout": 1.0,
            "average_turn_count": 18.0,
            "context_truncation_ratio": 0.1,
            "failed_api_calls": 10,
            "recovered_api_calls": 4,
            "error_recovery_success_rate": 0.25,
            "error_rollout_count": 10,
            "error_rollout_strict_success_count": 2,
            "error_rollout_recovery_success_rate": 0.2,
            "execution_failed_per_rollout": 20 / 144,
            "no_code_found_per_rollout": 1 / 144,
            "cancelled_count": 0,
        },
        "run": {
            "dataset_name": "train_difficulty_1_2",
            "base_model_path": (
                "/models/Qwen2.5-7B-Instruct-d12_100_1epoch-merged-bf16"
                if state.startswith("r1-")
                else "/models/Qwen2.5-7B-Instruct"
            ),
            "adapter_path": (
                None
                if state == "base"
                else (
                    "/adapters/d12_100_1epoch/checkpoint-69"
                    if state == "r0"
                    else f"/experiments/{state[:2]}/checkpoint-{iteration}"
                )
            ),
            "temperature": 1.0,
            "max_interactions": 40,
            "num_scenarios": 24,
            "rollouts_per_scenario": 6,
            "rollout_seeds": list(report_builder.DEFAULT_ROLLOUT_SEEDS),
            "num_scenario_runners": 12,
            "trajectory_count": 144,
        },
        "groups": groups,
    }


def _training_metric(iteration: int) -> dict[str, Any]:
    return {
        "schema_version": "loop-iteration-training-metrics-v1",
        "status": "completed",
        "iteration": iteration,
        "actual_optimizer_steps": 2,
        "attempted_gradient_steps": 2,
        "per_token_kl": {"mean": 0.01},
        "ppo_clip_fraction": {"fraction": 0.1},
        "grad_norm_before_clipping": {"mean": 1.0, "max": 1.2},
        "parameter_update_l2_norm": {"mean": 0.2, "max": 0.3},
        "sampling_entropy": {"mean_nats": 0.8},
        "learning_rate": {
            "before_iteration_scheduler_step": 5e-5,
            "after_iteration_scheduler_step": 5e-5,
        },
        "abs_adv_threshold_filter": {
            "actually_filtered_rollout_fraction": 0.25,
            "below_threshold_rollouts": 30,
            "below_threshold_output_tokens": 300,
            "actually_filtered_output_tokens": 350,
        },
        "invalid_loss_steps": 0,
        "high_kl_events": {"iteration": 0, "cumulative": 0},
        "failure": None,
    }


def _complete_fixture(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    dev_task_list = repo_root / "data" / "appworld_splits" / "dev_small64.txt"
    dev_task_list.parent.mkdir(parents=True)
    dev_task_list.write_text("dev_a\ndev_b\ndev_c\ndev_d\n", encoding="utf-8")
    scenario_manifest = tmp_path / "scenario_manifest.json"
    task_ids = _scenario_manifest(scenario_manifest)
    evaluation_root = tmp_path / "evaluation"
    r0_metrics = tmp_path / "r0.json"
    r0 = _metric_payload(10.0, checkpoint_name="d12_100_x1")
    r0.update(
        num_rollouts_analyzed=4,
        experiment_name="compact_d12_100_x1_dev_small64",
    )
    _write_json(r0_metrics, r0)

    dev_values = {
        "r1-5": (20.0, "checkpoint-5"),
        "r1-10": (30.0, "checkpoint-10"),
        "r1-15": (40.0, "checkpoint-15"),
        "base": (5.0, "base"),
        "r2-5": (10.0, "checkpoint-5"),
        "r2-10": (15.0, "checkpoint-10"),
        "r2-15": (20.0, "checkpoint-15"),
    }
    for state, (tgc, checkpoint_name) in dev_values.items():
        output_dir = evaluation_root / "dev_small64" / state
        payload = _metric_payload(tgc, checkpoint_name=checkpoint_name)
        payload.update(split="dev_small64", episode_count=4, num_rollouts_analyzed=4)
        _write_json(output_dir / "summary.json", [payload])
        _write_provenance(output_dir, _dev_command(), f"dev-{state}")

    for _branch, state, iteration in report_builder.DIAGNOSTIC_SPECS:
        output_dir = evaluation_root / "fixed_train_diagnostic" / state
        _write_json(
            output_dir / "rollout_diagnostics.json",
            _diagnostics(task_ids, state, iteration),
        )
        _write_json(
            output_dir / "resolved_config.json",
            _resolved_config(scenario_manifest, state, iteration),
        )
        _write_provenance(
            output_dir,
            ["python", "-m", "scripts.appworld.run_inference", f"state={state}"],
            f"diagnostic-{state}",
        )

    r1_run_dir = tmp_path / "r1"
    for iteration in range(1, 16):
        _write_json(
            r1_run_dir / "training_metrics" / f"iteration-{iteration:06d}.json",
            _training_metric(iteration),
        )
    return {
        "repo_root": repo_root,
        "r0_metrics": r0_metrics,
        "evaluation_root": evaluation_root,
        "r1_run_dir": r1_run_dir,
        "scenario_manifest": scenario_manifest,
        "dev_task_list": dev_task_list,
    }


def test_complete_report_computes_deltas_auc_and_training_summary(tmp_path: Path) -> None:
    inputs = _complete_fixture(tmp_path)
    report = report_builder.build_report(**inputs)

    assert len(report["capability"]) == 8
    assert len(report["fixed_diagnostics"]) == 8
    assert len(report["r1_training_metrics"]) == 15
    assert report["capability"][0]["historical_unpaired_seed"] is True
    assert all(row["historical_unpaired_seed"] is False for row in report["capability"][1:])
    assert report["comparability"]["status"] == "warning"
    assert report["comparability"]["issue_counts"] == {
        "error": 0,
        "warning": 1,
        "missing": 0,
    }

    auc = {
        (row["branch"], row["metric"]): row["value"]
        for row in report["comparisons"]["normalized_auc"]
    }
    assert auc[("r1", "TGC")] == pytest.approx(25.0)
    assert auc[("r2", "TGC")] == pytest.approx(12.5)
    assert report["comparisons"]["normalized_auc_r1_minus_r2"]["TGC"] == pytest.approx(12.5)
    endpoint = report["comparisons"]["endpoint_deltas"]
    assert endpoint["r1_15_minus_r0"]["TGC"] == pytest.approx(30.0)
    assert endpoint["r1_15_minus_r2_15"]["TGC"] == pytest.approx(20.0)
    assert report["r1_training_summary"]["total_optimizer_steps"] == 30
    assert report_builder.DEFAULT_ROLLOUT_SEEDS == (
        2026071800,
        2026171800,
        2026271800,
        2026371800,
        2026471800,
        2026571800,
    )
    r1_5 = next(row for row in report["capability"] if row["state"] == "r1-5")
    assert r1_5["http_401_count"] == 3
    assert r1_5["http_422_count"] == 4
    assert r1_5["name_error_count"] == 5
    assert r1_5["consecutive_repeated_failed_action_count"] == 6
    assert r1_5["average_execution_errors_before_strict_success"] == 0.75
    assert r1_5["SGC_1"] == pytest.approx(11.0)
    assert r1_5["SGC_2"] == pytest.approx(10.0)
    assert r1_5["SGC_3"] == pytest.approx(9.0)
    assert r1_5["error_rollout_count"] == 8
    assert r1_5["error_rollout_strict_success_count"] == 2
    diagnostic_r1_5 = next(row for row in report["fixed_diagnostics"] if row["state"] == "r1-5")
    assert diagnostic_r1_5["http_401_count"] == 2
    assert diagnostic_r1_5["http_422_count"] == 3
    assert diagnostic_r1_5["name_error_count"] == 4
    assert diagnostic_r1_5["consecutive_repeated_failed_action_count"] == 5
    assert diagnostic_r1_5["average_execution_errors_before_strict_success"] == 0.5
    assert diagnostic_r1_5["error_rollout_count"] == 10
    assert diagnostic_r1_5["error_rollout_strict_success_count"] == 2

    output_dir = tmp_path / "report"
    paths = report_builder.write_report(output_dir, report)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["schema_version"] == (
        report_builder.SCHEMA_VERSION
    )
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "record_type" in csv_text
    assert "http_401_count" in csv_text
    assert "average_execution_errors_before_strict_success" in csv_text
    assert "error_rollout_strict_success_count" in csv_text
    assert "SGC_3" in csv_text
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "Normalized 0–15 trapezoidal AUC" in markdown
    assert "Difficulty breakdown" in markdown
    assert "Behavior and error breakdown" in markdown


def test_missing_artifacts_are_explicit_and_auc_remains_null(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    dev_task_list = repo_root / "dev_small64.txt"
    dev_task_list.parent.mkdir(parents=True)
    dev_task_list.write_text("dev_a\n", encoding="utf-8")
    scenario_manifest = tmp_path / "manifest.json"
    _scenario_manifest(scenario_manifest)
    report = report_builder.build_report(
        repo_root=repo_root,
        r0_metrics=tmp_path / "missing-r0.json",
        evaluation_root=tmp_path / "missing-evals",
        r1_run_dir=tmp_path / "missing-r1",
        scenario_manifest=scenario_manifest,
        dev_task_list=dev_task_list,
    )

    assert report["comparability"]["status"] == "incomplete"
    assert [row["available"] for row in report["capability"]] == [False] * 8
    assert [row["available"] for row in report["fixed_diagnostics"]] == [False] * 8
    assert [row["available"] for row in report["r1_training_metrics"]] == [False] * 15
    for row in report["comparisons"]["normalized_auc"]:
        assert row["value"] is None
        assert row["missing_iterations"] == [0, 5, 10, 15]

    paths = report_builder.write_report(tmp_path / "report", report)
    assert "NA" in paths["csv"].read_text(encoding="utf-8")
    assert "—" in paths["markdown"].read_text(encoding="utf-8")


def test_task_or_seed_mismatch_fails_comparability_and_strict_cli(tmp_path: Path) -> None:
    inputs = _complete_fixture(tmp_path)
    bad_path = (
        inputs["evaluation_root"] / "fixed_train_diagnostic" / "r2-15" / "rollout_diagnostics.json"
    )
    payload = json.loads(bad_path.read_text(encoding="utf-8"))
    payload["groups"][0]["task_id"] = "wrong_task"
    payload["groups"][0]["rollouts"][0]["generation_seed"] = 999
    _write_json(bad_path, payload)

    report = report_builder.build_report(**inputs)
    assert report["comparability"]["status"] == "fail"
    assert report["comparability"]["issue_counts"]["error"] >= 2
    r2_15 = next(row for row in report["fixed_diagnostics"] if row["state"] == "r2-15")
    assert r2_15["comparability_status"] == "fail"

    return_code = report_builder.main(
        [
            "--repo-root",
            str(inputs["repo_root"]),
            "--r0-metrics",
            str(inputs["r0_metrics"]),
            "--evaluation-root",
            str(inputs["evaluation_root"]),
            "--r1-run-dir",
            str(inputs["r1_run_dir"]),
            "--scenario-manifest",
            str(inputs["scenario_manifest"]),
            "--dev-task-list",
            str(inputs["dev_task_list"]),
            "--output-dir",
            str(tmp_path / "strict-report"),
            "--strict-comparability",
        ]
    )
    assert return_code == 2
    assert (tmp_path / "strict-report" / "sft_loop15_report.json").is_file()
