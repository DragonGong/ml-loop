from __future__ import annotations

import json
from pathlib import Path

import pytest

from phi_agents.rl.appworld_scenario_runner import ManifestAppWorldScenarioSampler
from scripts.loop7b.extract_r2_first15_manifest import (
    MANIFEST_KIND,
    ManifestExtractionError,
    extract_manifest,
    validate_manifest_data,
    write_manifest,
)


def _write_synthetic_log(path: Path, *, omit_last_completion: bool = False) -> None:
    iterations = (
        (("aa_1", "bb_1"), "None"),
        (("cc_1", "dd_1"), "/tmp/run/checkpoint-1/lora"),
    )
    lines: list[str] = ["TOP_SECRET_MARKER must not be copied\n"]
    for task_ids, adapter in iterations:
        lines.append(f"Requesting rollout generation with: adapter_path={adapter}\n")
        for task_id in task_ids:
            for _ in range(2):
                lines.append(f"Generating episode; worker=0 task_id={task_id}\n")
        completion_index = 0
        for scenario_idx, task_id in enumerate(task_ids):
            for _ in range(2):
                completion_index += 1
                if omit_last_completion and adapter != "None" and completion_index == 4:
                    continue
                lines.append(f"Returning episode from {task_id}\n")
                lines.append(
                    "rank0: "
                    f"scenario_idx={scenario_idx} "
                    f"n_rollouts_collected={completion_index} "
                    "n_rollouts_cancelled=0 "
                    "n_rollouts_total=4\n"
                )
        lines.append(
            "Rollout collection completed: elapsed=1 "
            "n_rollouts_collected=4 n_rollouts_cancelled=0 workers=1 "
            "n_rollouts_total=4\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _extract(log_path: Path, expected_task_ids_path: Path | None = None) -> dict:
    return extract_manifest(
        log_path,
        num_iterations=2,
        scenarios_per_iteration=2,
        rollouts_per_scenario=2,
        dataset_name="train_difficulty_1_2",
        expected_task_ids_path=expected_task_ids_path,
    )


def test_extracts_exact_mapping_without_copying_log_text(tmp_path: Path) -> None:
    log_path = tmp_path / "historical.log"
    _write_synthetic_log(log_path)
    expected_path = tmp_path / "expected.txt"
    expected_path.write_text("aa_1\nbb_1\ncc_1\ndd_1\n", encoding="utf-8")

    manifest = _extract(log_path, expected_path)
    output_path = tmp_path / "manifest.json"
    write_manifest(manifest, output_path)
    output_text = output_path.read_text(encoding="utf-8")

    assert "TOP_SECRET_MARKER" not in output_text
    assert str(log_path) not in output_text
    assert manifest["kind"] == MANIFEST_KIND
    assert manifest["expected_task_count"] == 4
    assert [scenario["task_id"] for scenario in manifest["iterations"][0]["scenarios"]] == [
        "aa_1",
        "bb_1",
    ]
    assert [scenario["task_id"] for scenario in manifest["iterations"][1]["scenarios"]] == [
        "cc_1",
        "dd_1",
    ]
    validate_manifest_data(json.loads(output_text))


def test_rejects_incomplete_historical_iteration(tmp_path: Path) -> None:
    log_path = tmp_path / "historical.log"
    _write_synthetic_log(log_path, omit_last_completion=True)

    with pytest.raises(ManifestExtractionError, match="completed rollouts"):
        _extract(log_path)


def _write_manifest(path: Path) -> None:
    task_batches = (("a_1", "b_1"), ("c_1", "d_1"), ("e_1", "f_1"))
    manifest = {
        "schema_version": 1,
        "kind": MANIFEST_KIND,
        "dataset_name": "train_difficulty_1_2",
        "num_iterations": 3,
        "scenarios_per_iteration": 2,
        "rollouts_per_scenario": 6,
        "source_log_sha256": "0" * 64,
        "source_log_size_bytes": 1,
        "iterations": [
            {
                "iteration": iteration,
                "observed_rollouts": 12,
                "scenarios": [
                    {"scenario_idx": scenario_idx, "task_id": task_id}
                    for scenario_idx, task_id in enumerate(task_ids)
                ],
            }
            for iteration, task_ids in enumerate(task_batches, start=1)
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_sampler_resumes_and_cycles_at_iteration_boundaries(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    sampler = ManifestAppWorldScenarioSampler(
        manifest_path,
        dataset_name="train_difficulty_1_2",
        start_iteration=2,
        cycle=True,
    )

    assert sampler.next_iteration == 2
    observed = [next(sampler).task_id for _ in range(6)]
    assert observed == ["c_1", "d_1", "e_1", "f_1", "a_1", "b_1"]
    assert sampler.next_iteration == 2


def test_manifest_sampler_can_stop_and_rejects_parallel_prefetch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    sampler = ManifestAppWorldScenarioSampler(
        manifest_path,
        start_iteration=3,
        cycle=False,
    )

    assert [next(sampler).task_id for _ in range(2)] == ["e_1", "f_1"]
    assert sampler.next_iteration is None
    with pytest.raises(StopIteration):
        next(sampler)
    with pytest.raises(ValueError, match="max_parallel=1"):
        ManifestAppWorldScenarioSampler(manifest_path, max_parallel=2)


def test_manifest_sampler_rejects_dataset_mismatch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    with pytest.raises(ValueError, match="does not match manifest"):
        ManifestAppWorldScenarioSampler(manifest_path, dataset_name="dev")


def test_manifest_sampler_shards_each_iteration_across_distributed_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    observed_by_rank: list[list[str]] = []
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("WORLD_SIZE", "2")
        sampler = ManifestAppWorldScenarioSampler(
            manifest_path,
            start_iteration=2,
            cycle=True,
            distributed_shard=True,
        )
        observed_by_rank.append([next(sampler).task_id for _ in range(3)])

    assert observed_by_rank == [["c_1", "e_1", "a_1"], ["d_1", "f_1", "b_1"]]


def test_manifest_sampler_rejects_non_divisible_distributed_world_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")

    with pytest.raises(ValueError, match="divisible by WORLD_SIZE"):
        ManifestAppWorldScenarioSampler(manifest_path, distributed_shard=True)


def test_r1_launcher_disables_stale_in_process_manifest_restart() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "loop7b"
        / "train_sft_loop15_rtxpro6000.sh"
    ).read_text(encoding="utf-8")

    assert "export MAX_HARD_DEAD_RESTARTS=0" in launcher
    assert "recovery_quarantine" in launcher
    assert "unfinished_iteration_" in launcher
    assert "Quarantined incomplete checkpoint-" in launcher
