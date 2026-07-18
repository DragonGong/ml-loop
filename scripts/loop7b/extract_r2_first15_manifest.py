#!/usr/bin/env python3
"""Recover the first R2 LOOP scenario batches without copying rollout text.

The historical trainer did not persist scenario manifests for iterations 1--15,
but its console log records task completion and ``scenario_idx`` on adjacent
lines.  This utility extracts only task identifiers and aggregate validation
metadata.  Assistant actions, observations, credentials, and arbitrary log text
are never included in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MANIFEST_KIND = "appworld_loop_scenario_manifest"
TASK_ID_PATTERN = r"[0-9a-f]+_[0-9]+"
REQUEST_RE = re.compile(r"Requesting rollout generation with: adapter_path=(?P<adapter>.+)")
START_RE = re.compile(rf"Generating episode; .*task_id=(?P<task_id>{TASK_ID_PATTERN})")
RETURN_RE = re.compile(rf"Returning episode from (?P<task_id>{TASK_ID_PATTERN})")
COLLECTED_RE = re.compile(
    r"rank\d+: scenario_idx=(?P<scenario_idx>\d+) "
    r"n_rollouts_collected=(?P<collected>\d+) "
    r"n_rollouts_cancelled=(?P<cancelled>\d+) "
    r"n_rollouts_total=(?P<total>\d+)"
)
SUMMARY_RE = re.compile(
    r"Rollout collection completed: .*"
    r"n_rollouts_collected=(?P<collected>\d+) "
    r"n_rollouts_cancelled=(?P<cancelled>\d+) .*"
    r"n_rollouts_total=(?P<total>\d+)"
)


class ManifestExtractionError(ValueError):
    """Raised when the historical log cannot prove an exact scenario mapping."""


@dataclass
class _IterationLogState:
    request_adapter: str
    started_task_ids: list[str] = field(default_factory=list)
    pending_returned_task_ids: deque[str] = field(default_factory=deque)
    scenario_task_counts: dict[int, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    collected_events: int = 0
    summary_collected: int | None = None
    summary_cancelled: int | None = None
    summary_total: int | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_task_ids_sha256(task_ids: list[str]) -> str:
    payload = "".join(f"{task_id}\n" for task_id in task_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_adapter_iteration(adapter_text: str, iteration: int) -> None:
    if iteration == 1:
        if adapter_text != "None":
            raise ManifestExtractionError(
                "Iteration 1 was not generated from the base policy (adapter_path=None)."
            )
        return

    matches = re.findall(r"checkpoint-(\d+)(?:/lora)?", adapter_text)
    expected_checkpoint = iteration - 1
    if not matches or int(matches[-1]) != expected_checkpoint:
        raise ManifestExtractionError(
            f"Iteration {iteration} was not generated from checkpoint-{expected_checkpoint}."
        )


def _finalize_iteration(
    state: _IterationLogState,
    *,
    iteration: int,
    scenarios_per_iteration: int,
    rollouts_per_scenario: int,
) -> dict[str, Any]:
    _expected_adapter_iteration(state.request_adapter, iteration)
    expected_rollouts = scenarios_per_iteration * rollouts_per_scenario

    if state.pending_returned_task_ids:
        raise ManifestExtractionError(
            f"Iteration {iteration} has returned tasks without scenario_idx records."
        )
    if state.collected_events != expected_rollouts:
        raise ManifestExtractionError(
            f"Iteration {iteration} has {state.collected_events} completed rollouts; "
            f"expected {expected_rollouts}."
        )
    if (
        state.summary_collected != expected_rollouts
        or state.summary_total != expected_rollouts
        or state.summary_cancelled != 0
    ):
        raise ManifestExtractionError(
            f"Iteration {iteration} has an incomplete or cancelled rollout summary."
        )

    started_counts = Counter(state.started_task_ids)
    if len(state.started_task_ids) != expected_rollouts:
        raise ManifestExtractionError(
            f"Iteration {iteration} started {len(state.started_task_ids)} rollouts; "
            f"expected {expected_rollouts}."
        )
    if len(started_counts) != scenarios_per_iteration or set(started_counts.values()) != {
        rollouts_per_scenario
    }:
        raise ManifestExtractionError(
            f"Iteration {iteration} does not contain exactly "
            f"{scenarios_per_iteration} tasks x {rollouts_per_scenario} starts."
        )

    expected_indices = set(range(scenarios_per_iteration))
    if set(state.scenario_task_counts) != expected_indices:
        raise ManifestExtractionError(
            f"Iteration {iteration} does not contain scenario_idx 0.."
            f"{scenarios_per_iteration - 1}."
        )

    scenarios: list[dict[str, Any]] = []
    mapped_task_ids: list[str] = []
    for scenario_idx in range(scenarios_per_iteration):
        task_counts = state.scenario_task_counts[scenario_idx]
        if len(task_counts) != 1 or set(task_counts.values()) != {rollouts_per_scenario}:
            raise ManifestExtractionError(
                f"Iteration {iteration} scenario_idx={scenario_idx} is not mapped "
                f"to one task exactly {rollouts_per_scenario} times."
            )
        task_id = next(iter(task_counts))
        if started_counts[task_id] != rollouts_per_scenario:
            raise ManifestExtractionError(
                f"Iteration {iteration} task {task_id} start/completion counts disagree."
            )
        mapped_task_ids.append(task_id)
        scenarios.append({"scenario_idx": scenario_idx, "task_id": task_id})

    if len(set(mapped_task_ids)) != scenarios_per_iteration:
        raise ManifestExtractionError(
            f"Iteration {iteration} maps multiple scenario indices to the same task."
        )

    return {
        "iteration": iteration,
        "observed_rollouts": expected_rollouts,
        "scenarios": scenarios,
    }


def _load_expected_task_ids(path: Path) -> list[str]:
    task_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    task_ids = [task_id for task_id in task_ids if task_id]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ManifestExtractionError(f"Expected task list is empty or contains duplicates: {path}")
    return task_ids


def _validate_expected_task_cycles(
    iterations: list[dict[str, Any]], expected_task_ids: list[str]
) -> None:
    expected = set(expected_task_ids)
    for item in iterations:
        actual = {scenario["task_id"] for scenario in item["scenarios"]}
        unexpected = actual - expected
        if unexpected:
            raise ManifestExtractionError(
                f"Iteration {item['iteration']} contains unexpected task IDs: {sorted(unexpected)}"
            )

    scenarios_per_iteration = len(iterations[0]["scenarios"])
    if len(expected) % scenarios_per_iteration != 0:
        raise ManifestExtractionError(
            "Expected task count is not divisible by scenarios_per_iteration."
        )
    iterations_per_cycle = len(expected) // scenarios_per_iteration
    if iterations_per_cycle <= 1:
        return

    for offset in range(0, len(iterations), iterations_per_cycle):
        cycle = iterations[offset : offset + iterations_per_cycle]
        if len(cycle) != iterations_per_cycle:
            break
        task_sets = [{scenario["task_id"] for scenario in item["scenarios"]} for item in cycle]
        if set().union(*task_sets) != expected:
            raise ManifestExtractionError(
                f"Iterations {cycle[0]['iteration']}--{cycle[-1]['iteration']} "
                "do not cover the expected task list exactly once."
            )
        for left_index, left in enumerate(task_sets):
            for right in task_sets[left_index + 1 :]:
                if left & right:
                    raise ManifestExtractionError(
                        f"Iterations {cycle[0]['iteration']}--{cycle[-1]['iteration']} "
                        "contain repeated tasks within one expected-task cycle."
                    )


def extract_manifest(
    log_path: Path,
    *,
    num_iterations: int = 15,
    scenarios_per_iteration: int = 24,
    rollouts_per_scenario: int = 6,
    dataset_name: str = "train_difficulty_1_2",
    expected_task_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Extract and validate scenario mappings from a historical LOOP console log."""
    if min(num_iterations, scenarios_per_iteration, rollouts_per_scenario) <= 0:
        raise ValueError("Iteration, scenario, and rollout counts must be positive.")

    completed_states: list[_IterationLogState] = []
    current: _IterationLogState | None = None

    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            request_match = REQUEST_RE.search(line)
            if request_match:
                if current is not None:
                    raise ManifestExtractionError(
                        "Found a new rollout request before the previous iteration completed."
                    )
                current = _IterationLogState(request_adapter=request_match.group("adapter").strip())
                continue

            if current is None:
                continue

            if start_match := START_RE.search(line):
                current.started_task_ids.append(start_match.group("task_id"))
                continue

            if return_match := RETURN_RE.search(line):
                current.pending_returned_task_ids.append(return_match.group("task_id"))
                continue

            if collected_match := COLLECTED_RE.search(line):
                if not current.pending_returned_task_ids:
                    raise ManifestExtractionError(
                        "Found scenario_idx without a preceding returned task."
                    )
                scenario_idx = int(collected_match.group("scenario_idx"))
                task_id = current.pending_returned_task_ids.popleft()
                current.scenario_task_counts[scenario_idx][task_id] += 1
                current.collected_events += 1
                continue

            if summary_match := SUMMARY_RE.search(line):
                current.summary_collected = int(summary_match.group("collected"))
                current.summary_cancelled = int(summary_match.group("cancelled"))
                current.summary_total = int(summary_match.group("total"))
                completed_states.append(current)
                current = None
                if len(completed_states) == num_iterations:
                    break

    if len(completed_states) != num_iterations:
        raise ManifestExtractionError(
            f"Recovered {len(completed_states)} completed iterations; expected {num_iterations}."
        )

    iterations = [
        _finalize_iteration(
            state,
            iteration=iteration,
            scenarios_per_iteration=scenarios_per_iteration,
            rollouts_per_scenario=rollouts_per_scenario,
        )
        for iteration, state in enumerate(completed_states, start=1)
    ]

    expected_task_ids: list[str] | None = None
    if expected_task_ids_path is not None:
        expected_task_ids = _load_expected_task_ids(expected_task_ids_path)
        _validate_expected_task_cycles(iterations, expected_task_ids)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "dataset_name": dataset_name,
        "num_iterations": num_iterations,
        "scenarios_per_iteration": scenarios_per_iteration,
        "rollouts_per_scenario": rollouts_per_scenario,
        "source_log_sha256": _file_sha256(log_path),
        "source_log_size_bytes": log_path.stat().st_size,
        "iterations": iterations,
    }
    if expected_task_ids is not None:
        manifest["expected_task_count"] = len(expected_task_ids)
        manifest["expected_task_ids_sha256"] = _canonical_task_ids_sha256(expected_task_ids)

    validate_manifest_data(manifest)
    return manifest


def validate_manifest_data(manifest: dict[str, Any]) -> None:
    """Validate the public manifest schema without consulting the source log."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestExtractionError("Unsupported scenario manifest schema_version.")
    if manifest.get("kind") != MANIFEST_KIND:
        raise ManifestExtractionError("Unexpected scenario manifest kind.")

    num_iterations = manifest.get("num_iterations")
    scenarios_per_iteration = manifest.get("scenarios_per_iteration")
    rollouts_per_scenario = manifest.get("rollouts_per_scenario")
    if not all(
        isinstance(value, int) and value > 0
        for value in (num_iterations, scenarios_per_iteration, rollouts_per_scenario)
    ):
        raise ManifestExtractionError("Manifest counts must be positive integers.")

    iterations = manifest.get("iterations")
    if not isinstance(iterations, list) or len(iterations) != num_iterations:
        raise ManifestExtractionError("Manifest iteration count is inconsistent.")

    expected_rollouts = scenarios_per_iteration * rollouts_per_scenario
    for expected_iteration, item in enumerate(iterations, start=1):
        if not isinstance(item, dict) or item.get("iteration") != expected_iteration:
            raise ManifestExtractionError("Manifest iterations must be contiguous and one-based.")
        if item.get("observed_rollouts") != expected_rollouts:
            raise ManifestExtractionError("Manifest observed_rollouts is inconsistent.")
        scenarios = item.get("scenarios")
        if not isinstance(scenarios, list) or len(scenarios) != scenarios_per_iteration:
            raise ManifestExtractionError("Manifest scenario count is inconsistent.")
        task_ids: list[str] = []
        for expected_idx, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or scenario.get("scenario_idx") != expected_idx:
                raise ManifestExtractionError("Manifest scenario indices must be contiguous.")
            task_id = scenario.get("task_id")
            if not isinstance(task_id, str) or re.fullmatch(TASK_ID_PATTERN, task_id) is None:
                raise ManifestExtractionError("Manifest contains an invalid task ID.")
            task_ids.append(task_id)
        if len(task_ids) != len(set(task_ids)):
            raise ManifestExtractionError("Manifest iteration contains duplicate task IDs.")


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Atomically write a validated manifest as deterministic JSON."""
    validate_manifest_data(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--log", type=Path, help="Historical R2 console log to extract.")
    mode.add_argument(
        "--validate-manifest", type=Path, help="Validate an existing manifest and exit."
    )
    parser.add_argument("--output", type=Path, help="Output JSON path for extraction mode.")
    parser.add_argument("--num-iterations", type=int, default=15)
    parser.add_argument("--scenarios-per-iteration", type=int, default=24)
    parser.add_argument("--rollouts-per-scenario", type=int, default=6)
    parser.add_argument("--dataset-name", default="train_difficulty_1_2")
    parser.add_argument(
        "--expected-task-ids",
        type=Path,
        help="Optional split file used to prove complete, disjoint task cycles.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.validate_manifest is not None:
        manifest = json.loads(args.validate_manifest.read_text(encoding="utf-8"))
        validate_manifest_data(manifest)
        print(
            f"valid manifest: iterations={manifest['num_iterations']} "
            f"scenarios_per_iteration={manifest['scenarios_per_iteration']} "
            f"rollouts_per_scenario={manifest['rollouts_per_scenario']}"
        )
        return

    if args.output is None:
        raise SystemExit("--output is required with --log")
    manifest = extract_manifest(
        args.log,
        num_iterations=args.num_iterations,
        scenarios_per_iteration=args.scenarios_per_iteration,
        rollouts_per_scenario=args.rollouts_per_scenario,
        dataset_name=args.dataset_name,
        expected_task_ids_path=args.expected_task_ids,
    )
    write_manifest(manifest, args.output)
    print(
        f"wrote {args.output}: iterations={manifest['num_iterations']} "
        f"scenarios={manifest['scenarios_per_iteration']} "
        f"rollouts_per_scenario={manifest['rollouts_per_scenario']}"
    )


if __name__ == "__main__":
    main()
