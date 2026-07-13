from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phi_agents.sft.trainer import audit_sample_windows, window_sample

DEFAULT_SEED = 20_260_713
SUBSET_SIZES = {"d12_25": 127, "d12_50": 255, "d12_100": 509}


@dataclass(frozen=True)
class RankedRow:
    row: dict[str, Any]
    supervised_tokens: int
    token_bin: int
    tie_breaker: str

    @property
    def row_id(self) -> str:
        metadata = self.row["metadata"]
        return str(metadata["trajectory_id"])

    def category(self, dimension: str) -> str:
        metadata = self.row["metadata"]
        if dimension == "scenario":
            return str(metadata["scenario_id"])
        if dimension == "task":
            return str(metadata["task_id"])
        if dimension == "success_type":
            return str(metadata["success_type"])
        if dimension == "token_bin":
            return str(self.token_bin)
        raise ValueError(f"Unknown balancing dimension: {dimension}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_supervision_v2(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if not str(metadata.get("data_version") or "").endswith("supervision-v2"):
            errors.append(f"{metadata.get('task_id')}: invalid data_version")
            continue
        actions = [
            message
            for message in row["messages"]
            if message.get("role") == "assistant"
            and (
                message.get("message_type") == "appworld_action"
                or message.get("step_id") is not None
            )
        ]
        if not actions or any(message.get("step_id") in (None, "") for message in actions):
            errors.append(f"{metadata.get('task_id')}: missing per-step IDs")
        if any("mask_reason" not in message for message in actions):
            errors.append(f"{metadata.get('task_id')}: missing per-step mask reason")
    if errors:
        raise ValueError("Refusing non-supervision-v2 input: " + "; ".join(errors[:10]))


def _token_counts(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        windows = window_sample(tokenizer, row, max_length=max_length, preserve_history=True)
        audit = audit_sample_windows(row, windows, strict=True)
        counts[str(row["metadata"]["trajectory_id"])] = audit["effective_supervised_tokens"]
    return counts


def _quantile_bins(values: list[int], bins: int = 5) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0] * len(values)
    for rank, index in enumerate(order):
        result[index] = min(bins - 1, rank * bins // max(1, len(values)))
    return result


def balanced_nested_order(
    rows: list[dict[str, Any]], supervised_tokens: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    """Create one deterministic order whose prefixes preserve important marginals."""
    values = [supervised_tokens[str(row["metadata"]["trajectory_id"])] for row in rows]
    token_bins = _quantile_bins(values)
    ranked = [
        RankedRow(
            row=row,
            supervised_tokens=values[index],
            token_bin=token_bins[index],
            tie_breaker=hashlib.sha256(
                f"{seed}:{row['metadata']['trajectory_id']}".encode()
            ).hexdigest(),
        )
        for index, row in enumerate(rows)
    ]
    dimensions = ("scenario", "task", "success_type", "token_bin")
    totals = {
        dimension: Counter(item.category(dimension) for item in ranked) for dimension in dimensions
    }
    selected: list[RankedRow] = []
    selected_counts = {dimension: Counter() for dimension in dimensions}
    remaining = list(ranked)
    population = len(ranked)
    while remaining:
        prefix_size = len(selected) + 1

        def score(item: RankedRow, current_prefix_size: int = prefix_size) -> tuple[float, str]:
            balance_score = 0.0
            for dimension in dimensions:
                category = item.category(dimension)
                target = current_prefix_size * totals[dimension][category] / population
                deficit = target - selected_counts[dimension][category]
                balance_score += deficit / max(1.0, totals[dimension][category] ** 0.5)
            return balance_score, item.tie_breaker

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        for dimension in dimensions:
            selected_counts[dimension][chosen.category(dimension)] += 1
    return [item.row for item in selected]


def _subset_stats(rows: list[dict[str, Any]], token_counts: dict[str, int]) -> dict[str, Any]:
    ids = [str(row["metadata"]["trajectory_id"]) for row in rows]
    tokens = [token_counts[row_id] for row_id in ids]
    return {
        "samples": len(rows),
        "scenarios": len({row["metadata"]["scenario_id"] for row in rows}),
        "tasks": len({row["metadata"]["task_id"] for row in rows}),
        "success_types": dict(Counter(row["metadata"]["success_type"] for row in rows)),
        "difficulties": dict(Counter(str(row["metadata"]["difficulty"]) for row in rows)),
        "supervised_steps": sum(row["metadata"]["supervised_step_count"] for row in rows),
        "effective_supervised_tokens": sum(tokens),
        "min_supervised_tokens": min(tokens),
        "max_supervised_tokens": max(tokens),
        "mean_supervised_tokens": sum(tokens) / len(tokens),
        "trajectory_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def build_subsets(
    train_jsonl: Path,
    validation_jsonl: Path,
    output_dir: Path,
    tokenizer: Any,
    *,
    seed: int = DEFAULT_SEED,
    max_length: int = 16_384,
) -> dict[str, Any]:
    train = _read_jsonl(train_jsonl)
    validation = _read_jsonl(validation_jsonl)
    _assert_supervision_v2(train + validation)
    if len(train) != SUBSET_SIZES["d12_100"]:
        raise ValueError(f"Expected 509 D1/2 train rows, found {len(train)}")
    train_scenarios = {row["metadata"]["scenario_id"] for row in train}
    validation_scenarios = {row["metadata"]["scenario_id"] for row in validation}
    overlap = sorted(train_scenarios & validation_scenarios)
    if overlap:
        raise ValueError(f"Train/validation scenario overlap: {overlap}")

    token_counts = _token_counts(train, tokenizer, max_length)
    order = balanced_nested_order(train, token_counts, seed)
    manifest: dict[str, Any] = {
        "seed": seed,
        "max_length": max_length,
        "source_train": str(train_jsonl.resolve()),
        "source_validation": str(validation_jsonl.resolve()),
        "source_train_sha256": _sha256(train_jsonl),
        "source_validation_sha256": _sha256(validation_jsonl),
        "scenario_overlap": overlap,
        "validation_samples": len(validation),
        "subsets": {},
    }
    subset_ids: dict[str, set[str]] = {}
    for name, size in SUBSET_SIZES.items():
        rows = order[:size]
        subset_dir = output_dir / name
        train_path = subset_dir / "qwen_sft_train.jsonl"
        validation_path = subset_dir / "qwen_sft_validation.jsonl"
        _write_jsonl(train_path, rows)
        _write_jsonl(validation_path, validation)
        subset_ids[name] = {str(row["metadata"]["trajectory_id"]) for row in rows}
        manifest["subsets"][name] = {
            **_subset_stats(rows, token_counts),
            "train_jsonl": str(train_path.resolve()),
            "validation_jsonl": str(validation_path.resolve()),
            "train_sha256": _sha256(train_path),
            "validation_sha256": _sha256(validation_path),
        }
    if not subset_ids["d12_25"] < subset_ids["d12_50"] < subset_ids["d12_100"]:
        raise RuntimeError("Nested subset invariant failed")
    manifest["nested"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic nested D1/2 SFT subsets.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print(
        json.dumps(
            build_subsets(
                args.train_jsonl,
                args.validation_jsonl,
                args.output_dir,
                tokenizer,
                seed=args.seed,
                max_length=args.max_length,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
