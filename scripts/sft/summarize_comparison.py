from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METRICS = (
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
    "invalid_api_hits",
    "api_doc_calls_per_rollout",
    "num_turns_avg",
    "error_recovery_success_rate",
    "context_truncation_ratio",
    "train_loss",
    "validation_loss",
    "effective_supervised_token_exposures",
    "optimizer_steps",
    "train_runtime",
    "peak_gpu_memory_bytes",
)


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    strict_success = float(row.get("TGC") or 0.0)
    partial_pass = float(row.get("average_partial_pass_rate") or 0.0)
    execution_anomalies = float(row.get("execution_failed_count") or 0.0) + float(
        row.get("no_code_found_count") or 0.0
    )
    raw_validation_loss = row.get("validation_loss")
    validation_loss = float(
        raw_validation_loss if raw_validation_loss is not None else float("inf")
    )
    return (strict_success, partial_pass, -execution_anomalies, -validation_loss)


def summarize(
    paths: list[Path],
    output_dir: Path,
    training_metrics: dict[str, Path] | None = None,
) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in paths]
    training_metrics = training_metrics or {}
    for row in rows:
        metrics_path = training_metrics.get(row["checkpoint_name"])
        if metrics_path is not None:
            metrics = json.loads(metrics_path.read_text())
            row.update(
                {
                    "train_loss": metrics.get("train_loss"),
                    "validation_loss": metrics.get("eval_loss"),
                    "effective_supervised_token_exposures": metrics.get(
                        "effective_supervised_token_exposures"
                    ),
                    "optimizer_steps": metrics.get("optimizer_steps"),
                    "train_runtime": metrics.get("train_runtime"),
                    "peak_gpu_memory_bytes": metrics.get("peak_gpu_memory_bytes"),
                }
            )
    eligible = [row for row in rows if row["checkpoint_name"] in training_metrics]
    report = {
        "models": {row["checkpoint_name"]: {key: row.get(key) for key in METRICS} for row in rows},
        "recommended_checkpoint": (
            max(eligible, key=_selection_key)["checkpoint_name"] if eligible else None
        ),
        "selection_priority": [
            "strict_success",
            "partial_pass",
            "execution_failed_plus_no_code",
            "validation_loss",
        ],
        "failed_task_types": {
            row["checkpoint_name"]: row.get("failed_task_type_counts", {}) for row in rows
        },
        "interpretation_notes": [
            "Compare invalid_api_hits, doc-before-call behavior, and average turns for first valid exploration.",
            "A non-zero TGC_3 on dev indicates transfer beyond the difficulty 1/2 SFT scope.",
            "Scenario-disjoint SFT validation and dev-only evaluation reduce direct task memorization leakage.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    labels = list(report["models"])
    lines = ["# AppWorld SFT comparison", "", "| metric | " + " | ".join(labels) + " |"]
    lines.append("|---|" + "---|" * len(labels))
    for metric in METRICS:
        lines.append(
            f"| {metric} | "
            + " | ".join(str(report["models"][label].get(metric)) for label in labels)
            + " |"
        )
    lines.extend(["", "## Remaining failure types", ""])
    for label, counts in report["failed_task_types"].items():
        lines.append(f"- {label}: {counts}")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("checkpoint_name", *METRICS))
        writer.writeheader()
        for label, metrics in report["models"].items():
            writer.writerow({"checkpoint_name": label, **metrics})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--training-metrics",
        action="append",
        default=[],
        metavar="CHECKPOINT=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = {
        name: Path(path) for item in args.training_metrics for name, path in [item.split("=", 1)]
    }
    print(json.dumps(summarize(args.input, args.output_dir, metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
