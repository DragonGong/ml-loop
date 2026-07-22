from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METRICS = (
    "SGC",
    "TGC",
    "SGC_1",
    "SGC_2",
    "SGC_3",
    "TGC_1",
    "TGC_2",
    "TGC_3",
    "average_partial_pass_rate",
    "execution_failed_count",
    "execution_errors_per_turn",
    "invalid_api_hits",
    "api_doc_calls_per_rollout",
    "api_description_calls_per_rollout",
    "doc_before_api_call_rate",
    "error_recovery_success_rate",
    "error_rollout_recovery_success_rate",
    "context_truncation_ratio",
    "episode_count",
)


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else 0.0


def _load_eval(root: Path, label: str) -> dict[str, Any]:
    path = root / "evaluation" / label / "summary.json"
    rows = json.loads(path.read_text())
    matches = [row for row in rows if row.get("checkpoint_name") == label]
    if len(matches) != 1:
        raise ValueError(f"Expected one {label} row in {path}, found {len(matches)}")
    row = matches[0]
    if row.get("split") != "dev" or row.get("episode_count") != 57:
        raise ValueError(f"Incomplete {label} dev result: {row.get('episode_count')}")
    return row


def _failure_count(root: Path, label: str) -> int:
    path = root / "evaluation" / label / "eval_failures.jsonl"
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text().splitlines())


def _base_anchor(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    rows = json.loads(path.read_text())
    base = next((row for row in rows if row.get("checkpoint_name") == "base"), None)
    if base is None:
        return None
    return {key: base.get(key) for key in ("SGC", "TGC", "average_partial_pass_rate")}


def summarize(root: Path, base_summary: Path | None) -> dict[str, Any]:
    d12_eval = _load_eval(root, "d12")
    d3_eval = _load_eval(root, "d3")
    d12_train = json.loads((root / "training" / "d12" / "train_metrics.json").read_text())
    d3_train = json.loads((root / "training" / "d3" / "train_metrics.json").read_text())
    d3_selection = json.loads((root / "training" / "d3" / "selected_adapter.json").read_text())
    d3_epoch = int(d3_selection["selected_epoch"])
    d3_epoch_row = next(
        row for row in d3_selection["epochs"] if int(row["epoch"]) == d3_epoch
    )
    d12_sha_path = root / "training" / "d12" / "final_adapter" / "adapter_model.sha256"
    d12_sha = d12_sha_path.read_text().split()[0]

    rows = []
    for label, eval_row, validation_loss, adapter_sha, train_metrics in (
        ("d12", d12_eval, d12_train.get("eval_loss"), d12_sha, d12_train),
        (
            "d3",
            d3_eval,
            d3_epoch_row["validation_loss"],
            d3_selection["selected_adapter_sha256"],
            d3_train,
        ),
    ):
        rows.append(
            {
                "checkpoint_name": label,
                "selected_epoch": 1 if label == "d12" else d3_epoch,
                "adapter_sha256": adapter_sha,
                "validation_loss": validation_loss,
                "train_loss": train_metrics.get("train_loss"),
                "optimizer_steps": train_metrics.get("optimizer_steps"),
                "train_runtime_seconds": train_metrics.get("train_runtime"),
                "eval_failure_records": _failure_count(root, label),
                **{metric: eval_row.get(metric) for metric in METRICS},
            }
        )

    recommended = max(
        rows,
        key=lambda row: (
            _float(row, "TGC"),
            _float(row, "SGC"),
            _float(row, "average_partial_pass_rate"),
        ),
    )
    d12, d3 = rows
    deltas = {
        metric: _float(d3, metric) - _float(d12, metric)
        for metric in ("TGC", "SGC", "average_partial_pass_rate")
    }
    report = {
        "status": "complete",
        "selection_priority": ["TGC", "SGC", "average_partial_pass_rate"],
        "recommended_adapter": recommended["checkpoint_name"],
        "d3_selected_epoch": d3_epoch,
        "d3_minus_d12": deltas,
        "d3_damaged_d12": any(deltas[key] < 0 for key in ("TGC", "SGC")),
        "base_16k_anchor": _base_anchor(base_summary),
        "base_anchor_note": (
            "Base was evaluated on a different machine and was not rerun as a paired control."
        ),
        "models": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    fieldnames = list(rows[0])
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Qwen3.5-4B two-stage AppWorld SFT",
        "",
        "Both adapters were evaluated on all 57 AppWorld dev episodes at 16K context.",
        "The Base 16K result is a descriptive anchor from another machine, not a paired rerun.",
        "",
        "| adapter | selected epoch | SGC | TGC | partial pass | SGC D1/D2/D3 | TGC D1/D2/D3 | eval failures |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint_name']} | {row['selected_epoch']} | {row['SGC']} | "
            f"{row['TGC']} | {row['average_partial_pass_rate']} | "
            f"{row['SGC_1']}/{row['SGC_2']}/{row['SGC_3']} | "
            f"{row['TGC_1']}/{row['TGC_2']}/{row['TGC_3']} | "
            f"{row['eval_failure_records']} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Recommended adapter: `{recommended['checkpoint_name']}` "
            "(priority: TGC, then SGC, then partial pass).",
            f"D3 selected epoch: {d3_epoch}.",
            f"D3 - D1/2: TGC {deltas['TGC']:+.3f}, SGC {deltas['SGC']:+.3f}, "
            f"partial pass {deltas['average_partial_pass_rate']:+.6f}.",
            "",
            "## Operational Metrics",
            "",
            "| adapter | execution failures | errors/turn | invalid API | docs/rollout | "
            "recovery | context truncation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['checkpoint_name']} | {row['execution_failed_count']} | "
            f"{row['execution_errors_per_turn']} | {row['invalid_api_hits']} | "
            f"{row['api_doc_calls_per_rollout']} | {row['error_recovery_success_rate']} | "
            f"{row['context_truncation_ratio']} |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve(), args.base_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
