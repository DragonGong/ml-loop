from __future__ import annotations

import argparse
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
)


def summarize(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in paths]
    report = {
        "models": {
            row["checkpoint_name"]: {key: row.get(key) for key in METRICS} for row in rows
        },
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
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
