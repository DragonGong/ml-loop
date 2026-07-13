import csv
import json

from scripts.sft.build_compact_experiment_report import build
from scripts.sft.summarize_comparison import summarize


def test_summary_writes_json_csv_markdown_and_uses_selection_priority(tmp_path) -> None:
    behavior_paths = []
    metrics_paths = {}
    rows = (
        ("d12_25", 0.5, 0.75, 3, 0.40),
        ("d12_50", 0.5, 0.80, 5, 0.35),
        ("d12_100", 0.5, 0.80, 2, 0.45),
    )
    for name, strict, partial, errors, validation_loss in rows:
        behavior_path = tmp_path / f"{name}-behavior.json"
        behavior_path.write_text(
            json.dumps(
                {
                    "checkpoint_name": name,
                    "TGC": strict,
                    "average_partial_pass_rate": partial,
                    "execution_failed_count": errors,
                    "no_code_found_count": 0,
                }
            )
        )
        metrics_path = tmp_path / f"{name}-metrics.json"
        metrics_path.write_text(json.dumps({"eval_loss": validation_loss}))
        behavior_paths.append(behavior_path)
        metrics_paths[name] = metrics_path

    report = summarize(behavior_paths, tmp_path / "summary", metrics_paths)

    assert report["recommended_checkpoint"] == "d12_100"
    assert (tmp_path / "summary/comparison.json").is_file()
    assert (tmp_path / "summary/comparison.md").is_file()
    with (tmp_path / "summary/comparison.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["checkpoint_name"] for row in csv_rows] == [
        "d12_25",
        "d12_50",
        "d12_100",
    ]


def test_compact_report_emits_all_three_formats(tmp_path) -> None:
    root = tmp_path / "experiment"
    for directory in ("d12_supervision_v2", "d3_supervision_v2"):
        path = root / "data" / directory
        path.mkdir(parents=True)
        (path / "data_manifest.json").write_text(
            json.dumps(
                {
                    "train_samples": 1,
                    "validation_samples": 1,
                    "supervised_steps": 2,
                    "data_version": "supervision-v2",
                }
            )
        )
    nested = root / "data/d12_nested"
    nested.mkdir()
    (nested / "subset_manifest.json").write_text(
        json.dumps(
            {
                "validation_samples": 129,
                "scenario_overlap": [],
                "subsets": {
                    "d12_25": {
                        "samples": 127,
                        "supervised_steps": 10,
                        "effective_supervised_tokens": 20,
                        "success_types": {},
                    }
                },
            }
        )
    )

    output = root / "reports"
    report = build(root, output)

    assert len(report["rows"]) == 3
    assert (output / "compact_lora_report.json").is_file()
    assert (output / "compact_lora_report.csv").is_file()
    assert (output / "compact_lora_report.md").is_file()
