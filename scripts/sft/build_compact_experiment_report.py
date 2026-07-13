from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _epoch_eval_loss(run_dir: Path, checkpoint: int) -> float | None:
    state = _load(run_dir / f"checkpoint-{checkpoint}" / "trainer_state.json")
    matches = [
        entry.get("eval_loss")
        for entry in state.get("log_history", [])
        if entry.get("step") == checkpoint and entry.get("eval_loss") is not None
    ]
    return matches[-1] if matches else None


def _run_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted((root / "runs").glob("*/train_metrics.json")):
        run_dir = metrics_path.parent
        metrics = _load(metrics_path)
        artifact = _load(run_dir / "sft_artifact.json")
        config = artifact.get("sft_config") or {}
        epoch_steps = []
        for epoch in (1, 2):
            link = run_dir / f"epoch-{epoch}"
            if link.exists():
                epoch_steps.append(int(link.resolve().name.removeprefix("checkpoint-")))
            else:
                epoch_steps.append(None)
        epoch_checkpoints = {
            f"checkpoint_epoch_{epoch}": (
                str((run_dir / f"epoch-{epoch}").resolve())
                if (run_dir / f"epoch-{epoch}").exists()
                else None
            )
            for epoch in (1, 2)
        }
        rows.append(
            {
                "record_type": "training",
                "name": run_dir.name,
                "samples": metrics.get("train_samples"),
                "validation_samples": metrics.get("validation_samples"),
                "train_windows": metrics.get("train_windows"),
                "validation_windows": metrics.get("validation_windows"),
                "supervised_tokens": metrics.get("effective_supervised_token_exposures"),
                "optimizer_steps": metrics.get("optimizer_steps"),
                "train_loss": metrics.get("train_loss"),
                "validation_loss": metrics.get("eval_loss"),
                "epoch_1_validation_loss": (
                    _epoch_eval_loss(run_dir, epoch_steps[0])
                    if epoch_steps[0] is not None
                    else None
                ),
                "epoch_2_validation_loss": (
                    _epoch_eval_loss(run_dir, epoch_steps[1])
                    if epoch_steps[1] is not None
                    else None
                ),
                "runtime_seconds": metrics.get("train_runtime"),
                "peak_gpu_memory_bytes": metrics.get("peak_gpu_memory_bytes"),
                "attention_backend": metrics.get("attention_backend"),
                "data_sha256": artifact.get("data_sha256"),
                "train_data_sha256": artifact.get("train_data_sha256"),
                "validation_data_sha256": artifact.get("validation_data_sha256"),
                "initial_adapter_sha256": artifact.get("initial_adapter_sha256"),
                "learning_rate": config.get("learning_rate"),
                "epochs": config.get("epochs"),
                "batch_size": config.get("per_device_train_batch_size"),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
                "lora_rank": config.get("lora_rank"),
                "lora_alpha": config.get("lora_alpha"),
                "lora_dropout": config.get("lora_dropout"),
                "max_length": config.get("max_length"),
                "seed": config.get("seed"),
                "lora_target_modules": ",".join(artifact.get("lora_target_modules") or []),
                **epoch_checkpoints,
            }
        )
    return rows


def _decision_rows(
    root: Path, evaluation_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row.get("split"), row.get("name")): row for row in evaluation_rows}
    d12_comparison = _load(root / "evaluation" / "d12_comparison" / "comparison.json")
    d3_comparison = _load(root / "evaluation" / "d3_comparison" / "comparison.json")
    best_d12 = d12_comparison.get("recommended_checkpoint")
    best_d3 = d3_comparison.get("recommended_checkpoint")
    if not best_d12 or not best_d3:
        return [], {
            "best_d12": best_d12,
            "best_d3": best_d3,
            "d3_improved": False,
            "d12_strict_drop_exceeds_two_points": False,
            "replay_required": False,
            "replay_run": False,
            "d3_transfer": {},
            "d12_forgetting": {},
        }
    d12_before = by_key.get(("sft_d12_validation_20260713", best_d12), {})
    d12_after = by_key.get(("sft_d12_validation_20260713", f"{best_d3}_on_d12"), {})
    d3_before = by_key.get(("sft_d3_validation_20260713", f"{best_d12}_before_d3"), {})
    d3_after = by_key.get(("sft_d3_validation_20260713", best_d3), {})

    def delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float | None:
        if after.get(key) is None or before.get(key) is None:
            return None
        return after[key] - before[key]

    d3_strict_delta = delta(d3_after, d3_before, "TGC")
    d3_partial_delta = delta(d3_after, d3_before, "partial_pass")
    d12_strict_delta = delta(d12_after, d12_before, "TGC")
    d12_partial_delta = delta(d12_after, d12_before, "partial_pass")
    d3_improved = bool(
        d3_strict_delta is not None
        and d3_partial_delta is not None
        and (d3_strict_delta > 0 or (d3_strict_delta == 0 and d3_partial_delta > 0))
    )
    d12_drop_exceeds_two_points = bool(d12_strict_delta is not None and d12_strict_delta < -2.0)
    replay_required = d3_improved and d12_drop_exceeds_two_points
    rows = [
        {
            "record_type": "comparison",
            "name": "d3_transfer",
            "before_checkpoint": best_d12,
            "after_checkpoint": best_d3,
            "TGC_delta_points": d3_strict_delta,
            "partial_pass_delta": d3_partial_delta,
            "execution_failed_delta": delta(d3_after, d3_before, "execution_failed"),
            "no_code_delta": delta(d3_after, d3_before, "no_code"),
        },
        {
            "record_type": "comparison",
            "name": "d12_forgetting",
            "before_checkpoint": best_d12,
            "after_checkpoint": best_d3,
            "TGC_delta_points": d12_strict_delta,
            "partial_pass_delta": d12_partial_delta,
            "execution_failed_delta": delta(d12_after, d12_before, "execution_failed"),
            "no_code_delta": delta(d12_after, d12_before, "no_code"),
        },
    ]
    decision = {
        "best_d12": best_d12,
        "best_d3": best_d3,
        "d3_improved": d3_improved,
        "d12_strict_drop_exceeds_two_points": d12_drop_exceeds_two_points,
        "replay_required": replay_required,
        "replay_run": False,
        "d3_transfer": rows[0],
        "d12_forgetting": rows[1],
    }
    return rows, decision


def _evaluation_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "evaluation").glob("**/behavior.json")):
        value = _load(path)
        rows.append(
            {
                "record_type": "evaluation",
                "name": value.get("checkpoint_name", path.parent.name),
                "split": path.parent.parent.name,
                "TGC": value.get("TGC"),
                "SGC": value.get("SGC"),
                "partial_pass": value.get("average_partial_pass_rate"),
                "execution_failed": value.get("execution_failed_count"),
                "no_code": value.get("no_code_found_count"),
                "invalid_api": value.get("invalid_api_hits"),
                "api_doc_calls_per_rollout": value.get("api_doc_calls_per_rollout"),
                "average_turns": value.get("num_turns_avg"),
                "error_recovery_rate": value.get("error_recovery_success_rate"),
                "context_truncation_ratio": value.get("context_truncation_ratio"),
                "TGC_1": value.get("TGC_1"),
                "TGC_2": value.get("TGC_2"),
                "TGC_3": value.get("TGC_3"),
                "source": str(path.resolve()),
            }
        )
    return rows


def _data_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name, directory, audit_directory in (
        ("d12_supervision_v2", "d12_supervision_v2", "d12_full"),
        ("d3_supervision_v2", "d3_supervision_v2", "d3_full"),
    ):
        manifest = _load(root / "data" / directory / "data_manifest.json")
        audit = _load(root / "audits" / audit_directory / "sft_artifact.json").get("data_stats", {})
        rows.append(
            {
                "record_type": "dataset",
                "name": name,
                "samples": manifest.get("train_samples"),
                "validation_samples": manifest.get("validation_samples"),
                "supervised_steps": manifest.get("supervised_steps"),
                "supervised_tokens": audit.get("assistant_tokens"),
                "train_windows": audit.get("train_windows"),
                "validation_windows": audit.get("validation_windows"),
                "masked_failed_steps": (manifest.get("masked_step_reasons") or {}).get(
                    "execution_failed", 0
                ),
                "clean_success": (manifest.get("success_types") or {}).get("clean_success", 0),
                "recovered_success": (manifest.get("success_types") or {}).get(
                    "recovered_success", 0
                ),
                "data_version": manifest.get("data_version"),
                "scenario_overlap": 0,
            }
        )
    nested = _load(root / "data" / "d12_nested" / "subset_manifest.json")
    for name, stats in sorted((nested.get("subsets") or {}).items()):
        rows.append(
            {
                "record_type": "subset",
                "name": name,
                "samples": stats.get("samples"),
                "validation_samples": nested.get("validation_samples"),
                "supervised_steps": stats.get("supervised_steps"),
                "supervised_tokens": stats.get("effective_supervised_tokens"),
                "clean_success": (stats.get("success_types") or {}).get("clean_success", 0),
                "recovered_success": (stats.get("success_types") or {}).get("recovered_success", 0),
                "scenarios": stats.get("scenarios"),
                "tasks": stats.get("tasks"),
                "data_sha256": stats.get("train_sha256"),
                "scenario_overlap": len(nested.get("scenario_overlap") or []),
            }
        )
    return rows


def _markdown(rows: list[dict[str, Any]], decisions: dict[str, Any]) -> str:
    def format_delta(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+g}"

    lines = ["# Compact AppWorld LoRA SFT — 2026-07-13", ""]
    data = [row for row in rows if row["record_type"] in {"dataset", "subset"}]
    lines.extend(
        [
            "## Data and supervision audit",
            "",
            "| name | train | validation | supervised steps | supervised tokens | clean | recovered | scenario overlap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data:
        lines.append(
            "| {name} | {samples} | {validation_samples} | {supervised_steps} | "
            "{supervised_tokens} | {clean_success} | {recovered_success} | "
            "{scenario_overlap} |".format(
                **{
                    key: row.get(key, "")
                    for key in (
                        "name",
                        "samples",
                        "validation_samples",
                        "supervised_steps",
                        "supervised_tokens",
                        "clean_success",
                        "recovered_success",
                        "scenario_overlap",
                    )
                }
            )
        )
    training = [row for row in rows if row["record_type"] == "training"]
    lines.extend(
        [
            "",
            "## Training",
            "",
            "| run | windows | optimizer steps | supervised token exposures | train loss | validation loss | seconds | peak bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in training:
        lines.append(
            "| {name} | {train_windows} | {optimizer_steps} | {supervised_tokens} | "
            "{train_loss} | {validation_loss} | {runtime_seconds} | {peak_gpu_memory_bytes} |".format(
                **{
                    key: row.get(key, "")
                    for key in (
                        "name",
                        "train_windows",
                        "optimizer_steps",
                        "supervised_tokens",
                        "train_loss",
                        "validation_loss",
                        "runtime_seconds",
                        "peak_gpu_memory_bytes",
                    )
                }
            )
        )
    evaluation = [row for row in rows if row["record_type"] == "evaluation"]
    lines.extend(
        [
            "",
            "## AppWorld evaluation",
            "",
            "| split | checkpoint | TGC | SGC | partial pass | execution failed | no-code | invalid API | doc calls/rollout | turns | recovery | truncation |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evaluation:
        lines.append(
            "| {split} | {name} | {TGC} | {SGC} | {partial_pass} | "
            "{execution_failed} | {no_code} | {invalid_api} | "
            "{api_doc_calls_per_rollout} | {average_turns} | {error_recovery_rate} | "
            "{context_truncation_ratio} |".format(
                **{
                    key: row.get(key, "")
                    for key in (
                        "split",
                        "name",
                        "TGC",
                        "SGC",
                        "partial_pass",
                        "execution_failed",
                        "no_code",
                        "invalid_api",
                        "api_doc_calls_per_rollout",
                        "average_turns",
                        "error_recovery_rate",
                        "context_truncation_ratio",
                    )
                }
            )
        )
    lines.extend(
        [
            "",
            "## Selection, transfer, and replay decision",
            "",
            f"- Best D12: `{decisions.get('best_d12')}`.",
            f"- Best D12→D3: `{decisions.get('best_d3')}`.",
            f"- D3 transfer: TGC {format_delta(decisions.get('d3_transfer', {}).get('TGC_delta_points'))} points, "
            f"partial pass {format_delta(decisions.get('d3_transfer', {}).get('partial_pass_delta'))}.",
            f"- D1/2 forgetting: TGC {format_delta(decisions.get('d12_forgetting', {}).get('TGC_delta_points'))} points, "
            f"partial pass {format_delta(decisions.get('d12_forgetting', {}).get('partial_pass_delta'))}.",
            f"- Replay required: `{decisions.get('replay_required')}`; replay run: "
            f"`{decisions.get('replay_run')}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, output_dir: Path) -> dict[str, Any]:
    evaluation_rows = _evaluation_rows(root)
    decision_rows, decisions = _decision_rows(root, evaluation_rows)
    rows = [*_data_rows(root), *_run_rows(root), *evaluation_rows, *decision_rows]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(root.resolve()),
        "decisions": decisions,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compact_lora_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "compact_lora_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "compact_lora_report.md").write_text(_markdown(rows, decisions))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/appworld_sft/compact_lora_20260713")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "reports"
    print(json.dumps(build(args.root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
