from __future__ import annotations

import csv
import json

from scripts.sft.select_qwen35_d3 import select
from scripts.sft.summarize_qwen35_two_stage import summarize


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_d3_selection_uses_validation_loss_and_prefers_earlier_tie(tmp_path) -> None:
    run_dir = tmp_path / "training" / "d3"
    for epoch, loss in ((1, 0.8), (2, 0.7)):
        checkpoint = run_dir / f"checkpoint-{epoch * 10}"
        adapter = checkpoint / "lora"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(f"epoch-{epoch}".encode())
        _write_json(
            checkpoint / "trainer_state.json",
            {"log_history": [{"epoch": epoch, "eval_loss": loss}]},
        )
        _write_json(
            run_dir / f"epoch-{epoch}.json",
            {"epoch": epoch, "global_step": epoch * 10, "checkpoint": str(checkpoint)},
        )

    result = select(run_dir)
    assert result["selected_epoch"] == 2

    epoch_two_state = run_dir / "checkpoint-20" / "trainer_state.json"
    _write_json(epoch_two_state, {"log_history": [{"epoch": 2, "eval_loss": 0.8}]})
    assert select(run_dir)["selected_epoch"] == 1


def test_two_stage_summary_recommends_by_tgc_then_sgc_then_partial(tmp_path) -> None:
    root = tmp_path / "qwen35"
    metrics = {
        "SGC_1": 10.0,
        "SGC_2": 20.0,
        "SGC_3": 0.0,
        "TGC_1": 30.0,
        "TGC_2": 40.0,
        "TGC_3": 0.0,
        "execution_failed_count": 1,
        "execution_errors_per_turn": 0.1,
        "failed_api_calls": 2,
        "consecutive_repeated_failed_action_count": 3,
        "multiple_code_cells_per_turn": 0.0,
        "invalid_api_hits": 0,
        "api_doc_calls_per_rollout": 2.0,
        "api_description_calls_per_rollout": 1.0,
        "api_doc_or_description_calls_per_rollout": 3.0,
        "doc_before_api_call_rate": 0.8,
        "error_recovery_success_rate": 0.5,
        "error_rollout_recovery_success_rate": 0.4,
        "context_truncation_ratio": 0.0,
        "episode_count": 57,
        "num_rollouts_analyzed": 57,
        "split": "dev",
    }
    _write_json(
        root / "evaluation" / "d12" / "summary.json",
        [{"checkpoint_name": "d12", "SGC": 20.0, "TGC": 40.0, "average_partial_pass_rate": 0.6, **metrics}],
    )
    _write_json(
        root / "evaluation" / "d3" / "summary.json",
        [
            {
                "checkpoint_name": "d3",
                "SGC": 20.0,
                "TGC": 41.0,
                "average_partial_pass_rate": 0.5,
                **metrics,
                "execution_failed_count": 2,
            }
        ],
    )
    _write_json(
        root / "training" / "d12" / "train_metrics.json",
        {"eval_loss": 0.9, "train_loss": 1.0, "optimizer_steps": 70},
    )
    d12_adapter = root / "training" / "d12" / "final_adapter"
    d12_adapter.mkdir(parents=True)
    (d12_adapter / "adapter_model.sha256").write_text("a" * 64 + "  adapter\n")
    _write_json(
        root / "training" / "d3" / "train_metrics.json",
        {"train_loss": 0.8, "optimizer_steps": 68},
    )
    _write_json(
        root / "training" / "d3" / "selected_adapter.json",
        {
            "selected_epoch": 2,
            "selected_adapter_sha256": "b" * 64,
            "epochs": [{"epoch": 1, "validation_loss": 0.7}, {"epoch": 2, "validation_loss": 0.6}],
        },
    )
    base_path = tmp_path / "base.json"
    _write_json(base_path, [{"checkpoint_name": "base", "SGC": 21.1, "TGC": 42.1}])

    report = summarize(root, base_path)

    assert report["recommended_adapter"] == "d3"
    assert report["d3_minus_d12"]["TGC"] == 1.0
    assert report["d3_strict_score_damage"] is False
    assert report["d3_operational_regressions"] == ["execution_failed_count"]
    assert report["d3_damaged_d12"] is True
    assert report["recommended_over_base"] is False
    assert report["base_16k_anchor"]["TGC"] == 42.1
    with (root / "summary.csv").open() as handle:
        assert [row["checkpoint_name"] for row in csv.DictReader(handle)] == ["d12", "d3"]
    assert "Recommended adapter: `d3`" in (root / "report.md").read_text()
