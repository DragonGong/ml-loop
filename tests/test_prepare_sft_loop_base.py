from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

import pytest
import torch

from scripts.loop7b.prepare_sft_loop_base import (
    MANIFEST_NAME,
    ProbeSnapshot,
    build_manifest,
    build_parser,
    compare_probe_snapshots,
    evaluate_verification,
    hash_weight_files,
    validate_args,
    validate_paths,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(logits: list[list[float]], generated: list[tuple[int, ...]]) -> ProbeSnapshot:
    return ProbeSnapshot(
        prompt_hashes=tuple(f"hash-{index}" for index in range(len(logits))),
        next_token_logits=tuple(torch.tensor(row) for row in logits),
        generated_token_ids=tuple(generated),
    )


def test_parser_uses_safe_compact_loop_defaults() -> None:
    args = build_parser().parse_args(
        ["--base", "base", "--adapter", "adapter", "--output", "merged"]
    )
    validate_args(args)

    assert args.dtype == "bfloat16"
    assert args.device == "cuda:0"
    assert args.safe_merge is True
    assert args.probe_max_new_tokens == 16
    assert args.min_merge_cosine == pytest.approx(0.998)
    assert args.seed == 20260713


@pytest.mark.parametrize("value", (0, -1))
def test_validate_args_rejects_invalid_probe_length(value: int) -> None:
    args = argparse.Namespace(
        probe_max_new_tokens=value,
        min_merge_cosine=0.999,
        max_shard_size="5GB",
        safe_merge=True,
    )
    with pytest.raises(ValueError, match="positive"):
        validate_args(args)


def test_validate_paths_refuses_missing_inputs_and_nonempty_output(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()

    with pytest.raises(ValueError, match="missing"):
        validate_paths(base, adapter, output)

    (base / "config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(ValueError, match="cannot be"):
        validate_paths(base, adapter, base / "merged")

    output.mkdir()
    (output / "existing").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        validate_paths(base, adapter, output)


def test_probe_comparison_and_verification_distinguish_merge_from_exact_checks() -> None:
    premerge = _snapshot([[1.0, 3.0, -1.0]], [(10, 11)])
    merged = _snapshot([[1.001, 3.001, -0.999]], [(10, 12)])
    exact_reload = _snapshot([[1.001, 3.001, -0.999]], [(10, 12)])

    merge_comparison = compare_probe_snapshots(premerge, merged)
    reload_comparison = compare_probe_snapshots(merged, exact_reload)
    zero_lora_comparison = compare_probe_snapshots(exact_reload, exact_reload)
    result = evaluate_verification(
        merge_comparison,
        reload_comparison,
        zero_lora_comparison,
        min_merge_cosine=0.998,
        zero_lora_b_tensors=14,
        nonzero_lora_b_values=0,
    )

    assert not merge_comparison["exact_logits"]
    assert merge_comparison["next_token_equal"]
    assert not merge_comparison["generation_equal"]
    assert result["passed"]
    assert result["checks"]["merged_to_reloaded"]["exact_logits"]

    failed = evaluate_verification(
        merge_comparison,
        reload_comparison,
        zero_lora_comparison,
        min_merge_cosine=0.999,
        zero_lora_b_tensors=14,
        nonzero_lora_b_values=1,
    )
    assert not failed["passed"]


def test_weight_hash_and_manifest_are_complete(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "merged"
    for directory in (base, adapter, output):
        directory.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (adapter / "adapter_config.json").write_text('{"r":32}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (output / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (output / "model-00001-of-00002.safetensors").write_bytes(b"first")

    weights = hash_weight_files(output)
    verification = {"passed": True, "checks": {}}
    manifest = build_manifest(
        base=base,
        adapter=adapter,
        output=output,
        dtype="bfloat16",
        max_shard_size="5GB",
        max_new_tokens=16,
        min_merge_cosine=0.999,
        adapter_config={
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": ["v_proj", "q_proj"],
        },
        source_adapter_sha256="source-hash",
        output_weights=weights,
        verification=verification,
        package_versions={"torch": "test"},
    )
    write_manifest(output / MANIFEST_NAME, manifest)

    parsed = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert parsed["status"] == "verified"
    assert parsed["merged_model"]["safe_merge"] is True
    assert parsed["merged_model"]["tokenizer_saved_and_reloaded"] is True
    assert parsed["loop_adapter_initialization"]["r"] == 16
    assert parsed["loop_adapter_initialization"]["lora_alpha"] == 32
    assert parsed["sft_adapter"]["target_modules"] == ["q_proj", "v_proj"]
    assert [item["path"] for item in weights["files"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert len(weights["combined_sha256"]) == 64
