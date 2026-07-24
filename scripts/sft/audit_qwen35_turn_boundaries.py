from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from phi_agents.sft.trainer import SFTConfig, prepare_windows


def _python_code_block_count(text: str) -> int:
    return len(re.findall(r"```(?:python|py)\s*(?:\r?\n)", text, flags=re.IGNORECASE))


def supervised_spans(labels: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label != -100 and start is None:
            start = index
        elif label == -100 and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def audit_rows(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    special_tokens = {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
    }
    totals: Counter[str] = Counter()
    next_token_counts: Counter[str] = Counter()
    totals["supervised_spans_preceded_by_empty_think"] = 0
    empty_think_ids = list(
        tokenizer("<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    )
    for row in rows:
        input_ids = list(row["input_ids"])
        labels = list(row["labels"])
        if len(input_ids) != len(labels):
            raise ValueError("input_ids and labels differ in length")
        spans = supervised_spans(labels)
        totals["windows"] += 1
        totals["tokens"] += len(input_ids)
        totals["supervised_tokens"] += sum(label != -100 for label in labels)
        totals["supervised_spans"] += len(spans)
        totals["windows_with_multiple_supervised_spans"] += len(spans) > 1

        supervised_concat = tokenizer.decode(
            [token_id for token_id, label in zip(input_ids, labels, strict=True) if label != -100],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        totals["windows_whose_concatenated_labels_have_multiple_python_blocks"] += (
            _python_code_block_count(supervised_concat) > 1
        )
        for start, end in spans:
            if input_ids[max(0, start - len(empty_think_ids)) : start] == empty_think_ids:
                totals["supervised_spans_preceded_by_empty_think"] += 1
            span_text = tokenizer.decode(
                input_ids[start:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            totals["supervised_spans_with_multiple_python_blocks"] += (
                _python_code_block_count(span_text) > 1
            )
            if end >= len(input_ids):
                next_token_counts["end_of_window"] += 1
                continue
            next_token = input_ids[end]
            next_token_text = tokenizer.decode(
                [next_token],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            next_token_counts[next_token_text] += 1
            if next_token == special_tokens["<|im_end|>"]:
                totals["spans_followed_by_im_end"] += 1
                totals["spans_followed_by_masked_im_end"] += labels[end] == -100

        for token, token_id in special_tokens.items():
            positions = [index for index, value in enumerate(input_ids) if value == token_id]
            totals[f"{token}_input_count"] += len(positions)
            totals[f"{token}_supervised_count"] += sum(labels[index] != -100 for index in positions)

        messages = list(row.get("messages") or [])
        target_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("loss") is True
        ]
        totals["supervised_assistant_messages"] += len(target_indices)
        for index in target_indices:
            totals["supervised_assistant_messages_with_multiple_python_blocks"] += (
                _python_code_block_count(str(messages[index].get("content") or "")) > 1
            )
        for left, right in zip(target_indices, target_indices[1:], strict=False):
            intervening = messages[left + 1 : right]
            has_observation = any(
                message.get("message_type") == "appworld_observation"
                or message.get("role") in {"user", "tool", "ipython"}
                for message in intervening
            )
            totals["target_pairs_without_intervening_observation"] += not has_observation

    return {
        **dict(totals),
        "next_token_after_supervised_span": dict(next_token_counts),
        "special_token_ids": special_tokens,
        "empty_think_token_ids": empty_think_ids,
    }


def _load_stage(
    *,
    name: str,
    data_dir: Path,
    model_path: Path,
    tokenizer: Any,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = SFTConfig(
        train_jsonl=data_dir / "qwen_sft_train.jsonl",
        validation_jsonl=data_dir / "qwen_sft_validation.jsonl",
        output_dir=output_dir / "_unused" / name,
        model_name="Qwen/Qwen3.5-4B",
        model_path=model_path,
        max_length=16_384,
        epochs=1 if name == "d12" else 2,
        gradient_accumulation_steps=4 if name == "d12" else 2,
        sparse_assistant_logits=True,
        attention_backend="sdpa",
    )
    train_rows, validation_rows, preparation_stats = prepare_windows(config, tokenizer)
    tagged_rows = [
        {**row, "_stage": name, "_split": "train", "_row_index": index}
        for index, row in enumerate(train_rows)
    ]
    tagged_rows.extend(
        {**row, "_stage": name, "_split": "validation", "_row_index": index}
        for index, row in enumerate(validation_rows)
    )
    return tagged_rows, {
        "name": name,
        "data_dir": str(data_dir.resolve()),
        "train_windows": len(train_rows),
        "validation_windows": len(validation_rows),
        "preparation_stats": preparation_stats,
        "boundary_stats": audit_rows(tagged_rows, tokenizer),
    }


def _stratified_sample(
    stage_rows: dict[str, list[dict[str, Any]]], sample_count: int, seed: int
) -> list[dict[str, Any]]:
    if sample_count < len(stage_rows):
        raise ValueError(f"sample_count must be at least {len(stage_rows)}")
    total = sum(len(rows) for rows in stage_rows.values())
    if sample_count > total:
        raise ValueError(f"Requested {sample_count} samples from only {total} rows")
    rng = random.Random(seed)
    allocations = {
        name: max(1, round(sample_count * len(rows) / total)) for name, rows in stage_rows.items()
    }
    while sum(allocations.values()) > sample_count:
        name = max(allocations, key=lambda key: (allocations[key], len(stage_rows[key])))
        if allocations[name] <= 1:
            raise RuntimeError("Could not reduce stratified allocation")
        allocations[name] -= 1
    while sum(allocations.values()) < sample_count:
        candidates = [name for name, rows in stage_rows.items() if allocations[name] < len(rows)]
        name = max(candidates, key=lambda key: len(stage_rows[key]) - allocations[key])
        allocations[name] += 1
    selected: list[dict[str, Any]] = []
    for name in sorted(stage_rows):
        selected.extend(rng.sample(stage_rows[name], allocations[name]))
    rng.shuffle(selected)
    return selected


def _message_boundaries(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        boundaries.append(
            {
                "index": index,
                "role": message.get("role"),
                "loss": message.get("loss") is True,
                "message_type": message.get("message_type"),
                "step_id": message.get("step_id"),
                "window_role": message.get("window_role"),
                "characters": len(content),
                "python_code_blocks": _python_code_block_count(content),
                "preview": content[:160],
            }
        )
    return boundaries


def _write_decoded_sample(
    *,
    path: Path,
    row: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    input_ids = list(row["input_ids"])
    labels = list(row["labels"])
    spans = supervised_spans(labels)
    full_text = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    supervised_concat = tokenizer.decode(
        [token_id for token_id, label in zip(input_ids, labels, strict=True) if label != -100],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded_spans = [
        {
            "start": start,
            "end": end,
            "tokens": end - start,
            "text": tokenizer.decode(
                input_ids[start:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "next_token": (
                tokenizer.decode(
                    [input_ids[end]],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if end < len(input_ids)
                else None
            ),
            "next_token_label": labels[end] if end < len(labels) else None,
        }
        for start, end in spans
    ]
    boundaries = _message_boundaries(list(row.get("messages") or []))
    metadata = {
        "stage": row["_stage"],
        "split": row["_split"],
        "row_index": row["_row_index"],
        "window_index": row.get("window_index"),
        "window_start_turn": row.get("window_start_turn"),
        "window_end_turn": row.get("window_end_turn"),
        "source_turn_count": row.get("source_turn_count"),
        "tokens": len(input_ids),
        "supervised_tokens": sum(label != -100 for label in labels),
        "supervised_spans": len(spans),
        "task_id": (row.get("metadata") or {}).get("task_id"),
        "trajectory_id": (row.get("metadata") or {}).get("trajectory_id"),
        "file": str(path.resolve()),
    }
    sections = [
        "=== METADATA ===",
        json.dumps(metadata, indent=2, sort_keys=True),
        "",
        "=== MESSAGE BOUNDARIES ===",
        json.dumps(boundaries, indent=2, sort_keys=True),
        "",
        "=== tokenizer.decode(input_ids) ===",
        full_text,
        "",
        "=== tokenizer.decode(input_ids[labels != -100]) ===",
        supervised_concat,
        "",
        "=== CONTIGUOUS SUPERVISED SPANS ===",
        json.dumps(decoded_spans, indent=2, sort_keys=True),
        "",
    ]
    path.write_text("\n".join(sections))
    return metadata


def _write_report(
    output: Path, stages: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> None:
    lines = [
        "# Qwen3.5 SFT Turn-Boundary Audit",
        "",
        "This audit runs the exact training preprocessing path over all D1/2 and D3 rows. "
        "Ten deterministic stratified samples are decoded in full, both as `input_ids` and as "
        "`input_ids[labels != -100]`.",
        "",
        "## Aggregate",
        "",
        "| Stage | Windows | Target messages | Supervised spans | Target multi-block | "
        "No observation between targets | `<|im_end|>` supervised | "
        "Spans followed by masked `<|im_end|>` | Spans preceded by empty think |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in stages:
        stats = stage["boundary_stats"]
        lines.append(
            f"| {stage['name']} | {stats['windows']} | "
            f"{stats['supervised_assistant_messages']} | {stats['supervised_spans']} | "
            f"{stats.get('supervised_assistant_messages_with_multiple_python_blocks', 0)} | "
            f"{stats.get('target_pairs_without_intervening_observation', 0)} | "
            f"{stats.get('<|im_end|>_supervised_count', 0)}/"
            f"{stats.get('<|im_end|>_input_count', 0)} | "
            f"{stats.get('spans_followed_by_masked_im_end', 0)}/"
            f"{stats['supervised_spans']} | "
            f"{stats.get('supervised_spans_preceded_by_empty_think', 0)}/"
            f"{stats['supervised_spans']} |"
        )
    lines.extend(
        [
            "",
            "A window can contain multiple correctly separated target turns; therefore the "
            "concatenated supervised decode may contain several code blocks. The decisive checks "
            "are the per-message/per-span rows and whether the turn-ending token is supervised.",
            "",
            "## Decoded Samples",
            "",
            "| Stage | Split | Row | Task | Tokens | Supervised tokens | Spans | File |",
            "|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for sample in samples:
        file_name = Path(sample["file"]).name
        lines.append(
            f"| {sample['stage']} | {sample['split']} | {sample['row_index']} | "
            f"{sample['task_id']} | {sample['tokens']} | {sample['supervised_tokens']} | "
            f"{sample['supervised_spans']} | `{file_name}` |"
        )
    lines.append("")
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--d12-data-dir", type=Path, required=True)
    parser.add_argument("--d3-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2_026_072_3)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir = args.output_dir / "decoded_samples"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    stage_rows: dict[str, list[dict[str, Any]]] = {}
    stage_reports: list[dict[str, Any]] = []
    for name, data_dir in (("d12", args.d12_data_dir), ("d3", args.d3_data_dir)):
        rows, report = _load_stage(
            name=name,
            data_dir=data_dir,
            model_path=args.model_path,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
        )
        stage_rows[name] = rows
        stage_reports.append(report)

    selected = _stratified_sample(stage_rows, args.sample_count, args.seed)
    selected_metadata: list[dict[str, Any]] = []
    for ordinal, row in enumerate(selected):
        file_name = f"{ordinal:02d}_{row['_stage']}_{row['_split']}_{row['_row_index']:04d}.txt"
        selected_metadata.append(
            _write_decoded_sample(
                path=decoded_dir / file_name,
                row=row,
                tokenizer=tokenizer,
            )
        )

    result = {
        "status": "passed",
        "model_path": str(args.model_path.resolve()),
        "max_length": 16_384,
        "seed": args.seed,
        "sample_count": args.sample_count,
        "stages": stage_reports,
        "decoded_samples": selected_metadata,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_report(args.output_dir / "report.md", stage_reports, selected_metadata)
    print(
        json.dumps(
            {
                "status": "passed",
                "stages": {stage["name"]: stage["boundary_stats"] for stage in stage_reports},
                "decoded_samples": len(selected_metadata),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
