from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from phi_agents.sft.trainer import tokenize_messages


def _sample(repetitions: int) -> dict[str, Any]:
    step_id = "qwen35-fsdp-smoke:synthetic:0"
    return {
        "messages": [
            {"role": "system", "content": "You are an AppWorld agent.", "loss": False},
            {
                "role": "user",
                "content": "Exercise the full context window. " + "context " * repetitions,
                "loss": False,
            },
            {
                "role": "assistant",
                "content": "Code:\n```python\napis.supervisor.complete_task()\n```",
                "loss": True,
                "message_type": "appworld_action",
                "step_id": step_id,
                "step_index": 0,
                "mask_reason": None,
            },
        ],
        "metadata": {
            "task_id": "qwen35-fsdp-smoke",
            "trajectory_id": "synthetic",
            "split": "synthetic_smoke",
        },
    }


def build(model_path: Path, output_dir: Path, max_length: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    low, high = 1, max_length * 4
    best_sample: dict[str, Any] | None = None
    best_tokens = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = _sample(middle)
        token_count = len(tokenize_messages(tokenizer, candidate["messages"])["input_ids"])
        if token_count <= max_length:
            best_sample = candidate
            best_tokens = token_count
            low = middle + 1
        else:
            high = middle - 1
    if best_sample is None or best_tokens < max_length - 16:
        raise RuntimeError(
            f"Could not construct a near-16K smoke sample: tokens={best_tokens} max={max_length}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    train_path.write_text(json.dumps(best_sample, sort_keys=True) + "\n")
    validation_path.write_text("")
    result = {
        "train_jsonl": str(train_path.resolve()),
        "validation_jsonl": str(validation_path.resolve()),
        "sequence_tokens": best_tokens,
        "max_length": max_length,
    }
    (output_dir / "smoke_data.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=16_384)
    args = parser.parse_args()
    print(json.dumps(build(args.model_path, args.output_dir, args.max_length), sort_keys=True))


if __name__ == "__main__":
    main()
