#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from phi_agents.appworld.interface import load_task_ids
from phi_agents.sft.dataset import DatasetBuildConfig, assess_trajectory, trajectory_to_sample
from phi_agents.sft.trainer import audit_sample_windows, window_sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run real AppWorld trajectories through SFT windows."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train_difficulty_1_2", "train_difficulty_3"),
        default="train_difficulty_1_2",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        default=Path(".model_cache/Qwen/Qwen2.5-7B-Instruct/tokenizer.json"),
        help="Local Qwen tokenizer.json; avoids initializing CUDA-capable Transformers for a dry-run.",
    )
    parser.add_argument("--max-length", type=int, default=16_384)
    parser.add_argument("--turn-overlap", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--print-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _priority(trajectory: dict[str, Any]) -> tuple[int, int]:
    steps = trajectory.get("steps") or []
    errors = sum(bool(step.get("execution_failed")) for step in steps)
    no_code = sum(bool(step.get("no_code_found")) for step in steps)
    return (int(errors + no_code > 0), len(steps))


class _OfflineQwenTokenizer:
    def __init__(self, tokenizer_json: Path):
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        if tokenize:
            raise ValueError("The SFT dry-run requests rendered text before tokenization")
        pieces: list[str] = []
        start = 0
        if messages and messages[0]["role"] == "system":
            pieces.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n")
            start = 1
        else:
            pieces.append(
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                "You are a helpful assistant.<|im_end|>\n"
            )
        for message in messages[start:]:
            pieces.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")
        return "".join(pieces)

    def __call__(
        self, text: str, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, Any]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": encoding.ids, "offset_mapping": encoding.offsets}


def main() -> None:
    args = _parse_args()

    allowed_task_ids = set(load_task_ids(args.split))
    mode = "difficulty_1_2" if args.split == "train_difficulty_1_2" else "difficulty_3"
    config = DatasetBuildConfig((args.input_root,), Path(), mode=mode)
    candidates: list[tuple[Path, dict[str, Any], Any]] = []
    for path in sorted(args.input_root.rglob("trajectory.json")):
        trajectory = json.loads(path.read_text())
        decision = assess_trajectory(trajectory, allowed_task_ids)
        if decision.accepted:
            candidates.append((path, trajectory, decision))
    random.Random(args.seed).shuffle(candidates)
    candidates.sort(key=lambda row: _priority(row[1]), reverse=True)
    selected = candidates[: args.limit]
    if not selected:
        raise RuntimeError(f"No accepted trajectories found below {args.input_root}")

    tokenizer = _OfflineQwenTokenizer(args.tokenizer_json)
    totals: Counter[str] = Counter()
    all_windows: list[dict[str, Any]] = []
    for path, trajectory, decision in selected:
        sample = trajectory_to_sample(trajectory, path, decision, config)
        windows = window_sample(
            tokenizer,
            sample,
            args.max_length,
            args.turn_overlap,
            preserve_history=True,
        )
        totals.update(audit_sample_windows(sample, windows))
        all_windows.extend(windows)

    summary = {
        "split": args.split,
        "input_root": str(args.input_root.resolve()),
        "selected_trajectories": len(selected),
        "windows": len(all_windows),
        **dict(totals),
    }
    print(json.dumps({"summary": summary}, indent=2, sort_keys=True))
    for window in all_windows[: args.print_windows]:
        history_observations = [
            str(message.get("content") or "")[:240]
            for message in window["messages"]
            if message.get("message_type") == "appworld_observation"
            and str(message.get("window_role") or "").startswith("history")
        ]
        print(
            json.dumps(
                {
                    "task_id": window["metadata"].get("task_id"),
                    "trajectory_id": window["metadata"].get("trajectory_id"),
                    "window_index": window["window_index"],
                    "window_turns": [window["window_start_turn"], window["window_end_turn"]],
                    "input_tokens": len(window["input_ids"]),
                    "effective_label_tokens": sum(label != -100 for label in window["labels"]),
                    "steps": window["step_token_stats"],
                    "history_observations": history_observations,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
