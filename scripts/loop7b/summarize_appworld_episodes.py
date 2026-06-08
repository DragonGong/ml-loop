#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Summarize AppWorld episode logs beyond the official TGC/SGC metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_INVALID_API_PATTERNS = (
    r"invalid api",
    r"api .* not found",
    r"no api named",
    r"unknown api",
    r"attributeerror: .*apis\.",
    r"has no attribute",
)


@dataclass
class EpisodeStats:
    task_id: str
    success: bool
    pass_rate: float
    difficulty: int | None
    num_interactions: int | None
    n_execution_failed: int
    n_no_code_found: int
    cancelled: bool
    invalid_api_hits: int


def _episode_paths(appworld_root: Path, experiment_name: str) -> list[Path]:
    root = appworld_root / "experiments" / "outputs" / experiment_name / "tasks"
    return sorted(root.glob("*/logs/episode.json"))


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for child in value.values():
            values.extend(_text_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(_text_values(child))
        return values
    return []


def _count_invalid_api_hits(episode: dict[str, Any], patterns: list[re.Pattern[str]]) -> int:
    text = "\n".join(_text_values(episode.get("chat_history", []))).lower()
    return sum(len(pattern.findall(text)) for pattern in patterns)


def _load_episode(path: Path, patterns: list[re.Pattern[str]]) -> EpisodeStats:
    episode = json.loads(path.read_text())
    eval_result = episode.get("eval_result") or {}
    num_tests = int(eval_result.get("num_tests") or 0)
    passes = eval_result.get("passes") or []
    task = episode.get("task") or {}
    task_id = str(task.get("task_id") or path.parents[1].name)

    pass_rate = (len(passes) / num_tests) if num_tests else 0.0

    return EpisodeStats(
        task_id=task_id,
        success=bool(eval_result.get("success")),
        pass_rate=pass_rate,
        difficulty=eval_result.get("difficulty"),
        num_interactions=eval_result.get("num_interactions"),
        n_execution_failed=int(episode.get("n_execution_failed") or 0),
        n_no_code_found=int(episode.get("n_no_code_found") or 0),
        cancelled=bool(episode.get("cancelled")),
        invalid_api_hits=_count_invalid_api_hits(episode, patterns),
    )


def _format_rate(numerator: int | float, denominator: int) -> str:
    if denominator == 0:
        return "0.0000"
    return f"{numerator / denominator:.4f}"


def summarize(appworld_root: Path, experiment_name: str, patterns: list[re.Pattern[str]]) -> None:
    paths = _episode_paths(appworld_root, experiment_name)
    stats = [_load_episode(path, patterns) for path in paths]

    n = len(stats)
    n_success = sum(item.success for item in stats)
    pass_rate_sum = sum(item.pass_rate for item in stats)
    execution_failed_total = sum(item.n_execution_failed for item in stats)
    no_code_total = sum(item.n_no_code_found for item in stats)
    invalid_api_total = sum(item.invalid_api_hits for item in stats)
    cancelled_total = sum(item.cancelled for item in stats)
    interactions = [item.num_interactions for item in stats if item.num_interactions is not None]
    avg_interactions = (sum(interactions) / len(interactions)) if interactions else 0.0

    print(f"experiment: {experiment_name}")
    print(f"episodes: {n}")
    print(f"success_rate: {_format_rate(n_success, n)}")
    print(f"avg_subtest_pass_rate: {(pass_rate_sum / n) if n else 0.0:.4f}")
    print(f"avg_num_interactions: {avg_interactions:.2f}")
    print(f"execution_failed_total: {execution_failed_total}")
    print(f"execution_failed_task_rate: {_format_rate(sum(s.n_execution_failed > 0 for s in stats), n)}")
    print(f"no_code_found_total: {no_code_total}")
    print(f"no_code_found_task_rate: {_format_rate(sum(s.n_no_code_found > 0 for s in stats), n)}")
    print(f"invalid_api_regex_hits: {invalid_api_total}")
    print(f"invalid_api_task_rate: {_format_rate(sum(s.invalid_api_hits > 0 for s in stats), n)}")
    print(f"cancelled_total: {cancelled_total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name", nargs="+")
    parser.add_argument("--appworld-root", type=Path, default=None)
    parser.add_argument("--invalid-api-pattern", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    appworld_root = args.appworld_root or os.environ.get("APPWORLD_ROOT")
    if not appworld_root:
        raise RuntimeError("APPWORLD_ROOT is required unless --appworld-root is passed.")

    pattern_texts = args.invalid_api_pattern or list(DEFAULT_INVALID_API_PATTERNS)
    patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in pattern_texts]

    for idx, experiment_name in enumerate(args.experiment_name):
        if idx:
            print()
        summarize(Path(appworld_root), experiment_name, patterns)


if __name__ == "__main__":
    main()
