# 中文注释：分析 AppWorld episode 日志中的 agent 行为指标并输出 JSON/CSV。
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Analyze agent behavior metrics from saved AppWorld episode logs.

The failed-api recovery metric is an approximation because episode.json stores the final
conversation, not a structured per-call execution trace. We treat AppWorld API endpoints that
appear in code from a turn whose following observation contains "Execution failed." as failed,
and mark them recovered if a later non-failing turn calls the same endpoint again.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INVALID_API_PATTERNS = (
    r"invalid api",
    r"api .* not found",
    r"no api named",
    r"unknown api",
    r"attributeerror: .*apis\.",
    r"has no attribute",
)

CODE_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", flags=re.IGNORECASE | re.DOTALL)
PARTIAL_CODE_RE = re.compile(r"```(?:python|py)\s*\n?(.*)$", flags=re.IGNORECASE | re.DOTALL)
API_CALL_RE = re.compile(
    r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
DOC_DESC_RE = re.compile(
    r"\bapis\.api_docs\.show_api_descriptions\s*\((?P<args>[^)]*)\)",
    flags=re.DOTALL,
)
DOC_RE = re.compile(
    r"\bapis\.api_docs\.show_api_doc\s*\((?P<args>[^)]*)\)",
    flags=re.DOTALL,
)
APP_NAME_RE = re.compile(r"app_name\s*=\s*['\"]([^'\"]+)['\"]")
API_NAME_RE = re.compile(r"api_name\s*=\s*['\"]([^'\"]+)['\"]")
ASSUMPTION_RE = re.compile(r"\b(?:assume|assumed|assuming)\b", flags=re.IGNORECASE)
ASSUMPTION_EXTENDED_RE = re.compile(
    r"\b(?:assume|assumed|assuming|guess|guessed|guessing|suppose|supposed|supposing)\b",
    flags=re.IGNORECASE,
)
DUMMY_RE = re.compile(r"\bdummy\b", flags=re.IGNORECASE)
DUMMY_EXTENDED_RE = re.compile(
    r"\b(?:dummy|placeholder|fake|example)\b", flags=re.IGNORECASE
)
CAPITULATION_RE = re.compile(
    r"\b(?:cannot|can't|unable|give up)\b|instead assume|let'?s assume|dummy",
    flags=re.IGNORECASE,
)
EXECUTION_FAILED_TEXT = "Execution failed."
METRIC_KEYS = ("TGC", "SGC", "TGC_1", "TGC_2", "TGC_3", "SGC_1", "SGC_2", "SGC_3")


@dataclass
class Turn:
    assistant_text: str
    observation_text: str
    code_blocks: list[str]
    api_calls: list[tuple[str, str]]
    execution_failed: bool


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


def _extract_code_blocks(text: str) -> list[str]:
    blocks = [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text)]
    last_end = 0
    for match in CODE_BLOCK_RE.finditer(text):
        last_end = match.end()
    partial = PARTIAL_CODE_RE.search(text[last_end:])
    if partial:
        partial_code = partial.group(1).strip()
        if partial_code:
            blocks.append(partial_code)
    return blocks


def _kwarg_literal(args: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(args)
    if not match:
        return None
    return match.group(1)


def _api_calls(code: str) -> list[tuple[str, str]]:
    calls = []
    for match in API_CALL_RE.finditer(code):
        app_name, api_name = match.groups()
        if app_name == "api_docs":
            continue
        calls.append((app_name, api_name))
    return calls


def _doc_before_api_call_counts(code_blocks: list[str]) -> tuple[int, int]:
    documented_apps: set[str] = set()
    documented_endpoints: set[tuple[str, str]] = set()
    documented_calls = 0
    total_calls = 0

    for code in code_blocks:
        events: list[tuple[int, str, tuple[str, str | None]]] = []
        for match in DOC_DESC_RE.finditer(code):
            app_name = _kwarg_literal(match.group("args"), APP_NAME_RE)
            if app_name:
                events.append((match.start(), "doc_desc", (app_name, None)))
        for match in DOC_RE.finditer(code):
            args = match.group("args")
            app_name = _kwarg_literal(args, APP_NAME_RE)
            api_name = _kwarg_literal(args, API_NAME_RE)
            if app_name and api_name:
                events.append((match.start(), "doc", (app_name, api_name)))
        for match in API_CALL_RE.finditer(code):
            app_name, api_name = match.groups()
            if app_name == "api_docs":
                continue
            events.append((match.start(), "api", (app_name, api_name)))

        for _, event_type, payload in sorted(events, key=lambda item: item[0]):
            app_name, api_name = payload
            if event_type == "doc_desc":
                documented_apps.add(app_name)
            elif event_type == "doc" and api_name is not None:
                documented_endpoints.add((app_name, api_name))
            elif event_type == "api" and api_name is not None:
                total_calls += 1
                if app_name in documented_apps or (app_name, api_name) in documented_endpoints:
                    documented_calls += 1

    return documented_calls, total_calls


def _parse_turns(episode: dict[str, Any]) -> list[Turn]:
    messages = episode.get("chat_history") or []
    num_prompt_messages = episode.get("num_prompt_messages") or 0
    task_messages = messages[num_prompt_messages:]
    turns: list[Turn] = []

    for idx, message in enumerate(task_messages):
        if message.get("role") != "assistant":
            continue
        assistant_text = str(message.get("content") or "")
        observation_text = ""
        for next_message in task_messages[idx + 1 :]:
            if next_message.get("role") == "assistant":
                break
            if next_message.get("role") in {"user", "ipython"}:
                observation_text = str(next_message.get("content") or "")
                break
        code_blocks = _extract_code_blocks(assistant_text)
        turns.append(
            Turn(
                assistant_text=assistant_text,
                observation_text=observation_text,
                code_blocks=code_blocks,
                api_calls=[call for block in code_blocks for call in _api_calls(block)],
                execution_failed=EXECUTION_FAILED_TEXT in observation_text,
            )
        )
    return turns


def _parse_all_results(path: Path | None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {key: None for key in METRIC_KEYS}
    if path is None or not path.exists():
        return metrics

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return metrics
    line = lines[-1]
    for key in METRIC_KEYS:
        match = re.search(rf"\b{re.escape(key)}:\s*([0-9.]+)", line)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def _safe_avg(value: int | float, denominator: int) -> float:
    return float(value) / denominator if denominator else 0.0


def _checkpoint_iteration(checkpoint_name: str, explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    match = re.search(r"checkpoint-(\d+)|ckpt(\d+)", checkpoint_name)
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def analyze(
    *,
    appworld_root: Path,
    experiment_name: str,
    all_results_path: Path | None,
    checkpoint_name: str,
    checkpoint_iteration: int | None,
    eval_result_path: str,
    eval_log_path: str,
    behavior_summary_path: str,
) -> dict[str, Any]:
    paths = _episode_paths(appworld_root, experiment_name)
    invalid_patterns = [
        re.compile(pattern, flags=re.IGNORECASE) for pattern in INVALID_API_PATTERNS
    ]
    official_metrics = _parse_all_results(all_results_path)

    n_success = 0
    pass_rate_sum = 0.0
    execution_failed_count = 0
    invalid_api_hits = 0
    total_turns = 0
    code_chars = 0
    multiple_code_cell_turns = 0
    execution_error_turns = 0
    api_doc_calls = 0
    api_description_calls = 0
    documented_api_calls = 0
    total_api_calls = 0
    assumption_words = 0
    assumption_words_extended = 0
    dummy_words = 0
    dummy_words_extended = 0
    failed_api_calls = 0
    recovered_api_calls = 0
    capitulation_after_error_count = 0

    for path in paths:
        episode = json.loads(path.read_text())
        eval_result = episode.get("eval_result") or {}
        num_tests = int(eval_result.get("num_tests") or 0)
        passes = eval_result.get("passes") or []
        if eval_result.get("success"):
            n_success += 1
        pass_rate_sum += (len(passes) / num_tests) if num_tests else 0.0
        execution_failed_count += int(episode.get("n_execution_failed") or 0)

        all_text = "\n".join(_text_values(episode.get("chat_history", [])))
        invalid_api_hits += sum(len(pattern.findall(all_text)) for pattern in invalid_patterns)
        assumption_words += len(ASSUMPTION_RE.findall(all_text))
        assumption_words_extended += len(ASSUMPTION_EXTENDED_RE.findall(all_text))
        dummy_words += len(DUMMY_RE.findall(all_text))
        dummy_words_extended += len(DUMMY_EXTENDED_RE.findall(all_text))

        turns = _parse_turns(episode)
        if not turns and eval_result.get("num_interactions") is not None:
            total_turns += int(eval_result["num_interactions"])
        else:
            total_turns += len(turns)

        pending_failed: Counter[tuple[str, str]] = Counter()
        for idx, turn in enumerate(turns):
            code_chars += sum(len(block) for block in turn.code_blocks)
            if len(turn.code_blocks) > 1:
                multiple_code_cell_turns += 1
            if turn.execution_failed:
                execution_error_turns += 1
                for endpoint in set(turn.api_calls):
                    failed_api_calls += 1
                    pending_failed[endpoint] += 1
            else:
                for endpoint in set(turn.api_calls):
                    if pending_failed[endpoint] > 0:
                        recovered_api_calls += pending_failed[endpoint]
                        pending_failed[endpoint] = 0

            if turn.execution_failed and idx + 1 < len(turns):
                if CAPITULATION_RE.search(turns[idx + 1].assistant_text):
                    capitulation_after_error_count += 1

            for block in turn.code_blocks:
                api_doc_calls += len(DOC_RE.findall(block))
                api_description_calls += len(DOC_DESC_RE.findall(block))

        doc_count, api_count = _doc_before_api_call_counts(
            [block for turn in turns for block in turn.code_blocks]
        )
        documented_api_calls += doc_count
        total_api_calls += api_count

    n_rollouts = len(paths)
    failed_api_call_give_up_rate = (
        (failed_api_calls - recovered_api_calls) / failed_api_calls
        if failed_api_calls
        else 0.0
    )

    row: dict[str, Any] = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_iteration": _checkpoint_iteration(checkpoint_name, checkpoint_iteration),
        "experiment_name": experiment_name,
        "TGC": official_metrics["TGC"],
        "SGC": official_metrics["SGC"],
        "TGC_1": official_metrics["TGC_1"],
        "TGC_2": official_metrics["TGC_2"],
        "TGC_3": official_metrics["TGC_3"],
        "SGC_1": official_metrics["SGC_1"],
        "SGC_2": official_metrics["SGC_2"],
        "SGC_3": official_metrics["SGC_3"],
        "average_partial_pass_rate": _safe_avg(pass_rate_sum, n_rollouts),
        "execution_failed_count": execution_failed_count,
        "invalid_api_hits": invalid_api_hits,
        "num_rollouts_analyzed": n_rollouts,
        "num_turns_avg": _safe_avg(total_turns, n_rollouts),
        "code_chars_per_rollout": _safe_avg(code_chars, n_rollouts),
        "multiple_code_cells_per_turn": _safe_avg(multiple_code_cell_turns, total_turns),
        "execution_errors_per_turn": _safe_avg(execution_error_turns, total_turns),
        "api_doc_calls_per_rollout": _safe_avg(api_doc_calls, n_rollouts),
        "api_description_calls_per_rollout": _safe_avg(api_description_calls, n_rollouts),
        "api_doc_or_description_calls_per_rollout": _safe_avg(
            api_doc_calls + api_description_calls, n_rollouts
        ),
        "doc_before_api_call_rate": _safe_avg(documented_api_calls, total_api_calls),
        "assumption_words_per_rollout": _safe_avg(assumption_words, n_rollouts),
        "assumption_words_extended_per_rollout": _safe_avg(assumption_words_extended, n_rollouts),
        "dummy_words_per_rollout": _safe_avg(dummy_words, n_rollouts),
        "dummy_words_extended_per_rollout": _safe_avg(dummy_words_extended, n_rollouts),
        "failed_api_call_give_up_rate": failed_api_call_give_up_rate,
        "failed_api_calls": failed_api_calls,
        "recovered_api_calls": recovered_api_calls,
        "capitulation_after_error_count": capitulation_after_error_count,
        "eval_result_path": eval_result_path,
        "eval_log_path": eval_log_path,
        "behavior_summary_path": behavior_summary_path,
    }
    return row


def _write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--appworld-root", type=Path, default=None)
    parser.add_argument("--all-results", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default="unknown")
    parser.add_argument("--checkpoint-iteration", type=int, default=None)
    parser.add_argument("--eval-result-path", default="")
    parser.add_argument("--eval-log-path", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    appworld_root = args.appworld_root or os.environ.get("APPWORLD_ROOT")
    if not appworld_root:
        raise RuntimeError("APPWORLD_ROOT is required unless --appworld-root is passed.")

    row = analyze(
        appworld_root=Path(appworld_root),
        experiment_name=args.experiment_name,
        all_results_path=args.all_results,
        checkpoint_name=args.checkpoint_name,
        checkpoint_iteration=args.checkpoint_iteration,
        eval_result_path=args.eval_result_path,
        eval_log_path=args.eval_log_path,
        behavior_summary_path=str(args.output_json),
    )
    _write_json(args.output_json, row)
    if args.output_csv is not None:
        _write_csv(args.output_csv, row)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
