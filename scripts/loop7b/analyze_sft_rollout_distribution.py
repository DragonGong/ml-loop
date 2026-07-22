"""Analyze paired AppWorld rollout distributions saved as sanitized trajectories."""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import re
import statistics
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODEL_ORDER = (
    "base",
    "d12_100x1",
    "loop_best_ckpt130",
    "grpo_best_ckpt180",
)
MODEL_LABELS = {
    "base": "Base",
    "d12_100x1": "d12_100x1",
    "loop_best_ckpt130": "LOOP checkpoint-130",
    "grpo_best_ckpt180": "GRPO checkpoint-180",
}
CODE_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
PARTIAL_CODE_RE = re.compile(r"```(?:python|py)\s*\n?(.*)$", re.IGNORECASE | re.DOTALL)
API_CALL_RE = re.compile(
    r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
EXECUTION_FAILED_TEXT = "Execution failed."
LOW_RETURN_STD = 0.15
LOW_RETURN_RANGE = 0.25
BLOCKED_PARTIAL_RANGE = 0.05
ADVANTAGE_THRESHOLD = 0.01


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _extract_code_blocks(text: str) -> list[str]:
    matches = list(CODE_BLOCK_RE.finditer(text))
    blocks = [match.group(1).strip() for match in matches]
    last_end = max((match.end() for match in matches), default=0)
    partial = PARTIAL_CODE_RE.search(text[last_end:])
    if partial and (code := partial.group(1).strip()):
        blocks.append(code)
    return blocks


def _action_text(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    blocks = _extract_code_blocks(content)
    return "\n".join(blocks) if blocks else content


def _turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        observation = ""
        for following in messages[index + 1 :]:
            role = following.get("role")
            if role == "assistant":
                break
            if role in {"user", "ipython"}:
                observation = str(following.get("content") or "")
                break
        action = _action_text(message)
        turns.append(
            {
                "action": action,
                "normalized_action": " ".join(action.split()),
                "observation": observation,
                "execution_failed": EXECUTION_FAILED_TEXT in observation,
                "api_calls": tuple(
                    f"{app}.{api}" for app, api in API_CALL_RE.findall(action)
                ),
                "code_blocks": _extract_code_blocks(str(message.get("content") or "")),
            }
        )
    return turns


def _code_tokens(turns: list[dict[str, Any]]) -> tuple[str, ...]:
    result: list[str] = []
    for turn in turns:
        for code in turn["code_blocks"]:
            result.append("<TURN>")
            try:
                generated = tokenize.generate_tokens(io.StringIO(code).readline)
                for token in generated:
                    if token.type in {
                        tokenize.ENCODING,
                        tokenize.ENDMARKER,
                        tokenize.NEWLINE,
                        tokenize.NL,
                        tokenize.INDENT,
                        tokenize.DEDENT,
                        tokenize.COMMENT,
                    }:
                        continue
                    if token.type == tokenize.STRING:
                        result.append("<STR>")
                    elif token.type == tokenize.NUMBER:
                        result.append("<NUM>")
                    else:
                        result.append(token.string.lower())
            except (IndentationError, SyntaxError, tokenize.TokenError):
                result.extend(
                    re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", code.lower())
                )
    return tuple(result)


def _pair_similarity(sequences: list[tuple[str, ...]]) -> float:
    def shingles(sequence: tuple[str, ...]) -> set[tuple[str, ...]]:
        if len(sequence) < 3:
            return {(token,) for token in sequence}
        return set(zip(sequence, sequence[1:], sequence[2:]))

    scores = []
    for left, right in itertools.combinations(sequences, 2):
        if not left or not right:
            continue
        left_shingles = shingles(left)
        right_shingles = shingles(right)
        scores.append(
            _ratio(
                len(left_shingles & right_shingles),
                len(left_shingles | right_shingles),
            )
        )
    return _mean(scores)


def _majority_consistency(values: list[str]) -> float:
    return _ratio(max(Counter(values).values()), len(values)) if values else 0.0


def _rollout_features(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload["metadata"])
    turns = _turns(payload["messages"])
    api_sequence = tuple(endpoint for turn in turns for endpoint in turn["api_calls"])
    business_sequence = tuple(
        endpoint
        for endpoint in api_sequence
        if not endpoint.startswith("api_docs.") and not endpoint.startswith("supervisor.")
    )
    actions = [turn["normalized_action"] for turn in turns if turn["normalized_action"]]
    duplicate_actions = len(actions) - len(set(actions))
    consecutive_duplicates = sum(
        current == previous for previous, current in zip(actions, actions[1:])
    )

    failed_api_calls = 0
    recovered_api_calls = 0
    pending: Counter[str] = Counter()
    for turn in turns:
        endpoints = set(turn["api_calls"])
        if turn["execution_failed"]:
            for endpoint in endpoints:
                failed_api_calls += 1
                pending[endpoint] += 1
        else:
            for endpoint in endpoints:
                if pending[endpoint]:
                    recovered_api_calls += pending[endpoint]
                    pending[endpoint] = 0

    return {
        "scenario_idx": int(payload["scenario_idx"]),
        "rollout_idx": int(payload["rollout_idx"]),
        "task_id": payload["task_id"],
        **metadata,
        "api_sequence": api_sequence,
        "business_api_sequence": business_sequence,
        "first_api": api_sequence[0] if api_sequence else None,
        "first_business_api": business_sequence[0] if business_sequence else None,
        "code_tokens": _code_tokens(turns),
        "duplicate_code_action_count": duplicate_actions,
        "consecutive_duplicate_code_action_count": consecutive_duplicates,
        "failed_api_calls": failed_api_calls,
        "recovered_api_calls": recovered_api_calls,
    }


def _classify(success_count: int, return_std: float, return_range: float, partial_range: float) -> str:
    low_variance = return_std <= LOW_RETURN_STD and return_range <= LOW_RETURN_RANGE
    if success_count >= 4 and low_variance:
        return "mastered"
    if success_count == 0 and partial_range <= BLOCKED_PARTIAL_RANGE:
        return "blocked"
    return "learnable"


def _scenario_row(
    model: str, selection: dict[str, Any], rollouts: list[dict[str, Any]]
) -> dict[str, Any]:
    returns = [float(row["return"]) for row in rollouts]
    partials = [float(row["partial_pass"]) for row in rollouts]
    successes = [bool(row["strict_success"]) for row in rollouts]
    return_mean = _mean(returns)
    advantages = [len(returns) / (len(returns) - 1) * (value - return_mean) for value in returns]
    effective = [abs(value) >= ADVANTAGE_THRESHOLD for value in advantages]
    return_std = statistics.pstdev(returns)
    return_range = max(returns) - min(returns)
    partial_range = max(partials) - min(partials)
    api_sequences = [row["api_sequence"] for row in rollouts]
    business_sequences = [row["business_api_sequence"] for row in rollouts]
    first_apis = [row["first_api"] for row in rollouts if row["first_api"]]
    first_business = [
        row["first_business_api"] for row in rollouts if row["first_business_api"]
    ]
    code_sequences = [row["code_tokens"] for row in rollouts]
    success_count = sum(successes)
    classification = _classify(success_count, return_std, return_range, partial_range)
    failed_api_calls = sum(int(row["failed_api_calls"]) for row in rollouts)
    recovered_api_calls = sum(int(row["recovered_api_calls"]) for row in rollouts)

    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "scenario_idx": selection["scenario_idx"],
        "scenario_family": selection["scenario_family"],
        "task_id": selection["task_id"],
        "difficulty": selection["difficulty"],
        "classification": classification,
        "return_all_same": return_std <= 1e-8,
        "unique_return_count": len(set(round(value, 8) for value in returns)),
        "return_mean": return_mean,
        "return_std": return_std,
        "return_range": return_range,
        "effective_advantage_ratio": _mean(effective),
        "strict_success_count": success_count,
        "strict_success_rate": success_count / len(rollouts),
        "at_least_one_success": success_count > 0,
        "best_of_6_return": max(returns),
        "average_partial": _mean(partials),
        "partial_range": partial_range,
        "execution_failed_count": sum(int(row["execution_failed_count"]) for row in rollouts),
        "http_401_count": sum(int(row["http_401_count"]) for row in rollouts),
        "http_422_count": sum(int(row["http_422_count"]) for row in rollouts),
        "name_error_count": sum(int(row["name_error_count"]) for row in rollouts),
        "invalid_api_hits": sum(int(row["invalid_api_hits"]) for row in rollouts),
        "no_code_found_count": sum(int(row["no_code_found_count"]) for row in rollouts),
        "failed_api_calls": failed_api_calls,
        "recovered_api_calls": recovered_api_calls,
        "error_recovery_rate": _ratio(recovered_api_calls, failed_api_calls),
        "api_doc_calls": sum(int(row["api_doc_calls"]) for row in rollouts),
        "api_description_calls": sum(
            int(row["api_description_calls"]) for row in rollouts
        ),
        "duplicate_code_action_count": sum(
            int(row["duplicate_code_action_count"]) for row in rollouts
        ),
        "consecutive_repeated_failed_action_count": sum(
            int(row["consecutive_repeated_failed_action_count"]) for row in rollouts
        ),
        "average_interactions": _mean(int(row["num_interactions"]) for row in rollouts),
        "truncation_rate": _mean(bool(row["context_truncated"]) for row in rollouts),
        "unique_api_sequence_count": len(set(api_sequences)),
        "unique_business_api_sequence_count": len(set(business_sequences)),
        "first_api_coverage": len(first_apis) / len(rollouts),
        "first_api_consistency": _majority_consistency(first_apis),
        "first_business_api_coverage": len(first_business) / len(rollouts),
        "first_business_api_consistency": _majority_consistency(first_business),
        "code_sequence_coverage": _mean(bool(sequence) for sequence in code_sequences),
        "pairwise_code_sequence_similarity": _pair_similarity(code_sequences),
    }


def _model_summary(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_count = len(rows) * 6
    classes = Counter(row["classification"] for row in rows)
    d3_rows = [row for row in rows if row["difficulty"] == "D3"]
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "scenario_count": len(rows),
        "rollout_count": rollout_count,
        "return_all_same_scenario_rate": _mean(row["return_all_same"] for row in rows),
        "mean_unique_return_count": _mean(row["unique_return_count"] for row in rows),
        "mean_return_range": _mean(row["return_range"] for row in rows),
        "effective_advantage_rollout_ratio": _mean(
            row["effective_advantage_ratio"] for row in rows
        ),
        "at_least_one_success_scenario_rate": _mean(
            row["at_least_one_success"] for row in rows
        ),
        "mean_best_of_6_return": _mean(row["best_of_6_return"] for row in rows),
        "strict_success_rollout_rate": _ratio(
            sum(row["strict_success_count"] for row in rows), rollout_count
        ),
        "average_partial": _mean(row["average_partial"] for row in rows),
        "execution_failed_per_rollout": _ratio(
            sum(row["execution_failed_count"] for row in rows), rollout_count
        ),
        "http_401_per_rollout": _ratio(sum(row["http_401_count"] for row in rows), rollout_count),
        "http_422_per_rollout": _ratio(sum(row["http_422_count"] for row in rows), rollout_count),
        "name_error_per_rollout": _ratio(
            sum(row["name_error_count"] for row in rows), rollout_count
        ),
        "invalid_api_per_rollout": _ratio(
            sum(row["invalid_api_hits"] for row in rows), rollout_count
        ),
        "no_code_per_rollout": _ratio(
            sum(row["no_code_found_count"] for row in rows), rollout_count
        ),
        "error_recovery_rate": _ratio(
            sum(row["recovered_api_calls"] for row in rows),
            sum(row["failed_api_calls"] for row in rows),
        ),
        "doc_queries_per_rollout": _ratio(
            sum(row["api_doc_calls"] + row["api_description_calls"] for row in rows),
            rollout_count,
        ),
        "duplicate_code_actions_per_rollout": _ratio(
            sum(row["duplicate_code_action_count"] for row in rows), rollout_count
        ),
        "average_interactions": _mean(row["average_interactions"] for row in rows),
        "truncation_rate": _mean(row["truncation_rate"] for row in rows),
        "mean_unique_api_sequence_count": _mean(
            row["unique_api_sequence_count"] for row in rows
        ),
        "mean_unique_business_api_sequence_count": _mean(
            row["unique_business_api_sequence_count"] for row in rows
        ),
        "mean_first_business_api_consistency": _mean(
            row["first_business_api_consistency"] for row in rows
        ),
        "mean_code_sequence_similarity": _mean(
            row["pairwise_code_sequence_similarity"] for row in rows
        ),
        "mastered_count": classes["mastered"],
        "learnable_count": classes["learnable"],
        "blocked_count": classes["blocked"],
        "d3_mastered_count": sum(row["classification"] == "mastered" for row in d3_rows),
        "d3_learnable_count": sum(row["classification"] == "learnable" for row in d3_rows),
        "d3_blocked_count": sum(row["classification"] == "blocked" for row in d3_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: Any) -> str:
    return "" if value is None else f"{100 * float(value):.1f}%"


def _report(summaries: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> str:
    lines = [
        "# SFT initialization rollout distribution",
        "",
        "Each model uses the same 30 canonical train scenario instances (one `_1` task per",
        "scenario family), six request seeds, temperature 1.0, and max 40 interactions.",
        "",
        "## Model summary",
        "",
        "| Model | Success | Any success | Best-of-6 | Partial | Same return | Effective adv | Mastered | Learnable | Blocked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model_label} | {success} | {any_success} | {best:.3f} | {partial:.3f} | "
            "{same} | {adv} | {mastered} | {learnable} | {blocked} |".format(
                model_label=row["model_label"],
                success=_percent(row["strict_success_rollout_rate"]),
                any_success=_percent(row["at_least_one_success_scenario_rate"]),
                best=row["mean_best_of_6_return"],
                partial=row["average_partial"],
                same=_percent(row["return_all_same_scenario_rate"]),
                adv=_percent(row["effective_advantage_rollout_ratio"]),
                mastered=row["mastered_count"],
                learnable=row["learnable_count"],
                blocked=row["blocked_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Behavior and diversity",
            "",
            "| Model | Exec/rollout | 401 | 422 | NameError | Invalid API | No-code | Recovery | Docs | Turns | Truncation | API unique | First API consistency | Code similarity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {model_label} | {execution_failed_per_rollout:.2f} | {http_401_per_rollout:.2f} | "
            "{http_422_per_rollout:.2f} | {name_error_per_rollout:.2f} | "
            "{invalid_api_per_rollout:.2f} | {no_code_per_rollout:.2f} | {recovery} | "
            "{doc_queries_per_rollout:.2f} | {average_interactions:.1f} | {truncation} | "
            "{mean_unique_api_sequence_count:.2f} | {first} | {code:.3f} |".format(
                **row,
                recovery=_percent(row["error_recovery_rate"]),
                truncation=_percent(row["truncation_rate"]),
                first=_percent(row["mean_first_business_api_consistency"]),
                code=row["mean_code_sequence_similarity"],
            )
        )
    lines.extend(
        [
            "",
            "## D3 classification",
            "",
            "| Model | Mastered | Learnable | Blocked |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['model_label']} | {row['d3_mastered_count']} | "
            f"{row['d3_learnable_count']} | {row['d3_blocked_count']} |"
        )
    lines.extend(
        [
            "",
            "## Classification contract",
            "",
            f"- `mastered`: at least 4/6 strict successes, return std <= {LOW_RETURN_STD}, and return range <= {LOW_RETURN_RANGE}.",
            f"- `blocked`: 0/6 strict successes and partial-pass range <= {BLOCKED_PARTIAL_RANGE}.",
            "- `learnable`: every remaining scenario, including mixed success/failure or meaningful partial variation.",
            f"- Effective advantage uses leave-one-out returns and `abs(advantage) >= {ADVANTAGE_THRESHOLD}`.",
            "- Code similarity is mean pairwise Jaccard similarity over normalized Python token 3-grams.",
            "",
            "## d12_100x1 scenarios",
            "",
            "| Scenario | Difficulty | Class | Success | Return range | Partial range | Best-of-6 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in scenarios:
        if row["model"] != "d12_100x1":
            continue
        lines.append(
            f"| {row['task_id']} | {row['difficulty']} | {row['classification']} | "
            f"{row['strict_success_count']}/6 | {row['return_range']:.3f} | "
            f"{row['partial_range']:.3f} | {row['best_of_6_return']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    selection = json.loads((root / "scenario_selection.json").read_text(encoding="utf-8"))["rows"]
    if len(selection) != 30:
        raise ValueError(f"expected 30 selected scenarios, got {len(selection)}")

    scenario_rows: list[dict[str, Any]] = []
    for model in args.models:
        trajectory_root = root / "runs" / model / "trajectories" / "iteration-000000"
        for selected in selection:
            scenario_idx = int(selected["scenario_idx"])
            paths = sorted(
                (trajectory_root / f"scenario-{scenario_idx:04d}").glob(
                    "rollout-*/trajectory.json"
                )
            )
            if len(paths) != 6:
                raise ValueError(f"{model} scenario {scenario_idx} has {len(paths)} trajectories")
            rollouts = [
                _rollout_features(json.loads(path.read_text(encoding="utf-8")))
                for path in paths
            ]
            if {row["task_id"] for row in rollouts} != {selected["task_id"]}:
                raise ValueError(f"{model} scenario {scenario_idx} task mismatch")
            scenario_rows.append(_scenario_row(model, selected, rollouts))

    summaries = [
        _model_summary(model, [row for row in scenario_rows if row["model"] == model])
        for model in args.models
    ]
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "model_summary.csv", summaries)
    _write_csv(analysis_dir / "scenario_metrics.csv", scenario_rows)
    (analysis_dir / "model_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "scenario_metrics.json").write_text(
        json.dumps(scenario_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "report.md").write_text(
        _report(summaries, scenario_rows), encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "selection_policy": "one canonical _1 task per each of 30 train scenario families",
        "rollouts_per_scenario": 6,
        "temperature": 1.0,
        "mastered": {
            "minimum_successes": 4,
            "maximum_return_std": LOW_RETURN_STD,
            "maximum_return_range": LOW_RETURN_RANGE,
        },
        "blocked": {
            "successes": 0,
            "maximum_partial_range": BLOCKED_PARTIAL_RANGE,
        },
        "effective_advantage_threshold": ADVANTAGE_THRESHOLD,
    }
    (analysis_dir / "analysis_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
