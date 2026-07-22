"""Recompute and audit a turn-local process advantage from saved AppWorld traces.

This is an offline prototype.  It does not change the trainer.  The scorer combines
pairwise first-divergence credit within each six-rollout group with local invariants:

* an execution failure is negative only on the failing turn;
* the first successful correction receives a local recovery bonus;
* repeated API cycles at the tail receive an increasing negative signal; and
* evaluator-proven grounding failures veto positive mutation credit.

The default input is the five 24x6 diagnostic iterations selected by the existing
LOOP15 trajectory analysis: 1, 6, 9, 12, and 15 (720 trajectories total).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence


DEFAULT_ITERATIONS = (1, 6, 9, 12, 15)
EXPECTED_SCENARIOS_PER_ITERATION = 24
EXPECTED_ROLLOUTS_PER_SCENARIO = 6

ERROR_PENALTY = -0.25
RECOVERY_BONUS = 0.15
REPEAT_PENALTY = -0.10
GROUNDING_PENALTY = -0.25
SIGN_EPSILON = 1e-12

CODE_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
PARTIAL_CODE_RE = re.compile(r"```(?:python|py)\s*\n?(.*)$", re.IGNORECASE | re.DOTALL)
API_CALL_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
EXECUTION_FAILED_TEXT = "Execution failed."
ACTUAL_TASK_PREFIX = "Using these APIs, now generate code to solve the"


@dataclass(frozen=True)
class ApiCall:
    endpoint: str
    fingerprint: str


@dataclass
class Turn:
    index: int
    assistant_text: str
    observation: str
    code: str
    calls: list[ApiCall]
    execution_failed: bool

    @property
    def endpoint_names(self) -> tuple[str, ...]:
        return tuple(call.endpoint for call in self.calls)

    @property
    def decision_signature(self) -> tuple[str, ...]:
        return tuple(
            call.endpoint
            for call in self.calls
            if _is_business(call.endpoint) or call.endpoint == "supervisor.complete_task"
        )

    @property
    def repeat_signature(self) -> tuple[str, ...]:
        return tuple(call.fingerprint for call in self.calls)


@dataclass
class Trace:
    label: str
    iteration: int
    scenario_idx: int
    rollout_idx: int
    task_id: str
    difficulty: int
    ret: float
    strict_success: bool
    turns: list[Turn]
    failed_requirements: tuple[str, ...] = ()
    source_path: str = ""
    old_advantage: float = 0.0


@dataclass
class TurnScore:
    pairwise_samples: list[float] = field(default_factory=list)
    pairwise_component: float = 0.0
    error_component: float = 0.0
    repeat_component: float = 0.0
    recovery_component: float = 0.0
    grounding_component: float = 0.0
    new_advantage: float = 0.0
    repeat_ordinal: int = 0
    recovered_correction: bool = False
    recovery_from_turns: list[int] = field(default_factory=list)
    grounding_failures: list[str] = field(default_factory=list)


def _normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _short(text: str, limit: int = 180) -> str:
    normalized = _normalize_space(text)
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _extract_code_blocks(text: str) -> list[str]:
    matches = list(CODE_BLOCK_RE.finditer(text))
    blocks = [match.group(1).strip() for match in matches]
    last_end = matches[-1].end() if matches else 0
    partial = PARTIAL_CODE_RE.search(text[last_end:])
    if partial and (code := partial.group(1).strip()):
        blocks.append(code)
    return blocks


def _balanced_call(code: str, open_paren_index: int) -> str:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(open_paren_index, len(code)):
        char = code[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return code[open_paren_index : index + 1]
    return code[open_paren_index:]


def _calls_from_code(code: str) -> list[ApiCall]:
    calls: list[ApiCall] = []
    for match in API_CALL_RE.finditer(code):
        app_name, api_name = match.groups()
        tail = _balanced_call(code, match.end() - 1)
        expression = _normalize_space(f"apis.{app_name}.{api_name}{tail}")
        calls.append(ApiCall(endpoint=f"{app_name}.{api_name}", fingerprint=expression))
    return calls


def _is_business(endpoint: str) -> bool:
    return not endpoint.startswith("api_docs.") and not endpoint.startswith("supervisor.")


def _actual_task_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    marker_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and ACTUAL_TASK_PREFIX in str(message.get("content") or "")
    ]
    if marker_indices:
        return list(messages[marker_indices[-1] + 1 :])

    # Saved AppWorld episodes expose num_prompt_messages.  Synthetic tests and
    # legacy traces can also pass an already-trimmed message list.
    return list(messages)


def parse_turns(
    messages: Sequence[dict[str, Any]], *, num_prompt_messages: int | None = None
) -> list[Turn]:
    if num_prompt_messages is not None:
        task_messages = list(messages[num_prompt_messages:])
    else:
        task_messages = _actual_task_messages(messages)

    turns: list[Turn] = []
    for message_index, message in enumerate(task_messages):
        if message.get("role") != "assistant":
            continue
        observation = ""
        for following in task_messages[message_index + 1 :]:
            if following.get("role") == "assistant":
                break
            if following.get("role") in {"ipython", "user"}:
                observation = str(following.get("content") or "")
                break
        assistant_text = str(message.get("content") or "")
        code = "\n".join(_extract_code_blocks(assistant_text))
        turns.append(
            Turn(
                index=len(turns) + 1,
                assistant_text=assistant_text,
                observation=observation,
                code=code,
                calls=_calls_from_code(code),
                execution_failed=EXECUTION_FAILED_TEXT in observation,
            )
        )
    return turns


def load_training_trace(path: Path) -> Trace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    turns = parse_turns(list(payload.get("messages") or []))
    expected_turns = int(metadata.get("num_interactions") or 0)
    # A context-truncated rollout records the final attempted interaction in
    # metadata, but that unfinished assistant message is not persisted.
    expected_persisted_turns = expected_turns - int(bool(metadata.get("context_truncated")))
    if expected_turns and len(turns) != expected_persisted_turns:
        raise ValueError(
            f"{path}: parsed {len(turns)} task turns, expected "
            f"{expected_persisted_turns} persisted turns from metadata"
        )
    return Trace(
        label=(
            f"iter-{int(payload['iteration']):02d}/scenario-{int(payload['scenario_idx']):02d}/"
            f"rollout-{int(payload['rollout_idx']):02d}"
        ),
        iteration=int(payload["iteration"]),
        scenario_idx=int(payload["scenario_idx"]),
        rollout_idx=int(payload["rollout_idx"]),
        task_id=str(payload["task_id"]),
        difficulty=int(metadata.get("difficulty") or 0),
        ret=float(metadata.get("return") or 0.0),
        strict_success=bool(metadata.get("strict_success")),
        turns=turns,
        source_path=str(path),
    )


def _d3_label(path: Path, experiment_name: str) -> str:
    lowered = experiment_name.lower()
    for checkpoint in (6, 9, 12, 15):
        if f"ckpt{checkpoint}" in lowered or f"checkpoint-{checkpoint}" in lowered:
            return f"ckpt-{checkpoint}"
    if "r0" in {part.lower() for part in path.parts}:
        return "r0"
    return experiment_name or path.parents[2].name


def load_d3_trace(path: Path, rollout_idx: int) -> Trace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = dict(payload.get("eval_result") or {})
    passes = list(result.get("passes") or [])
    failures = tuple(
        str(item.get("requirement") or item.get("label") or "")
        for item in result.get("failures") or []
    )
    total = int(result.get("num_tests") or len(passes) + len(failures))
    return Trace(
        label=_d3_label(path, str(payload.get("experiment_name") or "")),
        iteration=0,
        scenario_idx=0,
        rollout_idx=rollout_idx,
        task_id=str((payload.get("task") or {}).get("task_id") or path.parents[1].name),
        difficulty=int(result.get("difficulty") or 0),
        ret=float(len(passes) / total) if total else 0.0,
        strict_success=bool(result.get("success")),
        turns=parse_turns(
            list(payload.get("chat_history") or []),
            num_prompt_messages=int(payload.get("num_prompt_messages") or 0),
        ),
        failed_requirements=failures,
        source_path=str(path),
    )


def _decision_turns(trace: Trace) -> list[Turn]:
    return [turn for turn in trace.turns if turn.decision_signature]


def _first_decision_divergence(
    left: Trace, right: Trace
) -> tuple[Turn | None, Turn | None, int] | None:
    left_turns = _decision_turns(left)
    right_turns = _decision_turns(right)
    for index in range(max(len(left_turns), len(right_turns))):
        left_turn = left_turns[index] if index < len(left_turns) else None
        right_turn = right_turns[index] if index < len(right_turns) else None
        left_signature = left_turn.decision_signature if left_turn else None
        right_signature = right_turn.decision_signature if right_turn else None
        if left_signature != right_signature:
            return left_turn, right_turn, index
    return None


def _score_map(traces: Sequence[Trace]) -> dict[tuple[str, int], TurnScore]:
    return {
        (trace.label, turn.index): TurnScore() for trace in traces for turn in trace.turns
    }


def _old_leave_one_out_advantages(traces: Sequence[Trace]) -> None:
    if len(traces) < 2:
        for trace in traces:
            trace.old_advantage = 0.0
        return
    total = sum(trace.ret for trace in traces)
    for trace in traces:
        trace.old_advantage = trace.ret - (total - trace.ret) / (len(traces) - 1)


def _pairwise_first_divergence_credit(
    traces: Sequence[Trace],
    scores: dict[tuple[str, int], TurnScore],
) -> list[dict[str, Any]]:
    forks: list[dict[str, Any]] = []
    for left, right in itertools.combinations(traces, 2):
        if math.isclose(left.ret, right.ret, abs_tol=1e-12):
            continue
        high, low = (left, right) if left.ret > right.ret else (right, left)
        divergence = _first_decision_divergence(high, low)
        if divergence is None:
            continue
        high_turn, low_turn, prefix_length = divergence
        half_delta = (high.ret - low.ret) / 2.0
        if high_turn is not None:
            scores[(high.label, high_turn.index)].pairwise_samples.append(half_delta)
        if low_turn is not None:
            scores[(low.label, low_turn.index)].pairwise_samples.append(-half_delta)

        strict_pair = high.strict_success and not low.strict_success
        forks.append(
            {
                "iteration": high.iteration,
                "scenario_idx": high.scenario_idx,
                "task_id": high.task_id,
                "strict_success_failure_pair": strict_pair,
                "higher_label": high.label,
                "lower_label": low.label,
                "higher_return": high.ret,
                "lower_return": low.ret,
                "common_decision_prefix_length": prefix_length,
                "higher_turn": high_turn.index if high_turn else None,
                "lower_turn": low_turn.index if low_turn else None,
                "higher_action": _endpoint_text(high_turn),
                "lower_action": _endpoint_text(low_turn),
            }
        )
    return forks


def _endpoint_text(turn: Turn | None) -> str:
    if turn is None:
        return "<ended>"
    return " + ".join(turn.decision_signature or turn.endpoint_names) or "<no-api>"


def _tail_repeat_ordinals(turns: Sequence[Turn], max_period: int = 3) -> dict[int, int]:
    api_turns = [turn for turn in turns if turn.repeat_signature]
    signatures = [turn.repeat_signature for turn in api_turns]
    best_positions: dict[int, int] = {}
    for period in range(1, min(max_period, len(signatures) // 2) + 1):
        for start in range(0, len(signatures) - 2 * period + 1):
            suffix = signatures[start:]
            if all(value == suffix[index % period] for index, value in enumerate(suffix)):
                positions = {
                    api_turns[index].index: index - start - period + 1
                    for index in range(start + period, len(api_turns))
                }
                if len(positions) > len(best_positions):
                    best_positions = positions
                break
    return best_positions


def _recovery_pairs(trace: Trace) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for failed_turn in trace.turns:
        if not failed_turn.execution_failed:
            continue
        failed_endpoints = {
            call.endpoint for call in failed_turn.calls if _is_business(call.endpoint)
        }
        correction: Turn | None = None
        for later in trace.turns:
            if later.index <= failed_turn.index or later.execution_failed:
                continue
            later_endpoints = {call.endpoint for call in later.calls if _is_business(call.endpoint)}
            if failed_endpoints:
                if failed_endpoints & later_endpoints:
                    correction = later
                    break
            elif later.code:
                correction = later
                break
        if correction is not None:
            pairs.append((failed_turn.index, correction.index))
    return pairs


def _grounding_failures(turn: Turn, failed_requirements: Sequence[str]) -> list[str]:
    endpoints = set(turn.endpoint_names)
    lowered = [requirement.lower() for requirement in failed_requirements]
    matches: list[str] = []
    if "venmo.create_transaction" in endpoints:
        matches.extend(
            requirement
            for requirement, normalized in zip(failed_requirements, lowered, strict=True)
            if "added transaction" in normalized
            and any(field in normalized for field in ("amount", "receiver_id", "description"))
        )
    if "phone.send_text_message" in endpoints:
        matches.extend(
            requirement
            for requirement, normalized in zip(failed_requirements, lowered, strict=True)
            if "global_text_message" in normalized
            and any(field in normalized for field in ("receiver_id", "message"))
        )
    return matches


def score_trace_group(
    traces: Sequence[Trace],
) -> tuple[dict[tuple[str, int], TurnScore], list[dict[str, Any]], list[dict[str, Any]]]:
    scores = _score_map(traces)
    _old_leave_one_out_advantages(traces)
    forks = _pairwise_first_divergence_credit(traces, scores)
    recoveries: list[dict[str, Any]] = []

    for trace in traces:
        for turn in trace.turns:
            score = scores[(trace.label, turn.index)]
            score.pairwise_component = _safe_mean(score.pairwise_samples)
            if turn.execution_failed:
                score.error_component = ERROR_PENALTY

            failures = _grounding_failures(turn, trace.failed_requirements)
            if failures:
                score.grounding_failures.extend(failures)
                score.grounding_component = GROUNDING_PENALTY

        repeat_ordinals = _tail_repeat_ordinals(trace.turns)
        for turn_index, ordinal in repeat_ordinals.items():
            score = scores[(trace.label, turn_index)]
            score.repeat_ordinal = ordinal
            score.repeat_component = REPEAT_PENALTY * min(ordinal, 3)

        for error_turn, correction_turn in _recovery_pairs(trace):
            score = scores[(trace.label, correction_turn)]
            score.recovered_correction = True
            score.recovery_from_turns.append(error_turn)
            score.recovery_component = RECOVERY_BONUS
            recoveries.append(
                {
                    "iteration": trace.iteration,
                    "scenario_idx": trace.scenario_idx,
                    "rollout_idx": trace.rollout_idx,
                    "task_id": trace.task_id,
                    "trace_label": trace.label,
                    "error_turn": error_turn,
                    "correction_turn": correction_turn,
                }
            )

    for trace in traces:
        for turn in trace.turns:
            score = scores[(trace.label, turn.index)]
            raw = (
                score.pairwise_component
                + score.error_component
                + score.repeat_component
                + score.recovery_component
                + score.grounding_component
            )

            # Semantic vetoes have precedence over empirical terminal credit.
            if score.grounding_failures:
                raw = min(raw, GROUNDING_PENALTY)
            elif turn.execution_failed:
                raw = min(raw, ERROR_PENALTY)
            elif score.recovered_correction:
                # A repaired action must not inherit the failing turn's penalty.
                raw = max(raw - score.repeat_component, RECOVERY_BONUS)
                score.repeat_component = 0.0
            elif score.repeat_ordinal:
                raw = min(raw, score.repeat_component)
            score.new_advantage = _clip(raw)

    turn_lookup = {
        (trace.label, turn.index): turn for trace in traces for turn in trace.turns
    }
    for fork in forks:
        higher_turn = fork["higher_turn"]
        lower_turn = fork["lower_turn"]
        higher_score = (
            scores[(fork["higher_label"], higher_turn)] if higher_turn is not None else None
        )
        lower_score = (
            scores[(fork["lower_label"], lower_turn)] if lower_turn is not None else None
        )
        higher_turn_data = (
            turn_lookup[(fork["higher_label"], higher_turn)]
            if higher_turn is not None
            else None
        )
        lower_turn_data = (
            turn_lookup[(fork["lower_label"], lower_turn)]
            if lower_turn is not None
            else None
        )
        fork["higher_new_advantage"] = (
            higher_score.new_advantage if higher_score is not None else None
        )
        fork["lower_new_advantage"] = (
            lower_score.new_advantage if lower_score is not None else None
        )
        fork["advantage_margin"] = (
            fork["higher_new_advantage"] - fork["lower_new_advantage"]
            if fork["higher_new_advantage"] is not None
            and fork["lower_new_advantage"] is not None
            else None
        )
        fork["higher_execution_failed"] = bool(
            higher_turn_data and higher_turn_data.execution_failed
        )
        fork["lower_execution_failed"] = bool(
            lower_turn_data and lower_turn_data.execution_failed
        )
        fork["higher_recovered_correction"] = bool(
            higher_score and higher_score.recovered_correction
        )
        fork["lower_recovered_correction"] = bool(
            lower_score and lower_score.recovered_correction
        )
        fork["higher_repeat_tail"] = bool(
            higher_score
            and higher_score.repeat_ordinal
            and not higher_score.recovered_correction
        )
        fork["higher_grounding_invalid"] = bool(
            higher_score and higher_score.grounding_failures
        )
        exclusion_reasons: list[str] = []
        if higher_turn is None or lower_turn is None:
            exclusion_reasons.append("one_branch_ended")
        if fork["higher_execution_failed"]:
            exclusion_reasons.append("higher_action_execution_failed")
        if fork["higher_repeat_tail"]:
            exclusion_reasons.append("higher_action_is_repeat_tail")
        if fork["higher_grounding_invalid"]:
            exclusion_reasons.append("higher_action_grounding_invalid")
        if fork["lower_recovered_correction"]:
            exclusion_reasons.append("lower_action_is_recovery")
        fork["attribution_exclusion_reasons"] = exclusion_reasons
        fork["attribution_eligible"] = not exclusion_reasons

    for recovery in recoveries:
        label = recovery["trace_label"]
        recovery["error_new_advantage"] = scores[
            (label, recovery["error_turn"])
        ].new_advantage
        recovery["correction_new_advantage"] = scores[
            (label, recovery["correction_turn"])
        ].new_advantage
        recovery["locality_pass"] = (
            recovery["error_new_advantage"] < 0
            and recovery["correction_new_advantage"] > 0
        )
    return scores, forks, recoveries


def _action_hash(turn: Turn) -> str:
    payload = "\n".join(turn.repeat_signature).encode()
    return hashlib.sha256(payload).hexdigest()[:16] if payload else ""


def _turn_rows(
    traces: Sequence[Trace], scores: dict[tuple[str, int], TurnScore]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for turn in trace.turns:
            score = scores[(trace.label, turn.index)]
            rows.append(
                {
                    "iteration": trace.iteration,
                    "scenario_idx": trace.scenario_idx,
                    "rollout_idx": trace.rollout_idx,
                    "task_id": trace.task_id,
                    "difficulty": trace.difficulty,
                    "trace_label": trace.label,
                    "turn_index": turn.index,
                    "return": trace.ret,
                    "strict_success": trace.strict_success,
                    "old_rollout_advantage": trace.old_advantage,
                    "pairwise_component": score.pairwise_component,
                    "error_component": score.error_component,
                    "repeat_component": score.repeat_component,
                    "recovery_component": score.recovery_component,
                    "grounding_component": score.grounding_component,
                    "new_advantage": score.new_advantage,
                    "execution_failed": turn.execution_failed,
                    "repeat_ordinal": score.repeat_ordinal,
                    "recovered_correction": score.recovered_correction,
                    "recovery_from_turns": score.recovery_from_turns,
                    "grounding_invalid": bool(score.grounding_failures),
                    "grounding_failure_count": len(score.grounding_failures),
                    "endpoints": list(turn.endpoint_names),
                    "decision_endpoints": list(turn.decision_signature),
                    "action_sha256_16": _action_hash(turn),
                }
            )
    return rows


def _group_training_traces(
    run_root: Path, iterations: Sequence[int]
) -> tuple[list[Trace], list[list[Trace]]]:
    all_traces: list[Trace] = []
    groups: list[list[Trace]] = []
    for iteration in iterations:
        paths = sorted(
            (run_root / "rollouts" / f"iteration-{iteration:06d}").glob(
                "scenario-*/rollout-*/trajectory.json"
            )
        )
        expected = EXPECTED_SCENARIOS_PER_ITERATION * EXPECTED_ROLLOUTS_PER_SCENARIO
        if len(paths) != expected:
            raise ValueError(
                f"iteration {iteration}: expected {expected} traces, found {len(paths)}"
            )
        traces = [load_training_trace(path) for path in paths]
        by_scenario: dict[int, list[Trace]] = defaultdict(list)
        for trace in traces:
            by_scenario[trace.scenario_idx].append(trace)
        if len(by_scenario) != EXPECTED_SCENARIOS_PER_ITERATION:
            raise ValueError(f"iteration {iteration}: expected 24 scenario groups")
        for scenario_idx in sorted(by_scenario):
            group = sorted(by_scenario[scenario_idx], key=lambda trace: trace.rollout_idx)
            if len(group) != EXPECTED_ROLLOUTS_PER_SCENARIO:
                raise ValueError(
                    f"iteration {iteration} scenario {scenario_idx}: expected six traces"
                )
            if len({trace.task_id for trace in group}) != 1:
                raise ValueError(
                    f"iteration {iteration} scenario {scenario_idx}: task IDs disagree"
                )
            groups.append(group)
        all_traces.extend(traces)
    return all_traces, groups


def _load_d3_group(root: Path, task_id: str = "530b157_3") -> list[Trace]:
    paths = sorted(root.glob(f"**/tasks/{task_id}/logs/episode.json"))
    traces = [load_d3_trace(path, index) for index, path in enumerate(paths)]
    labels = [trace.label for trace in traces]
    if len(labels) != len(set(labels)):
        raise ValueError(f"D3 model labels are not unique: {labels}")
    return sorted(traces, key=lambda trace: trace.label)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _audit_summary(
    traces: Sequence[Trace],
    turn_rows: Sequence[dict[str, Any]],
    forks: Sequence[dict[str, Any]],
    recoveries: Sequence[dict[str, Any]],
    d3_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    strict_forks = [row for row in forks if row["strict_success_failure_pair"]]
    raw_scorable_forks = [
        row for row in strict_forks if row["advantage_margin"] is not None
    ]
    raw_fork_passes = [
        row for row in raw_scorable_forks if row["advantage_margin"] > 0
    ]
    attributable_forks = [
        row for row in strict_forks if bool(row["attribution_eligible"])
    ]
    fork_passes = [
        row for row in attributable_forks if row["advantage_margin"] > 0
    ]
    fork_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in attributable_forks:
        fork_groups[(int(row["iteration"]), int(row["scenario_idx"]))].append(row)
    group_passes = sum(
        median([float(row["advantage_margin"]) for row in rows]) > 0
        for rows in fork_groups.values()
    )

    excluded_forks = [row for row in strict_forks if not row["attribution_eligible"]]
    exclusion_counts = Counter(
        reason
        for row in excluded_forks
        for reason in row["attribution_exclusion_reasons"]
    )

    repeat_rows = [row for row in turn_rows if int(row["repeat_ordinal"]) > 0]
    repeat_recoveries = [row for row in repeat_rows if row["recovered_correction"]]
    wasteful_repeat_rows = [
        row for row in repeat_rows if not row["recovered_correction"]
    ]
    recovery_passes = [row for row in recoveries if row["locality_pass"]]
    grounding_rows = [row for row in d3_rows if row["grounding_invalid"]]
    values = [float(row["new_advantage"]) for row in turn_rows]
    execution_failed_rows = [row for row in turn_rows if row["execution_failed"]]
    recoverable_error_keys = {
        (row["trace_label"], row["error_turn"]) for row in recoveries
    }
    return {
        "schema_version": "offline-process-advantage-audit-v2",
        "configuration": {
            "iterations": sorted({trace.iteration for trace in traces}),
            "error_penalty": ERROR_PENALTY,
            "recovery_bonus": RECOVERY_BONUS,
            "repeat_penalty_per_ordinal": REPEAT_PENALTY,
            "grounding_penalty": GROUNDING_PENALTY,
        },
        "coverage": {
            "training_trajectory_count": len(traces),
            "training_turn_count": len(turn_rows),
            "training_difficulty_counts": {
                str(difficulty): count
                for difficulty, count in sorted(
                    Counter(trace.difficulty for trace in traces).items()
                )
            },
            "strict_success_failure_fork_pair_count": len(strict_forks),
            "raw_scorable_strict_fork_pair_count": len(raw_scorable_forks),
            "attributable_strict_fork_pair_count": len(attributable_forks),
            "excluded_confounded_fork_pair_count": len(excluded_forks),
            "mixed_group_with_attributable_fork_count": len(fork_groups),
            "repeat_tail_turn_count": len(repeat_rows),
            "wasteful_repeat_tail_turn_count": len(wasteful_repeat_rows),
            "repeat_recovery_overlap_count": len(repeat_recoveries),
            "recoverable_error_count": len(recoveries),
            "execution_failed_turn_count": len(execution_failed_rows),
            "unique_recoverable_error_turn_count": len(
                recoverable_error_keys
            ),
            "unrecovered_error_turn_count": len(execution_failed_rows)
            - len(recoverable_error_keys),
            "unique_recovery_correction_turn_count": len(
                {(row["trace_label"], row["correction_turn"]) for row in recoveries}
            ),
            "d3_grounding_invalid_operation_count": len(grounding_rows),
            "d3_trace_count": len({row["trace_label"] for row in d3_rows}),
        },
        "checks": {
            "successful_first_fork_action_is_higher": {
                "passed": len(fork_passes) == len(attributable_forks)
                and bool(attributable_forks),
                "pair_pass_count": len(fork_passes),
                "pair_count": len(attributable_forks),
                "pair_pass_rate": len(fork_passes) / len(attributable_forks)
                if attributable_forks
                else 0.0,
                "raw_pair_pass_count": len(raw_fork_passes),
                "raw_pair_count": len(raw_scorable_forks),
                "raw_pair_pass_rate": len(raw_fork_passes) / len(raw_scorable_forks)
                if raw_scorable_forks
                else 0.0,
                "excluded_confounded_pair_count": len(excluded_forks),
                "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
                "group_median_pass_count": group_passes,
                "group_count": len(fork_groups),
                "mean_advantage_margin": _safe_mean(
                    [float(row["advantage_margin"]) for row in attributable_forks]
                ),
            },
            "repeated_api_tail_is_negative": {
                "passed": bool(wasteful_repeat_rows)
                and all(
                    float(row["new_advantage"]) < 0 for row in wasteful_repeat_rows
                ),
                "negative_count": sum(
                    float(row["new_advantage"]) < 0 for row in wasteful_repeat_rows
                ),
                "count": len(wasteful_repeat_rows),
                "raw_repeat_count": len(repeat_rows),
                "recovery_overlap_count": len(repeat_recoveries),
                "recovery_overlap_positive_count": sum(
                    float(row["new_advantage"]) > 0 for row in repeat_recoveries
                ),
            },
            "recoverable_error_is_turn_local": {
                "passed": bool(recoveries) and len(recovery_passes) == len(recoveries),
                "pass_count": len(recovery_passes),
                "count": len(recoveries),
            },
            "d3_wrong_grounded_mutation_has_no_positive_progress": {
                "passed": bool(grounding_rows)
                and all(float(row["new_advantage"]) <= 0 for row in grounding_rows),
                "nonpositive_count": sum(
                    float(row["new_advantage"]) <= 0 for row in grounding_rows
                ),
                "count": len(grounding_rows),
            },
        },
        "distribution": {
            "positive_turn_count": sum(value > SIGN_EPSILON for value in values),
            "zero_turn_count": sum(abs(value) <= SIGN_EPSILON for value in values),
            "negative_turn_count": sum(value < -SIGN_EPSILON for value in values),
            "mean": _safe_mean(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        },
    }


def _d3_rows(
    traces: Sequence[Trace], scores: dict[tuple[str, int], TurnScore]
) -> list[dict[str, Any]]:
    rows = _turn_rows(traces, scores)
    by_label = {trace.label: trace for trace in traces}
    for row in rows:
        trace = by_label[str(row["trace_label"])]
        score = scores[(trace.label, int(row["turn_index"]))]
        row["grounding_failures"] = [_short(item, 220) for item in score.grounding_failures]
    return rows


def _report_markdown(summary: dict[str, Any], d3_rows: Sequence[dict[str, Any]]) -> str:
    coverage = summary["coverage"]
    checks = summary["checks"]
    distribution = summary["distribution"]
    lines = [
        "# Offline process-advantage audit",
        "",
        "## Scope",
        "",
        (
            f"Recomputed {coverage['training_trajectory_count']} saved training trajectories "
            f"({coverage['training_turn_count']} task turns) from iterations 1/6/9/12/15."
        ),
        (
            "The 720 training traces contain D1/D2 tasks only "
            f"({coverage['training_difficulty_counts']}); the D3 grounding check uses "
            f"{coverage['d3_trace_count']} saved dev traces as a separate holdout audit."
        ),
        "The production trainer was not changed.",
        "",
        "## Scoring contract",
        "",
        (
            "- Pairwise terminal-return difference is assigned only at the first different "
            "business decision within each six-rollout group."
        ),
        f"- An execution failure is locally capped at `{ERROR_PENALTY}`.",
        f"- Its first observed successful correction is locally floored at `+{RECOVERY_BONUS}`.",
        (
            f"- Repeated 1-3 action cycles at the API tail receive `{REPEAT_PENALTY}` per "
            "repeat ordinal, capped after three ordinals; a verified recovery takes precedence."
        ),
        (
            "- Evaluator-proven amount/receiver/message grounding failures veto positive "
            f"mutation credit at `{GROUNDING_PENALTY}`."
        ),
        "",
        "## Four checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    fork = checks["successful_first_fork_action_is_higher"]
    repeat = checks["repeated_api_tail_is_negative"]
    recovery = checks["recoverable_error_is_turn_local"]
    grounding = checks["d3_wrong_grounded_mutation_has_no_positive_progress"]
    lines.extend(
        [
            (
                "| Successful action is higher at first strict fork | "
                f"{'PASS' if fork['passed'] else 'FAIL'} | "
                f"{fork['pair_pass_count']}/{fork['pair_count']} attributable pairs; "
                f"raw {fork['raw_pair_pass_count']}/{fork['raw_pair_count']}; "
                f"{fork['group_median_pass_count']}/{fork['group_count']} group medians; "
                f"mean margin {fork['mean_advantage_margin']:.4f} |"
            ),
            (
                "| Repeated API tail is negative | "
                f"{'PASS' if repeat['passed'] else 'FAIL'} | "
                f"{repeat['negative_count']}/{repeat['count']} wasteful repeats negative; "
                f"{repeat['recovery_overlap_count']} recovery overlaps protected |"
            ),
            (
                "| Recoverable error is turn-local | "
                f"{'PASS' if recovery['passed'] else 'FAIL'} | "
                f"{recovery['pass_count']}/{recovery['count']} error/correction pairs have "
                "negative error and positive correction |"
            ),
            (
                "| D3 wrong grounded mutation has no positive progress | "
                f"{'PASS' if grounding['passed'] else 'FAIL'} | "
                f"{grounding['nonpositive_count']}/{grounding['count']} evaluator-invalid "
                "mutations non-positive |"
            ),
            "",
            "## Attribution exclusions",
            "",
            (
                f"The raw strict success/failure comparison orders "
                f"{fork['raw_pair_pass_count']}/{fork['raw_pair_count']} pairs correctly. "
                f"The hard check excludes {fork['excluded_confounded_pair_count']} pairs where "
                "the higher-return branch's divergent turn failed locally, the lower-return "
                "branch was already correcting an earlier error, or one branch had ended."
            ),
            "",
            "Exclusion counts: `"
            + json.dumps(fork["exclusion_reason_counts"], sort_keys=True)
            + "`.",
            "",
            (
                f"Repeat detection found {repeat['raw_repeat_count']} tail turns. "
                f"{repeat['recovery_overlap_count']} are also verified corrections and retain "
                "positive recovery credit; the remaining repeats are the wasteful-tail check."
            ),
            (
                f"The recovery check covers "
                f"{coverage['unique_recoverable_error_turn_count']}/"
                f"{coverage['execution_failed_turn_count']} failed turns with an observed "
                f"successful correction. The other {coverage['unrecovered_error_turn_count']} "
                "failed turns remain negative but are not called recoverable."
            ),
            "",
            "## Signal distribution",
            "",
            (
                f"Positive/zero/negative turns: {distribution['positive_turn_count']}/"
                f"{distribution['zero_turn_count']}/{distribution['negative_turn_count']}; "
                f"mean `{distribution['mean']:.6f}`, range "
                f"`[{distribution['min']:.3f}, {distribution['max']:.3f}]`."
            ),
            "",
            "## D3 grounding counterexamples",
            "",
            "| Model | Turn | Endpoint | Return | Grounding failure | New advantage |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for row in d3_rows:
        if not row["grounding_invalid"]:
            continue
        failures = "; ".join(row["grounding_failures"])
        lines.append(
            f"| {row['trace_label']} | {row['turn_index']} | "
            f"{', '.join(row['endpoints'])} | {float(row['return']):.3f} | "
            f"{failures} | {float(row['new_advantage']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This audit validates credit-assignment invariants on saved behavior; it does "
                "not estimate policy improvement from retraining. A final successful trajectory "
                "does not make each of its intermediate failed calls a successful action, and a "
                "final failed trajectory can still contain a valid correction. Those locally "
                "confounded forks stay visible in the raw rate but are not used for the clean "
                "ordering check. Zero-valued turns remain intentionally uncredited rather than "
                "inheriting a trajectory-wide scalar."
            ),
            "",
            "## Files",
            "",
            "- `turn_advantages.csv`: every training turn and all advantage components.",
            "- `critical_forks.csv`: pairwise first-divergence comparisons.",
            "- `repeat_tail_audit.csv`: repeated-tail turns and recovery overlaps.",
            "- `recoverable_errors.csv`: failing turn and first successful correction.",
            "- `d3_grounding_audit.csv`: D3 turn scores and evaluator grounding vetoes.",
            "- `summary.json`: machine-readable counts and pass/fail checks.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    traces, groups = _group_training_traces(args.run_root.resolve(), args.iterations)
    expected = (
        len(args.iterations)
        * EXPECTED_SCENARIOS_PER_ITERATION
        * EXPECTED_ROLLOUTS_PER_SCENARIO
    )
    if len(traces) != expected:
        raise ValueError(f"expected {expected} training trajectories, found {len(traces)}")

    all_turn_rows: list[dict[str, Any]] = []
    all_forks: list[dict[str, Any]] = []
    all_recoveries: list[dict[str, Any]] = []
    for group in groups:
        scores, forks, recoveries = score_trace_group(group)
        all_turn_rows.extend(_turn_rows(group, scores))
        all_forks.extend(forks)
        all_recoveries.extend(recoveries)

    d3_traces = _load_d3_group(args.d3_root.resolve(), args.d3_task_id)
    d3_scores, _, _ = score_trace_group(d3_traces)
    d3_rows = _d3_rows(d3_traces, d3_scores)

    summary = _audit_summary(
        traces,
        all_turn_rows,
        all_forks,
        all_recoveries,
        d3_rows,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "turn_advantages.csv", all_turn_rows)
    _write_csv(output_dir / "critical_forks.csv", all_forks)
    _write_csv(
        output_dir / "repeat_tail_audit.csv",
        [row for row in all_turn_rows if int(row["repeat_ordinal"]) > 0],
    )
    _write_csv(output_dir / "recoverable_errors.csv", all_recoveries)
    _write_csv(output_dir / "d3_grounding_audit.csv", d3_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _report_markdown(summary, d3_rows), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "experiments/qwen25_7b_d12_100x1_loop15_lora16/"
            "r1_loop15_seed20260718"
        ),
    )
    parser.add_argument("--iterations", type=int, nargs="+", default=list(DEFAULT_ITERATIONS))
    parser.add_argument(
        "--d3-root",
        type=Path,
        default=Path("artifacts/sft_loop15/trajectory_analysis_20260720/raw_dev"),
    )
    parser.add_argument("--d3-task-id", default="530b157_3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sft_loop15/offline_process_advantage_20260720"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(check["passed"] for check in summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
