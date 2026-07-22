"""Paired AppWorld dev and grouped LOOP trajectory analysis.

The dev analysis is task paired: R0 is compared with checkpoints 6, 9, 12,
and 15 on the same 57 tasks.  The training analysis uses the complete rollout
diagnostics for iterations 1, 6, 9, 12, and 15.  Per-requirement training
evaluator reports are optional because AppWorld runner output paths can be
overwritten; their coverage is always reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


CODE_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
PARTIAL_CODE_RE = re.compile(r"```(?:python|py)\s*\n?(.*)$", re.IGNORECASE | re.DOTALL)
API_CALL_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
HTTP_401_RE = re.compile(r"(?<!\d)401(?!\d)")
HTTP_422_RE = re.compile(r"(?<!\d)422(?!\d)")
NAME_ERROR_RE = re.compile(r"\bNameError\b", re.IGNORECASE)
NAME_ERROR_NAME_RE = re.compile(r"name ['\"]([^'\"]+)['\"] is not defined", re.IGNORECASE)
EXECUTION_FAILED_TEXT = "Execution failed."

MODEL_DIRS = {
    "r0": "r0",
    "ckpt-6": "eval_qwen25_7b_d12_100x1_loop15_lora16_r1_ckpt6_dev_small64",
    "ckpt-9": "eval_qwen25_7b_d12_100x1_loop15_lora16_r1_ckpt9_dev_small64",
    "ckpt-12": "eval_qwen25_7b_d12_100x1_loop15_lora16_r1_ckpt12_dev_small64",
    "ckpt-15": "eval_qwen25_7b_d12_100x1_loop15_lora16_r1_ckpt15_dev_small64",
}
LOOP_LABELS = tuple(label for label in MODEL_DIRS if label != "r0")
DEFAULT_ITERATIONS = (1, 6, 9, 12, 15)

MUTATION_RE = re.compile(
    r"(?:^|_)(?:add|archive|block|cancel|create|delete|follow|like|logout|move|"
    r"pay|purchase|rate|remove|rename|reply|review|send|share|signup|transfer|"
    r"unfollow|unlike|update|upload)(?:_|$)",
    re.IGNORECASE,
)
IRREVERSIBLE_RE = re.compile(
    r"(?:^|_)(?:archive|cancel|delete|logout|move|purchase|remove|transfer)(?:_|$)",
    re.IGNORECASE,
)
QUERY_RE = re.compile(r"(?:^|_)(?:fetch|find|get|list|search|show)(?:_|$)", re.IGNORECASE)
TASK_MUTATION_RE = re.compile(
    r"\b(?:add|archive|block|buy|cancel|create|delete|follow|like|move|pay|"
    r"purchase|rate|remove|rename|reply|review|send|share|transfer|unfollow|"
    r"unlike|update|upload)\b",
    re.IGNORECASE,
)
ALL_OBJECTS_RE = re.compile(r"\b(?:all|each|every|entire|exactly)\b", re.IGNORECASE)
PAGINATION_RE = re.compile(
    r"\b(?:cursor|has_more|next_page|offset|page_index|page_number|page_size)\b",
    re.IGNORECASE,
)
LOOP_RE = re.compile(r"\b(?:for|while)\b|\+=\s*1|range\s*\(", re.IGNORECASE)

EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
PHONE_RE = re.compile(r"(?<!\d)\d{7,15}(?!\d)")
TOKEN_VALUE_RE = re.compile(
    r"(?i)(?P<prefix>(?:access_token|password|token)\s*[:=]\s*)"
    r"(?P<quote>['\"]?)[^,\s}\)]+(?P=quote)"
)


@dataclass
class ApiCall:
    endpoint: str
    turn_index: int
    kwargs: tuple[str, ...]
    fingerprint: str
    turn_failed: bool


@dataclass
class Turn:
    index: int
    assistant_text: str
    observation: str
    code: str
    calls: list[ApiCall]
    execution_failed: bool


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def _short(text: str, limit: int = 180) -> str:
    text = _normalize_space(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _redact(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return TOKEN_VALUE_RE.sub(lambda match: match.group("prefix") + "[SECRET]", text)


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


def _calls_from_code(code: str, *, turn_index: int, failed: bool) -> list[ApiCall]:
    calls: list[ApiCall] = []
    for match in API_CALL_RE.finditer(code):
        app_name, api_name = match.groups()
        call_tail = _balanced_call(code, match.end() - 1)
        expression = f"apis.{app_name}.{api_name}{call_tail}"
        kwargs = tuple(sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", call_tail))))
        calls.append(
            ApiCall(
                endpoint=f"{app_name}.{api_name}",
                turn_index=turn_index,
                kwargs=kwargs,
                fingerprint=_normalize_space(expression),
                turn_failed=failed,
            )
        )
    return calls


def _parse_turns(episode: dict[str, Any]) -> list[Turn]:
    messages = list(episode.get("chat_history") or [])
    prompt_count = int(episode.get("num_prompt_messages") or 0)
    messages = messages[prompt_count:]
    turns: list[Turn] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        observation = ""
        for following in messages[message_index + 1 :]:
            if following.get("role") == "assistant":
                break
            if following.get("role") in {"ipython", "user"}:
                observation = str(following.get("content") or "")
                break
        assistant_text = str(message.get("content") or "")
        code = "\n".join(_extract_code_blocks(assistant_text))
        failed = EXECUTION_FAILED_TEXT in observation
        turn_index = len(turns) + 1
        turns.append(
            Turn(
                index=turn_index,
                assistant_text=assistant_text,
                observation=observation,
                code=code,
                calls=_calls_from_code(code, turn_index=turn_index, failed=failed),
                execution_failed=failed,
            )
        )
    return turns


def _is_business(endpoint: str) -> bool:
    return not endpoint.startswith("api_docs.") and not endpoint.startswith("supervisor.")


def _doc_target(call: ApiCall) -> str | None:
    if call.endpoint == "api_docs.show_api_descriptions":
        app = re.search(r"app_name\s*=\s*['\"]([^'\"]+)", call.fingerprint)
        return f"{app.group(1)}.*" if app else "unknown.*"
    if call.endpoint == "api_docs.show_api_doc":
        app = re.search(r"app_name\s*=\s*['\"]([^'\"]+)", call.fingerprint)
        api = re.search(r"api_name\s*=\s*['\"]([^'\"]+)", call.fingerprint)
        if app and api:
            return f"{app.group(1)}.{api.group(1)}"
        return "unknown.unknown"
    return None


def _error_type(observation: str) -> str:
    if NAME_ERROR_RE.search(observation):
        return "NameError"
    if HTTP_401_RE.search(observation):
        return "401"
    if HTTP_422_RE.search(observation):
        return "422"
    if EXECUTION_FAILED_TEXT in observation:
        return "execution_failed"
    return ""


def _error_excerpt(observation: str) -> str:
    normalized = _normalize_space(_redact(observation))
    markers = ("NameError", "401", "422", "Error", "detail", "message")
    positions = [normalized.lower().find(marker.lower()) for marker in markers]
    positions = [position for position in positions if position >= 0]
    start = max(0, min(positions) - 40) if positions else 0
    return _short(normalized[start:], 260)


def _error_cause(
    error_type: str, observation: str, failed_calls: Sequence[ApiCall], turns: Sequence[Turn]
) -> str:
    endpoint = failed_calls[0].endpoint if failed_calls else "unknown endpoint"
    if error_type == "NameError":
        match = NAME_ERROR_NAME_RE.search(observation)
        return (
            f"undefined local variable {match.group(1)!r}" if match else "undefined local variable"
        )
    if error_type == "401":
        if endpoint.endswith(".login"):
            return f"credentials rejected by {endpoint}"
        prior_login = any(
            call.endpoint.endswith(".login") and not turn.execution_failed
            for turn in turns
            for call in turn.calls
            if turn.index < (failed_calls[0].turn_index if failed_calls else math.inf)
        )
        if not prior_login:
            return f"protected {endpoint} called before a successful login"
        return f"access token rejected by {endpoint}"
    if error_type == "422":
        location = re.search(r"['\"]loc['\"]\s*:\s*\[[^\]]*['\"]([^'\"]+)['\"]", observation)
        message = re.search(r"['\"](?:msg|message|detail)['\"]\s*:\s*['\"]([^'\"]+)", observation)
        detail = message.group(1) if message else "request validation failed"
        if location:
            detail = f"parameter {location.group(1)!r}: {detail}"
        return f"{endpoint}: {_short(detail, 150)}"
    return f"runtime failure in {endpoint}: {_error_excerpt(observation)}"


def _requirement_signature(item: dict[str, Any]) -> str:
    return _normalize_space(str(item.get("requirement") or item.get("label") or "unknown"))


def _scenario_id(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def _trace_from_episode(path: Path, label: str) -> dict[str, Any]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    task = dict(episode.get("task") or {})
    task_id = str(task.get("task_id") or path.parents[1].name)
    instruction = str(task.get("instruction") or "")
    eval_result = dict(episode.get("eval_result") or {})
    passes = [_requirement_signature(item) for item in eval_result.get("passes") or []]
    failures = [_requirement_signature(item) for item in eval_result.get("failures") or []]
    num_tests = int(eval_result.get("num_tests") or len(passes) + len(failures))
    turns = _parse_turns(episode)
    calls = [call for turn in turns for call in turn.calls]
    business_calls = [call for call in calls if _is_business(call.endpoint)]
    docs = [target for call in calls if (target := _doc_target(call)) is not None]
    doc_counts = Counter(docs)
    repeated_doc_calls = sum(count - 1 for count in doc_counts.values() if count > 1)

    error_turns = [
        turn
        for turn in turns
        if turn.execution_failed
        or HTTP_401_RE.search(turn.observation)
        or HTTP_422_RE.search(turn.observation)
        or NAME_ERROR_RE.search(turn.observation)
    ]
    first_error = error_turns[0] if error_turns else None
    first_error_type = _error_type(first_error.observation) if first_error else ""
    first_error_calls = first_error.calls if first_error else []
    first_error_cause = (
        _error_cause(first_error_type, first_error.observation, first_error_calls, turns)
        if first_error
        else ""
    )
    error_events = []
    for error_turn in error_turns:
        event_type = _error_type(error_turn.observation)
        event_calls = error_turn.calls
        error_events.append(
            {
                "turn": error_turn.index,
                "type": event_type,
                "endpoint": next(
                    (call.endpoint for call in event_calls if _is_business(call.endpoint)),
                    event_calls[0].endpoint if event_calls else "",
                ),
                "cause": _error_cause(event_type, error_turn.observation, event_calls, turns),
                "excerpt": _error_excerpt(error_turn.observation),
            }
        )

    recovered = False
    changed_api = False
    changed_parameters = False
    recovery_endpoint = ""
    if first_error:
        failed_business = [call for call in first_error.calls if _is_business(call.endpoint)]
        later_calls = [
            call
            for turn in turns
            if turn.index > first_error.index
            for call in turn.calls
            if _is_business(call.endpoint)
        ]
        if failed_business and later_calls:
            changed_api = later_calls[0].endpoint != failed_business[0].endpoint
        for failed_call in failed_business:
            for later in later_calls:
                if later.endpoint == failed_call.endpoint:
                    changed_parameters |= later.fingerprint != failed_call.fingerprint
                    if not later.turn_failed:
                        recovered = True
                        recovery_endpoint = later.endpoint
                        break
            if recovered:
                break

    code = "\n".join(turn.code for turn in turns)
    requires_all = bool(ALL_OBJECTS_RE.search(instruction + "\n" + "\n".join(failures)))
    pagination_signal = bool(PAGINATION_RE.search(code))
    repeated_query_with_changes = False
    by_endpoint: dict[str, set[str]] = defaultdict(set)
    for call in business_calls:
        if QUERY_RE.search(call.endpoint.rsplit(".", 1)[-1]):
            by_endpoint[call.endpoint].add(call.fingerprint)
    repeated_query_with_changes = any(len(values) > 1 for values in by_endpoint.values())
    pagination_handled = pagination_signal and (
        bool(LOOP_RE.search(code)) or repeated_query_with_changes
    )
    if pagination_handled:
        pagination_status = "handled_by_code"
    elif requires_all and any(
        QUERY_RE.search(call.endpoint.rsplit(".", 1)[-1]) for call in business_calls
    ):
        pagination_status = "likely_missing_or_unproven"
    else:
        pagination_status = "not_required_or_not_evident"

    success = bool(eval_result.get("success"))
    if success:
        all_objects_status = "evaluator_complete"
    elif requires_all and any(ALL_OBJECTS_RE.search(requirement) for requirement in failures):
        all_objects_status = "evaluator_incomplete"
    else:
        all_objects_status = "unknown"

    mutation_calls = [
        call for call in business_calls if MUTATION_RE.search(call.endpoint.rsplit(".", 1)[-1])
    ]
    task_requires_mutation = bool(TASK_MUTATION_RE.search(instruction))
    missing_final_operation = task_requires_mutation and not mutation_calls
    complete_task_calls = [call for call in calls if call.endpoint == "supervisor.complete_task"]
    irreversible = [
        call for call in business_calls if IRREVERSIBLE_RE.search(call.endpoint.rsplit(".", 1)[-1])
    ]
    early_cutoff = max(2, math.ceil(len(turns) / 3))
    early_irreversible = [call for call in irreversible if call.turn_index <= early_cutoff]

    last_turn_status = "not_truncated"
    if bool(episode.get("context_truncated")):
        recent = turns[-3:]
        earlier_endpoints = {
            call.endpoint
            for turn in turns[:-3]
            for call in turn.calls
            if _is_business(call.endpoint)
        }
        recent_business = [
            call for turn in recent for call in turn.calls if _is_business(call.endpoint)
        ]
        new_endpoint = any(call.endpoint not in earlier_endpoints for call in recent_business)
        successful_mutation = any(
            MUTATION_RE.search(call.endpoint.rsplit(".", 1)[-1]) and not call.turn_failed
            for call in recent_business
        )
        repeated_fail = bool(recent) and all(turn.execution_failed for turn in recent)
        if new_endpoint or successful_mutation:
            last_turn_status = "still_advancing"
        elif repeated_fail or not recent_business:
            last_turn_status = "stalled"
        else:
            last_turn_status = "repeating_without_clear_progress"

    execution_failed_count = int(
        episode.get("n_execution_failed")
        if isinstance(episode.get("n_execution_failed"), int)
        else sum(turn.execution_failed for turn in turns)
    )
    first_business = business_calls[0] if business_calls else None
    wasteful_success = success and (
        repeated_doc_calls >= 2
        or execution_failed_count >= 3
        or len(business_calls) >= 30
        or bool(episode.get("context_truncated"))
    )
    return {
        "label": label,
        "task_id": task_id,
        "scenario_id": _scenario_id(task_id),
        "difficulty": int(eval_result.get("difficulty") or task_id.rsplit("_", 1)[-1]),
        "instruction": _short(_redact(instruction), 300),
        "success": success,
        "partial_pass": _safe_ratio(len(passes), num_tests),
        "num_tests": num_tests,
        "passed_requirements": passes,
        "failed_requirements": failures,
        "turn_count": len(turns),
        "business_call_count": len(business_calls),
        "business_api_sequence": [call.endpoint for call in business_calls],
        "business_call_signatures": [
            f"{call.endpoint}({','.join(call.kwargs)})" for call in business_calls
        ],
        "first_business_api": first_business.endpoint if first_business else "",
        "first_business_api_turn": first_business.turn_index if first_business else None,
        "first_business_api_turn_failed": first_business.turn_failed if first_business else None,
        "first_business_api_matches_success_reference": None,
        "first_error_turn": first_error.index if first_error else None,
        "first_error_type": first_error_type,
        "first_error_endpoint": (
            next((call.endpoint for call in first_error_calls if _is_business(call.endpoint)), "")
            if first_error
            else ""
        ),
        "first_error_cause": first_error_cause,
        "first_error_excerpt": _error_excerpt(first_error.observation) if first_error else "",
        "error_events": error_events,
        "recovered_after_first_error": recovered,
        "changed_api_after_first_error": changed_api,
        "changed_parameters_after_first_error": changed_parameters,
        "recovery_endpoint": recovery_endpoint,
        "execution_failed_count": execution_failed_count,
        "http_401_count": sum(len(HTTP_401_RE.findall(turn.observation)) for turn in turns),
        "http_422_count": sum(len(HTTP_422_RE.findall(turn.observation)) for turn in turns),
        "name_error_count": sum(len(NAME_ERROR_RE.findall(turn.observation)) for turn in turns),
        "api_doc_call_count": len(docs),
        "repeated_api_doc_call_count": repeated_doc_calls,
        "repeated_api_docs": [target for target, count in doc_counts.items() if count > 1],
        "pagination_status": pagination_status,
        "all_objects_status": all_objects_status,
        "mutation_api_count": len(mutation_calls),
        "mutation_apis": [call.endpoint for call in mutation_calls],
        "missing_final_operation": missing_final_operation,
        "complete_task_called": bool(complete_task_calls),
        "complete_task_call_count": len(complete_task_calls),
        "early_irreversible_apis": [call.endpoint for call in early_irreversible],
        "context_truncated": bool(episode.get("context_truncated")),
        "last_turn_status": last_turn_status,
        "wasteful_success": wasteful_success,
        "episode_path": str(path),
    }


def _longest_common_prefix(left: Sequence[str], right: Sequence[str]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def _divergence(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, str, str]:
    left_sequence = left["business_api_sequence"]
    right_sequence = right["business_api_sequence"]
    prefix = _longest_common_prefix(left_sequence, right_sequence)
    left_api = left_sequence[prefix] if prefix < len(left_sequence) else "<ended>"
    right_api = right_sequence[prefix] if prefix < len(right_sequence) else "<ended>"
    return prefix, left_api, right_api


def _short_requirement(requirement: str) -> str:
    requirement = requirement.split("```", 1)[0]
    return _short(requirement, 110)


def _fix_summary(r0: dict[str, Any], loop: dict[str, Any]) -> str:
    prefix, r0_api, loop_api = _divergence(r0, loop)
    parts = [f"API[{prefix + 1}] {r0_api} -> {loop_api}"]
    for key, label in (
        ("http_401_count", "401"),
        ("http_422_count", "422"),
        ("name_error_count", "NameError"),
        ("execution_failed_count", "execution failures"),
    ):
        if loop[key] < r0[key]:
            parts.append(f"{label} {r0[key]}->{loop[key]}")
    if r0["missing_final_operation"] and not loop["missing_final_operation"]:
        parts.append("added the required mutation")
    if not r0["complete_task_called"] and loop["complete_task_called"]:
        parts.append("added complete_task")
    if (
        r0["pagination_status"] != "handled_by_code"
        and loop["pagination_status"] == "handled_by_code"
    ):
        parts.append("added pagination/all-object loop")
    if r0["context_truncated"] and not loop["context_truncated"]:
        parts.append("finished before truncation")
    new_passes = sorted(set(loop["passed_requirements"]) - set(r0["passed_requirements"]))
    if new_passes:
        parts.append("new pass: " + _short_requirement(new_passes[0]))
    return "; ".join(parts)


def _pair_transition(r0: dict[str, Any], loop: dict[str, Any]) -> str:
    if not r0["success"] and loop["success"]:
        return "gain"
    if r0["success"] and not loop["success"]:
        return "regression"
    if r0["success"] and loop["success"]:
        return "stable_success"
    if loop["partial_pass"] > r0["partial_pass"] + 1e-12:
        return "failed_partial_up"
    if loop["partial_pass"] < r0["partial_pass"] - 1e-12:
        return "failed_partial_down"
    return "failed_partial_same"


def _task_category(traces: dict[str, dict[str, Any]]) -> str:
    r0 = traces["r0"]
    loops = [traces[label] for label in LOOP_LABELS]
    if not r0["success"] and any(trace["success"] for trace in loops):
        return "sft_failed_loop_succeeded"
    if r0["success"] and any(not trace["success"] for trace in loops):
        return "sft_succeeded_loop_failed"
    if not r0["success"] and all(not trace["success"] for trace in loops):
        if max(trace["partial_pass"] for trace in loops) > r0["partial_pass"] + 1e-12:
            return "always_failed_partial_improved"
        return "always_failed_no_progress"
    return "stable_success"


def _model_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scenario_success = Counter()
    scenario_total = Counter()
    for row in rows:
        scenario_total[row["scenario_id"]] += 1
        scenario_success[row["scenario_id"]] += int(row["success"])
    completed_scenarios = sum(
        scenario_total[key] == scenario_success[key] for key in scenario_total
    )
    first_reference = [
        row["first_business_api_matches_success_reference"]
        for row in rows
        if row["first_business_api_matches_success_reference"] is not None
    ]
    return {
        "model": rows[0]["label"],
        "episode_count": len(rows),
        "task_success_count": sum(row["success"] for row in rows),
        "TGC": 100.0 * _mean([int(row["success"]) for row in rows]),
        "SGC": 100.0 * _safe_ratio(completed_scenarios, len(scenario_total)),
        "mean_partial_pass": _mean([row["partial_pass"] for row in rows]),
        "first_api_success_reference_rate": _mean([int(value) for value in first_reference]),
        "first_api_failed_turn_rate": _mean(
            [int(bool(row["first_business_api_turn_failed"])) for row in rows]
        ),
        "execution_failed_count": sum(row["execution_failed_count"] for row in rows),
        "http_401_count": sum(row["http_401_count"] for row in rows),
        "http_422_count": sum(row["http_422_count"] for row in rows),
        "name_error_count": sum(row["name_error_count"] for row in rows),
        "first_error_recovery_rate": _mean(
            [
                int(row["recovered_after_first_error"])
                for row in rows
                if row["first_error_turn"] is not None
            ]
        ),
        "api_doc_calls": sum(row["api_doc_call_count"] for row in rows),
        "repeated_api_doc_calls": sum(row["repeated_api_doc_call_count"] for row in rows),
        "complete_task_rate": _mean([int(row["complete_task_called"]) for row in rows]),
        "missing_final_operation_count": sum(row["missing_final_operation"] for row in rows),
        "early_irreversible_episode_count": sum(
            bool(row["early_irreversible_apis"]) for row in rows
        ),
        "context_truncated_count": sum(row["context_truncated"] for row in rows),
        "context_truncation_rate": _mean([int(row["context_truncated"]) for row in rows]),
        "truncated_still_advancing_count": sum(
            row["last_turn_status"] == "still_advancing" for row in rows
        ),
        "wasteful_success_count": sum(row["wasteful_success"] for row in rows),
    }


def analyze_dev(
    dev_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for label, dirname in MODEL_DIRS.items():
        paths = sorted((dev_root / dirname).glob("tasks/*/logs/episode.json"))
        if len(paths) != 57:
            raise ValueError(
                f"{label}: expected 57 dev episodes under {dev_root / dirname}, got {len(paths)}"
            )
        by_model[label] = {
            trace["task_id"]: trace
            for trace in (_trace_from_episode(path, label) for path in paths)
        }

    task_ids = sorted(by_model["r0"])
    for label in LOOP_LABELS:
        if set(by_model[label]) != set(task_ids):
            raise ValueError(f"task ids differ between r0 and {label}")

    for task_id in task_ids:
        success_first_apis = {
            by_model[label][task_id]["first_business_api"]
            for label in MODEL_DIRS
            if by_model[label][task_id]["success"]
            and by_model[label][task_id]["first_business_api"]
        }
        for label in MODEL_DIRS:
            trace = by_model[label][task_id]
            trace["first_business_api_matches_success_reference"] = (
                trace["first_business_api"] in success_first_apis if success_first_apis else None
            )

    task_matrix: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        traces = {label: by_model[label][task_id] for label in MODEL_DIRS}
        category = _task_category(traces)
        matrix = {
            "task_id": task_id,
            "scenario_id": traces["r0"]["scenario_id"],
            "difficulty": traces["r0"]["difficulty"],
            "category": category,
            "instruction": traces["r0"]["instruction"],
        }
        for label, trace in traces.items():
            prefix = label.replace("-", "_")
            matrix[f"{prefix}_success"] = trace["success"]
            matrix[f"{prefix}_partial"] = trace["partial_pass"]
            matrix[f"{prefix}_first_api"] = trace["first_business_api"]
            matrix[f"{prefix}_first_error"] = trace["first_error_type"]
            matrix[f"{prefix}_truncated"] = trace["context_truncated"]
        task_matrix.append(matrix)

        r0 = traces["r0"]
        for label in LOOP_LABELS:
            loop = traces[label]
            prefix, r0_api, loop_api = _divergence(r0, loop)
            pair_rows.append(
                {
                    "task_id": task_id,
                    "difficulty": r0["difficulty"],
                    "checkpoint": label,
                    "category": category,
                    "transition": _pair_transition(r0, loop),
                    "r0_success": r0["success"],
                    "loop_success": loop["success"],
                    "r0_partial": r0["partial_pass"],
                    "loop_partial": loop["partial_pass"],
                    "common_business_api_prefix_length": prefix,
                    "r0_divergence_api": r0_api,
                    "loop_divergence_api": loop_api,
                    "r0_first_api": r0["first_business_api"],
                    "loop_first_api": loop["first_business_api"],
                    "r0_first_error_turn": r0["first_error_turn"],
                    "r0_first_error_type": r0["first_error_type"],
                    "r0_first_error_cause": r0["first_error_cause"],
                    "loop_first_error_turn": loop["first_error_turn"],
                    "loop_first_error_type": loop["first_error_type"],
                    "loop_first_error_cause": loop["first_error_cause"],
                    "r0_error_recovered": r0["recovered_after_first_error"],
                    "loop_error_recovered": loop["recovered_after_first_error"],
                    "r0_repeated_docs": r0["repeated_api_doc_call_count"],
                    "loop_repeated_docs": loop["repeated_api_doc_call_count"],
                    "r0_pagination": r0["pagination_status"],
                    "loop_pagination": loop["pagination_status"],
                    "r0_all_objects": r0["all_objects_status"],
                    "loop_all_objects": loop["all_objects_status"],
                    "r0_missing_final_operation": r0["missing_final_operation"],
                    "loop_missing_final_operation": loop["missing_final_operation"],
                    "r0_complete_task": r0["complete_task_called"],
                    "loop_complete_task": loop["complete_task_called"],
                    "r0_early_irreversible": r0["early_irreversible_apis"],
                    "loop_early_irreversible": loop["early_irreversible_apis"],
                    "r0_last_turn_status": r0["last_turn_status"],
                    "loop_last_turn_status": loop["last_turn_status"],
                    "fix_summary": _fix_summary(r0, loop),
                }
            )

    trace_rows = [by_model[label][task_id] for label in MODEL_DIRS for task_id in task_ids]
    model_stats = [
        _model_stats([by_model[label][task_id] for task_id in task_ids]) for label in MODEL_DIRS
    ]
    return task_matrix, pair_rows, trace_rows, model_stats


def _report_pass_set(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    passed = re.search(r"Num Passed Tests\s*:\s*(\d+)", text)
    total = re.search(r"Num Total\s+Tests\s*:\s*(\d+)", text)
    requirements = []
    for match in re.finditer(
        r">> Passed Requirement\s*\n(.*?)(?=\n>> |\n[-─]{5,}|\Z)", text, re.DOTALL
    ):
        requirement = match.group(1).split("```", 1)[0]
        requirements.append(_normalize_space(requirement))
    parts = path.parts
    try:
        task_id = parts[parts.index("tasks") + 1]
    except (ValueError, IndexError):
        task_id = path.parents[2].name
    passed_count = int(passed.group(1)) if passed else len(requirements)
    total_count = int(total.group(1)) if total else 0
    return {
        "task_id": task_id,
        "passed_count": passed_count,
        "total_count": total_count,
        "return": _safe_ratio(passed_count, total_count),
        "pass_set": tuple(sorted(requirements)),
        "path": str(path),
    }


def _training_report_rows(root: Path, iteration: int) -> list[dict[str, Any]]:
    return [
        _report_pass_set(path)
        for path in sorted((root / f"iteration-{iteration}").glob("**/evaluation/report.md"))
    ]


def _rollout_bad_variant(row: dict[str, Any]) -> bool:
    return bool(
        row.get("context_truncated")
        or int(row.get("execution_failed_count") or 0) > 0
        or int(row.get("no_code_found_count") or 0) > 0
        or int(row.get("consecutive_repeated_failed_action_count") or 0) > 0
    )


def _group_divergence(group: dict[str, Any]) -> tuple[str, str, int]:
    successes = [row for row in group["rollouts"] if row.get("strict_success")]
    failures = [row for row in group["rollouts"] if not row.get("strict_success")]
    if not successes or not failures:
        return "", "", 0
    success = max(successes, key=lambda row: float(row.get("return") or 0.0))
    failure = min(failures, key=lambda row: float(row.get("return") or 0.0))
    success_sequence = list(success.get("business_api_sequence") or [])
    failure_sequence = list(failure.get("business_api_sequence") or [])
    prefix = _longest_common_prefix(success_sequence, failure_sequence)
    success_api = success_sequence[prefix] if prefix < len(success_sequence) else "<ended>"
    failure_api = failure_sequence[prefix] if prefix < len(failure_sequence) else "<ended>"
    return success_api, failure_api, prefix


def analyze_training(
    run_root: Path,
    report_root: Path,
    iterations: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    group_details: list[dict[str, Any]] = []
    for iteration in iterations:
        diagnostics_path = run_root / "rollout_diagnostics" / f"iteration-{iteration:06d}.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        groups = list(diagnostics.get("groups") or [])
        if len(groups) != 24 or int(diagnostics.get("num_rollouts") or 0) != 144:
            raise ValueError(f"iteration {iteration}: expected 24x6 diagnostics")

        reports = _training_report_rows(report_root, iteration)
        reports_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for report in reports:
            reports_by_task[report["task_id"]].append(report)

        comparable_groups = 0
        different_pass_groups = 0
        same_reward_pairs = 0
        different_pass_pairs = 0
        error_only_sequence_count = 0
        distinct_sequence_count = 0
        error_dominated_groups = 0
        mixed_groups = 0

        for group in groups:
            rollouts = list(group.get("rollouts") or [])
            sequence_rows: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for rollout in rollouts:
                sequence_rows[tuple(rollout.get("business_api_sequence") or [])].append(rollout)
            group_error_only = sum(
                all(_rollout_bad_variant(row) for row in owners)
                for owners in sequence_rows.values()
            )
            error_only_sequence_count += group_error_only
            distinct_sequence_count += len(sequence_rows)
            error_only_ratio = _safe_ratio(group_error_only, len(sequence_rows))
            error_dominated_groups += error_only_ratio > 0.5

            strict_success_count = int(group.get("strict_success_count") or 0)
            mixed = 0 < strict_success_count < len(rollouts)
            mixed_groups += mixed
            success_api, failure_api, prefix = _group_divergence(group)

            task_reports = reports_by_task.get(str(group.get("task_id")), [])
            group_same_pairs = 0
            group_diff_pairs = 0
            for left_index, left in enumerate(task_reports):
                for right in task_reports[left_index + 1 :]:
                    if (left["passed_count"], left["total_count"]) != (
                        right["passed_count"],
                        right["total_count"],
                    ):
                        continue
                    group_same_pairs += 1
                    group_diff_pairs += left["pass_set"] != right["pass_set"]
            if group_same_pairs:
                comparable_groups += 1
                different_pass_groups += group_diff_pairs > 0
            same_reward_pairs += group_same_pairs
            different_pass_pairs += group_diff_pairs

            group_details.append(
                {
                    "iteration": iteration,
                    "scenario_idx": group.get("scenario_idx"),
                    "task_id": group.get("task_id"),
                    "returns": group.get("returns"),
                    "zero_return_class": group.get("zero_return_class"),
                    "strict_success_count": strict_success_count,
                    "unique_return_count": group.get("unique_return_count"),
                    "unique_business_api_sequence_count": group.get(
                        "unique_business_api_sequence_count"
                    ),
                    "effective_advantage_rollout_count": group.get(
                        "effective_advantage_rollout_count"
                    ),
                    "error_only_business_sequence_ratio": error_only_ratio,
                    "success_failure_common_prefix_length": prefix if mixed else None,
                    "success_divergence_api": success_api,
                    "failure_divergence_api": failure_api,
                    "evaluator_report_count": len(task_reports),
                    "same_reward_report_pair_count": group_same_pairs,
                    "same_reward_different_pass_pair_count": group_diff_pairs,
                }
            )

        classes = diagnostics.get("zero_return_group_class_counts") or {}
        summaries.append(
            {
                "iteration": iteration,
                "group_count": len(groups),
                "rollout_count": diagnostics.get("num_rollouts"),
                "all_returns_same_group_count": diagnostics.get("zero_return_std_group_count"),
                "all_returns_same_group_rate": diagnostics.get("zero_return_std_group_rate"),
                "all_success_group_count": classes.get("all_success", 0),
                "all_failure_group_count": classes.get("all_failure", 0),
                "partial_same_group_count": classes.get("partial_same", 0),
                "non_zero_variance_group_count": classes.get("non_zero_variance", 0),
                "mean_unique_return_count": diagnostics.get("mean_unique_return_count_per_group"),
                "effective_advantage_rollout_ratio": diagnostics.get(
                    "effective_advantage_rollout_ratio"
                ),
                "effective_advantage_token_ratio": diagnostics.get(
                    "effective_advantage_token_ratio"
                ),
                "mean_unique_business_api_sequence_count": diagnostics.get(
                    "mean_unique_business_api_sequence_count_per_group"
                ),
                "same_business_api_sequence_ratio": diagnostics.get(
                    "same_business_api_sequence_ratio"
                ),
                "mixed_success_failure_group_count": mixed_groups,
                "error_only_business_sequence_ratio": _safe_ratio(
                    error_only_sequence_count, distinct_sequence_count
                ),
                "error_dominated_diversity_group_rate": _safe_ratio(
                    error_dominated_groups, len(groups)
                ),
                "evaluator_report_count": len(reports),
                "evaluator_report_coverage": _safe_ratio(len(reports), 144),
                "same_reward_comparable_group_count": comparable_groups,
                "same_reward_different_pass_group_count": different_pass_groups,
                "same_reward_different_pass_group_rate_lower_bound": _safe_ratio(
                    different_pass_groups, len(groups)
                ),
                "same_reward_different_pass_rate_among_comparable_groups": _safe_ratio(
                    different_pass_groups, comparable_groups
                ),
                "same_reward_report_pair_count": same_reward_pairs,
                "same_reward_different_pass_pair_count": different_pass_pairs,
                "same_reward_different_pass_pair_rate": _safe_ratio(
                    different_pass_pairs, same_reward_pairs
                ),
                "execution_failed_per_rollout": (diagnostics.get("behavior") or {}).get(
                    "execution_failed_per_rollout"
                ),
                "context_truncation_ratio": (diagnostics.get("behavior") or {}).get(
                    "context_truncation_ratio"
                ),
                "strict_success_rate": (diagnostics.get("behavior") or {}).get(
                    "strict_success_rate"
                ),
                "average_partial_pass": (diagnostics.get("behavior") or {}).get(
                    "average_partial_pass"
                ),
            }
        )
    return summaries, group_details


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _number(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _category_counts(task_matrix: Sequence[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row["category"]) for row in task_matrix)


def _report_markdown(
    task_matrix: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
    model_stats: Sequence[dict[str, Any]],
    training: Sequence[dict[str, Any]],
    group_details: Sequence[dict[str, Any]],
) -> str:
    counts = _category_counts(task_matrix)
    total_successes = sum(bool(row["success"]) for row in trace_rows)
    wasteful_successes = sum(bool(row["wasteful_success"]) for row in trace_rows)
    gain_pairs = [row for row in pair_rows if row["transition"] == "gain"]
    unique_gain_tasks = len({str(row["task_id"]) for row in gain_pairs})
    lines = [
        "# SFT LOOP15 paired trajectory analysis",
        "",
        "## Scope and evidence",
        "",
        "- Dev: 57 same tasks, one episode each for R0 and checkpoints 6/9/12/15.",
        "- Task success below is AppWorld `eval_result.success` (the numerator of TGC). SGC is all three difficulty tasks in a scenario succeeding.",
        "- Training: complete 24x6 diagnostics for iterations 1/6/9/12/15 (720 rollouts).",
        "- Training per-requirement reports are a recovered subset. Rates using them are marked as observed lower bounds.",
        "- `pagination_status`, early irreversible action, and wasteful-success flags are behavioral heuristics; evaluator pass/fail and API/error traces are direct observations.",
        "",
        "## Dev headline",
        "",
        "| Model | SGC | TGC | Mean partial | Exec failures | 401 | 422 | NameError | Repeated docs | Truncation | Wasteful successes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_stats:
        lines.append(
            "| {model} | {sgc:.1f} | {tgc:.1f} | {partial:.3f} | {errors} | {e401} | {e422} | {name} | {docs} | {trunc} | {waste} |".format(
                model=row["model"],
                sgc=row["SGC"],
                tgc=row["TGC"],
                partial=row["mean_partial_pass"],
                errors=row["execution_failed_count"],
                e401=row["http_401_count"],
                e422=row["http_422_count"],
                name=row["name_error_count"],
                docs=row["repeated_api_doc_calls"],
                trunc=_pct(row["context_truncation_rate"]),
                waste=row["wasteful_success_count"],
            )
        )

    lines.extend(
        [
            "",
            "### Behavioral completion signals",
            "",
            "The first-API reference rate asks whether the first non-doc/non-supervisor API agrees with at least one successful trajectory for that same task. It is undefined for tasks with no successful reference.",
            "",
            "| Model | First API matches success reference | First API turn failed | First-error recovery | complete_task | Missing required mutation | Early irreversible heuristic | Still advancing at truncation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_stats:
        lines.append(
            f"| {row['model']} | {_pct(row['first_api_success_reference_rate'])} | "
            f"{_pct(row['first_api_failed_turn_rate'])} | {_pct(row['first_error_recovery_rate'])} | "
            f"{_pct(row['complete_task_rate'])} | {row['missing_final_operation_count']} | "
            f"{row['early_irreversible_episode_count']} | "
            f"{row['truncated_still_advancing_count']}/{row['context_truncated_count']} |"
        )
    lines.extend(
        [
            "",
            f"Across the five dev runs there are {total_successes} successful episodes; only {wasteful_successes} met the conservative wasteful-success flag. The improvement is therefore not explained by successful trajectories merely wandering longer.",
            "",
            "### Error causes seen in paired episodes",
            "",
            "- **401:** R0 repeatedly hard-coded or reused rejected credentials. On `d4e9306_1` it retried the identical Spotify login 25 times. The successful checkpoint-12 path obtains the Spotify password from supervisor, logs in once, and proceeds.",
            "- **422:** this combines schema mistakes and valid state conflicts. Examples include asking docs for nonexistent APIs, omitting the required `username` in `phone.login` (`530b157_2` R0), and repeatedly liking a song already liked (`57c3486_2` R0).",
            "- **NameError:** these are local state-management failures, not server failures: `venmo_login_result` was never assigned (`530b157_3` R0), `simple_note_access_token` was used before assignment (`6c2c621_1/2` checkpoint-12), and `unique_song` was referenced instead of `unique_song_ids` (`fac291d_3` R0).",
            "- Checkpoint-12 shows that an error need not doom the task: on `6c2c621_2`, it fixes the undefined note token on the next attempt, later replaces the hallucinated `write_file` API with `create_file`, and still reaches strict success.",
            "- The only early-irreversible heuristic hit is checkpoint-6 on `68ee2c9_3`: it invents one fixed date prefix and an extra directory, then starts moving files before establishing the required per-file date mapping. The evaluator confirms the rename/move end state is wrong.",
        ]
    )

    lines.extend(
        [
            "",
            "## Four failure-oriented task classes",
            "",
            "| Class | Tasks | D3 tasks |",
            "|---|---:|---:|",
        ]
    )
    for category in (
        "sft_failed_loop_succeeded",
        "sft_succeeded_loop_failed",
        "always_failed_partial_improved",
        "always_failed_no_progress",
        "stable_success",
    ):
        d3_count = sum(
            row["category"] == category and int(row["difficulty"]) == 3 for row in task_matrix
        )
        lines.append(f"| `{category}` | {counts[category]} | {d3_count} |")
    lines.append("")
    lines.append(
        "`stable_success` is shown separately because the requested four failure classes do not cover it."
    )

    lines.extend(
        [
            "",
            "## Same-task gains: where R0 and LOOP first diverge",
            "",
            "The API position is one-based in the business API sequence (docs and supervisor calls excluded).",
            "",
            "| Task | D | Checkpoint | R0 -> LOOP partial | First API divergence | Concrete correction |",
            "|---|---:|---|---:|---|---|",
        ]
    )
    gains = list(gain_pairs)
    gains.sort(key=lambda row: (int(str(row["checkpoint"]).split("-")[-1]), row["task_id"]))
    for row in gains:
        divergence = (
            f"#{int(row['common_business_api_prefix_length']) + 1}: "
            f"`{row['r0_divergence_api']}` -> `{row['loop_divergence_api']}`"
        )
        lines.append(
            f"| `{row['task_id']}` | {row['difficulty']} | {row['checkpoint']} | "
            f"{float(row['r0_partial']):.2f} -> {float(row['loop_partial']):.2f} | "
            f"{divergence} | {_short(str(row['fix_summary']), 210)} |"
        )

    lines.extend(
        [
            "",
            "### Representative paired mechanisms",
            "",
            f"There are {len(gain_pairs)} gain pairs covering {unique_gain_tasks} distinct tasks. Four representative same-task comparisons isolate what changed:",
            "",
            "1. **Authentication and escape from exact retries (`d4e9306_1`, R0 -> ckpt-12).** R0 makes 25 identical rejected Spotify login calls and never reaches task state. Ckpt-12 retrieves the stored credential, then paginates `show_liked_songs` and `show_liked_albums`, unions artist IDs, calls `follow_artist`, and completes.",
            "2. **Recover, then execute the cross-app plan (`6c2c621_2`, R0 -> ckpt-12).** R0 reads `file_system.login` documentation 33 times and never makes a business call. Ckpt-12 logs into both apps, creates the directory, recovers from a token NameError and a nonexistent `write_file` API, paginates every note, writes every `.md` file, and calls `complete_task()`.",
            "3. **Ground the entity and perform the final mutation (`396c5a2_2`, R0 -> ckpt-15).** R0 tries a fabricated artist ID, eventually finds `search_artists`, but keeps querying until truncation and never changes the queue. Ckpt-15 resolves Aria Sterling, paginates `search_songs(min_play_count=990)`, calls `add_to_queue` for all results, and completes.",
            "4. **Handle a state conflict instead of replaying it (`57c3486_2`, R0 -> ckpt-12).** R0 hits `422 already liked` and repeats the same three-API block until truncation. Ckpt-12 enumerates followed artists and their songs, performs the required likes, and finishes in ten turns.",
        ]
    )

    lines.extend(
        [
            "",
            "## Same-task regressions",
            "",
            "| Task | D | Checkpoint | R0 -> LOOP partial | First API divergence | LOOP first error |",
            "|---|---:|---|---:|---|---|",
        ]
    )
    regressions = [row for row in pair_rows if row["transition"] == "regression"]
    for row in regressions:
        lines.append(
            f"| `{row['task_id']}` | {row['difficulty']} | {row['checkpoint']} | "
            f"{float(row['r0_partial']):.2f} -> {float(row['loop_partial']):.2f} | "
            f"`{row['r0_divergence_api']}` -> `{row['loop_divergence_api']}` | "
            f"{row['loop_first_error_type'] or 'none'} at turn {row['loop_first_error_turn'] or 'n/a'}: "
            f"{_short(str(row['loop_first_error_cause']), 150)} |"
        )
    lines.extend(
        [
            "",
            "The regressions have the opposite shape. On `d4e9306_2`, R0 reads both liked songs and liked albums and follows the union; checkpoint-15 repeatedly re-reads only `show_liked_songs`, never queries albums, never mutates, and truncates. On `df61dc5_2`, R0 paginates transactions then likes the filtered set; checkpoint-6 keeps paginating/reissuing `show_transactions` and never reaches `like_transaction`.",
        ]
    )

    no_progress_d3 = [
        row
        for row in task_matrix
        if row["category"] == "always_failed_no_progress" and int(row["difficulty"]) == 3
    ]
    lines.extend(
        [
            "",
            "## D3 deep dive",
            "",
            "The table first isolates the D3 no-progress subset; the bullets then cover all three D3 tasks.",
            "",
            "| Task | R0/6/9/12/15 partial | First APIs (R0 -> 15) |",
            "|---|---:|---|",
        ]
    )
    for row in no_progress_d3:
        partials = "/".join(
            f"{float(row[key]):.2f}"
            for key in (
                "r0_partial",
                "ckpt_6_partial",
                "ckpt_9_partial",
                "ckpt_12_partial",
                "ckpt_15_partial",
            )
        )
        lines.append(
            f"| `{row['task_id']}` | {partials} | `{row['r0_first_api']}` -> `{row['ckpt_15_first_api']}` |"
        )
    lines.extend(
        [
            "",
            "D3 is a clear grounding bottleneck, not merely an API-syntax bottleneck:",
            "",
            "- `530b157_1`: every model remains at 0.10. They only retrieve phone data, never switch to Venmo, never send money or the final text, and all truncate. Later checkpoints often repeat `search_text_messages` without consuming the result.",
            "- `530b157_2`: checkpoint-9 rises to 0.40 by sending the phone message and completing, but it never creates the Venmo transaction. The other checkpoints stall in retrieval; checkpoint-12 ends with a generated-variable NameError.",
            "- `530b157_3`: checkpoint-9 reaches 0.80 and is the strongest D3 behavior. It logs into both apps, creates a Venmo transaction, sends a text, and completes. It still hard-codes the grocery amount and sends the text to the account owner's phone rather than the friend inferred from history, so exactly the amount and receiver requirements fail. This is evidence that LOOP learned decomposition and execution before it learned reliable value grounding from conversation history.",
        ]
    )

    lines.extend(
        [
            "",
            "## Training group diversity",
            "",
            "| Iter | Same-return groups | all-success/all-failure/partial-same | Unique returns/group | Effective rollout/token | Business API seq/group | Error-only seq | Truncation | Reports | Same reward, different pass set |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in training:
        lines.append(
            f"| {row['iteration']} | {row['all_returns_same_group_count']}/24 ({_pct(row['all_returns_same_group_rate'])}) | "
            f"{row['all_success_group_count']}/{row['all_failure_group_count']}/{row['partial_same_group_count']} | "
            f"{_number(row['mean_unique_return_count'])} | "
            f"{_pct(row['effective_advantage_rollout_ratio'])}/{_pct(row['effective_advantage_token_ratio'])} | "
            f"{_number(row['mean_unique_business_api_sequence_count'])} | "
            f"{_pct(row['error_only_business_sequence_ratio'])} | "
            f"{_pct(row['context_truncation_ratio'])} | "
            f"{row['evaluator_report_count']}/144 ({_pct(row['evaluator_report_coverage'])}) | "
            f"{row['same_reward_different_pass_group_count']}/24 lower bound; "
            f"{_pct(row['same_reward_different_pass_rate_among_comparable_groups'])} observed |"
        )

    lines.extend(
        [
            "",
            "### Within-group paired examples",
            "",
            "- Iteration 1, scenario 13 (`07b42fd_1`): after login, the successful rollout goes to `spotify.search_artists`; a failed rollout goes to `spotify.show_artist_following`. The group contains returns 0.2/0.4/0.6/1.0, so the initial SFT policy already exposes a useful API-choice contrast.",
            "- Iteration 1, scenario 6 (`2a163ab_2`): the successful branch starts with Venmo while the lowest-return branch starts with Phone and accumulates repeated auth failures. The first application choice is already predictive of return.",
            "- Iteration 15, scenario 0 (`07b42fd_2`): five rollouts succeed. Their short path ends after `login -> search_artists -> follow_artist`; the 0.8 rollout continues by logging in again and wandering. Here the useful distinction is knowing when to stop, not discovering another API.",
            "- Equal reward can hide different state: iteration 15 task `e3d6c94_3` has two 7/9 reports, but one passes the genre-song membership requirement while the other passes the recent-song membership requirement. Iteration 1 task `07b42fd_2` likewise has two 2/5 reports with different passed requirements.",
        ]
    )

    first = training[0]
    last = training[-1]
    lines.extend(
        [
            "",
            "## What the paired evidence says",
            "",
            f"- Learning signal remained available: effective-advantage rollout ratio changed from {_pct(first['effective_advantage_rollout_ratio'])} at iteration {first['iteration']} to {_pct(last['effective_advantage_rollout_ratio'])} at iteration {last['iteration']}; mean unique returns/group changed from {_number(first['mean_unique_return_count'])} to {_number(last['mean_unique_return_count'])}.",
            f"- API-path diversity changed from {_number(first['mean_unique_business_api_sequence_count'])} to {_number(last['mean_unique_business_api_sequence_count'])} unique business sequences/group.",
            f"- Not all diversity was useful: the share of distinct API sequences seen only in error/truncated rollouts was {_pct(first['error_only_business_sequence_ratio'])} at iteration {first['iteration']} and {_pct(last['error_only_business_sequence_ratio'])} at iteration {last['iteration']}.",
            f"- Rollout quality improved while raw path diversity contracted: strict success rose from {_pct(first['strict_success_rate'])} to {_pct(last['strict_success_rate'])}, mean partial from {_number(first['average_partial_pass'], 3)} to {_number(last['average_partial_pass'], 3)}, execution failures/rollout fell from {_number(first['execution_failed_per_rollout'])} to {_number(last['execution_failed_per_rollout'])}, and truncation fell from {_pct(first['context_truncation_ratio'])} to {_pct(last['context_truncation_ratio'])}.",
            "- Therefore the answer is **yes, SFT initialization produced ample groupwise learning signal**, but **no, LOOP did not mainly create more raw behavioral diversity**. Iteration 1 was already maximally diverse at 6.00 business paths/group, with 22/24 non-constant-return groups and 87.5% effective-advantage rollouts. Training mostly converted error-driven diversity into more competent, shorter alternatives while preserving return variation.",
            "- Dev gains and regressions must be read from the paired rows above. Aggregate error counts alone cannot identify the learned correction.",
            "- Equal scalar reward does not imply equal evaluator state. The report table gives the observed lower bound; missing runner reports are not treated as identical pass sets.",
            "",
            "## Files",
            "",
            "- `dev_task_matrix.csv`: one row per task across all five models.",
            "- `dev_pair_transitions.csv`: all 228 R0-to-checkpoint pairs and their first divergence.",
            "- `dev_trace_details.csv`: per-episode errors, recovery, docs, pagination, final action, truncation, and detour fields.",
            "- `dev_error_events.csv`: every observed 401/422/NameError/runtime error turn with a redacted cause excerpt.",
            "- `training_iteration_summary.csv`: five complete iteration summaries.",
            "- `training_group_details.csv`: all 120 scenario groups and success/failure divergence.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-root",
        type=Path,
        default=Path("artifacts/sft_loop15/trajectory_analysis_20260720/raw_dev"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("experiments/qwen25_7b_d12_100x1_loop15_lora16/r1_loop15_seed20260718"),
    )
    parser.add_argument(
        "--training-report-root",
        type=Path,
        default=Path("artifacts/sft_loop15/trajectory_analysis_20260720/raw_training_reports"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sft_loop15/trajectory_analysis_20260720"),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERATIONS),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    task_matrix, pair_rows, trace_rows, model_stats = analyze_dev(args.dev_root)
    training, group_details = analyze_training(
        args.run_root, args.training_report_root, args.iterations
    )

    _write_csv(output_dir / "dev_task_matrix.csv", task_matrix)
    _write_csv(output_dir / "dev_pair_transitions.csv", pair_rows)
    _write_csv(output_dir / "dev_trace_details.csv", trace_rows)
    error_rows = [
        {
            "model": trace["label"],
            "task_id": trace["task_id"],
            "difficulty": trace["difficulty"],
            "success": trace["success"],
            "partial_pass": trace["partial_pass"],
            **event,
        }
        for trace in trace_rows
        for event in trace["error_events"]
    ]
    _write_csv(output_dir / "dev_error_events.csv", error_rows)
    _write_csv(output_dir / "dev_model_summary.csv", model_stats)
    _write_csv(output_dir / "training_iteration_summary.csv", training)
    _write_csv(output_dir / "training_group_details.csv", group_details)
    (output_dir / "report.md").write_text(
        _report_markdown(task_matrix, pair_rows, trace_rows, model_stats, training, group_details),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dev_tasks": len(task_matrix),
                "pairs": len(pair_rows),
                "training_groups": len(group_details),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
