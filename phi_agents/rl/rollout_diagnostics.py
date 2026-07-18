"""Pure AppWorld rollout diagnostics and privacy-conscious trajectory persistence.

This module deliberately has no trainer, worker, or evaluator side effects.  It operates on
already-completed grouped rollouts, making it suitable both for training-time summaries and for
fixed diagnostic rollouts that must never update the policy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from phi_agents.rl.rl_utils import Baseline, compute_loop_advantages

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from phi_agents.evals.appworld_rollout_data import AppWorldTrainingRollout

SCHEMA_VERSION = "appworld-rollout-diagnostics-v1"
SANITIZED_TRAJECTORY_SCHEMA_VERSION = "appworld-sanitized-trajectory-v1"

CODE_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", flags=re.IGNORECASE | re.DOTALL)
PARTIAL_CODE_RE = re.compile(r"```(?:python|py)\s*\n?(.*)$", flags=re.IGNORECASE | re.DOTALL)
API_CALL_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DOC_RE = re.compile(r"\bapis\.api_docs\.show_api_doc\s*\(", flags=re.IGNORECASE)
DOC_DESCRIPTION_RE = re.compile(
    r"\bapis\.api_docs\.show_api_descriptions\s*\(", flags=re.IGNORECASE
)
HTTP_401_RE = re.compile(r"(?<!\d)401(?!\d)")
HTTP_422_RE = re.compile(r"(?<!\d)422(?!\d)")
NAME_ERROR_RE = re.compile(r"\bNameError\b", flags=re.IGNORECASE)
EXECUTION_FAILED_TEXT = "Execution failed."
INVALID_API_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"invalid api",
        r"api .* not found",
        r"no api named",
        r"unknown api",
        r"attributeerror: .*apis\.",
        r"has no attribute",
    )
)

# These expressions remove common credentials without altering ordinary AppWorld observations.
# They intentionally do not redact task-relevant names, email addresses, or phone numbers from the
# visible conversation; callers that require PII removal should apply a domain-specific policy.
SECRET_HEADER_RE = re.compile(r"(?im)^(?P<prefix>\s*(?:authorization|cookie|set-cookie)\s*:\s*).+$")
QUOTED_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>[\"'](?:password|passwd|api[_-]?key|token|auth[_-]?token|"
    r"access[_-]?token|refresh[_-]?token|authorization|cookie|webhook(?:_url)?)[\"']\s*:\s*)"
    r"(?P<value>[\"'][^\"']*[\"']|[^,}\]\s]+)"
)
ASSIGNED_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:password|passwd|api[_-]?key|token|auth[_-]?token|"
    r"access[_-]?token|refresh[_-]?token|authorization|cookie|webhook(?:_url)?)\b\s*=\s*)"
    r"(?P<value>[\"'][^\"']*[\"']|[^,;}\]\s]+)"
)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _partial_pass(rollout: AppWorldTrainingRollout) -> float:
    result = rollout.appworld_rollout_data.eval_result
    return _safe_ratio(len(result.passes), result.num_tests)


def _output_token_count(rollout: AppWorldTrainingRollout) -> int:
    return sum(bool(is_output) for is_output in rollout.policy_token_info.is_output)


def _generation_seed(rollout: AppWorldTrainingRollout) -> int | None:
    seed = getattr(rollout, "generation_seed", None)
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"invalid generation_seed on rollout: {seed!r}")
    return int(seed)


def _indexed_group(
    group: Sequence[AppWorldTrainingRollout], *, scenario_idx: int
) -> list[tuple[int, AppWorldTrainingRollout]]:
    indexed: list[tuple[int, AppWorldTrainingRollout]] = []
    for fallback_idx, rollout in enumerate(group):
        rollout_idx = getattr(rollout, "rollout_idx", fallback_idx)
        if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, Integral):
            raise ValueError(
                f"scenario group {scenario_idx} has invalid rollout_idx: {rollout_idx!r}"
            )
        rollout_idx = int(rollout_idx)
        if rollout_idx < 0 or rollout_idx >= len(group):
            raise ValueError(
                f"scenario group {scenario_idx} rollout_idx {rollout_idx} is outside "
                f"the expected range 0..{len(group) - 1}"
            )
        indexed.append((rollout_idx, rollout))

    indices = [rollout_idx for rollout_idx, _rollout in indexed]
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"scenario group {scenario_idx} contains duplicate rollout_idx values: {indices}"
        )
    return sorted(indexed, key=lambda item: item[0])


def _extract_code_blocks(text: str) -> list[str]:
    blocks = [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text)]
    last_end = max((match.end() for match in CODE_BLOCK_RE.finditer(text)), default=0)
    partial = PARTIAL_CODE_RE.search(text[last_end:])
    if partial and (partial_code := partial.group(1).strip()):
        blocks.append(partial_code)
    return blocks


def _action_text(message: Any) -> str:
    content = str(getattr(message, "content", "") or "")
    blocks = _extract_code_blocks(content)
    if blocks:
        return "\n".join(blocks)
    return content


def _normalize_action(message: Any) -> str:
    return " ".join(_action_text(message).split())


def _api_sequence(messages: Sequence[Any]) -> tuple[str, ...]:
    endpoints: list[str] = []
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        endpoints.extend(
            f"{app_name}.{api_name}"
            for app_name, api_name in API_CALL_RE.findall(_action_text(message))
        )
    return tuple(endpoints)


def _business_api_sequence(api_sequence: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        endpoint
        for endpoint in api_sequence
        if not endpoint.startswith("api_docs.") and not endpoint.startswith("supervisor.")
    )


def _task_messages(rollout: AppWorldTrainingRollout) -> list[Any]:
    prompt_count = rollout.appworld_rollout_data.num_prompt_messages or 0
    return list(rollout.messages[prompt_count:])


def _turns(rollout: AppWorldTrainingRollout) -> list[dict[str, Any]]:
    messages = _task_messages(rollout)
    turns: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if getattr(message, "role", None) != "assistant":
            continue
        observation = ""
        for following in messages[index + 1 :]:
            role = getattr(following, "role", None)
            if role == "assistant":
                break
            if role in {"user", "ipython"}:
                observation = str(getattr(following, "content", "") or "")
                break
        action = _action_text(message)
        turns.append(
            {
                "message": message,
                "action": action,
                "normalized_action": _normalize_action(message),
                "observation": observation,
                "execution_failed": EXECUTION_FAILED_TEXT in observation,
                "api_calls": tuple(
                    f"{app_name}.{api_name}" for app_name, api_name in API_CALL_RE.findall(action)
                ),
            }
        )
    return turns


def _nonnegative_metadata_count(value: int, fallback: int) -> int:
    return int(value) if value >= 0 else fallback


def _rollout_behavior(rollout: AppWorldTrainingRollout) -> dict[str, Any]:
    turns = _turns(rollout)
    observations = [str(turn["observation"]) for turn in turns]
    observation_text = "\n".join(observations)
    parsed_execution_failures = sum(bool(turn["execution_failed"]) for turn in turns)
    execution_failed_count = _nonnegative_metadata_count(
        rollout.appworld_rollout_data.n_execution_failed, parsed_execution_failures
    )
    parsed_no_code = sum(not str(turn["action"]).strip() for turn in turns)
    no_code_found_count = _nonnegative_metadata_count(
        rollout.appworld_rollout_data.n_no_code_found, parsed_no_code
    )

    repeated_failed_actions = 0
    previous_failed_action: str | None = None
    failed_api_calls = 0
    recovered_api_calls = 0
    pending_failed: Counter[str] = Counter()
    for turn in turns:
        normalized_action = str(turn["normalized_action"])
        if turn["execution_failed"]:
            if normalized_action and normalized_action == previous_failed_action:
                repeated_failed_actions += 1
            previous_failed_action = normalized_action
            for endpoint in set(turn["api_calls"]):
                failed_api_calls += 1
                pending_failed[endpoint] += 1
        else:
            previous_failed_action = None
            for endpoint in set(turn["api_calls"]):
                if pending_failed[endpoint]:
                    recovered_api_calls += pending_failed[endpoint]
                    pending_failed[endpoint] = 0

    action_text = "\n".join(str(turn["action"]) for turn in turns)
    api_sequence = _api_sequence(_task_messages(rollout))
    strict_success = bool(rollout.appworld_rollout_data.eval_result.success)
    return {
        "turn_count": len(turns),
        "execution_failed_count": execution_failed_count,
        "no_code_found_count": no_code_found_count,
        "http_401_count": len(HTTP_401_RE.findall(observation_text)),
        "http_422_count": len(HTTP_422_RE.findall(observation_text)),
        "name_error_count": len(NAME_ERROR_RE.findall(observation_text)),
        "invalid_api_hits": sum(
            len(pattern.findall(observation_text)) for pattern in INVALID_API_PATTERNS
        ),
        "api_doc_calls": len(DOC_RE.findall(action_text)),
        "api_description_calls": len(DOC_DESCRIPTION_RE.findall(action_text)),
        "consecutive_repeated_failed_action_count": repeated_failed_actions,
        "failed_api_calls": failed_api_calls,
        "recovered_api_calls": recovered_api_calls,
        "context_truncated": bool(rollout.appworld_rollout_data.context_truncated),
        "strict_success": strict_success,
        "partial_pass": _partial_pass(rollout),
        "cancelled": bool(rollout.cancelled),
        "output_token_count": _output_token_count(rollout),
        "api_sequence": list(api_sequence),
        "business_api_sequence": list(_business_api_sequence(api_sequence)),
    }


def _unique_float_count(values: Sequence[float], tolerance: float) -> int:
    if not values:
        return 0
    count = 1
    previous = sorted(values)[0]
    for value in sorted(values)[1:]:
        if not math.isclose(value, previous, rel_tol=0.0, abs_tol=tolerance):
            count += 1
            previous = value
    return count


def _same_sequence_pair_rate(sequences: Sequence[tuple[str, ...]]) -> float:
    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    matching_pairs = sum(count * (count - 1) // 2 for count in Counter(sequences).values())
    return _safe_ratio(matching_pairs, total_pairs)


def _duplicated_sequence_rollout_ratio(sequences: Sequence[tuple[str, ...]]) -> float:
    counts = Counter(sequences)
    duplicated_rollouts = sum(count for count in counts.values() if count > 1)
    return _safe_ratio(duplicated_rollouts, len(sequences))


def _zero_return_class(
    returns: Sequence[float], successes: Sequence[bool], *, tolerance: float
) -> str:
    if float(np.std(returns)) > tolerance:
        return "non_zero_variance"
    if all(successes):
        return "all_success"
    if all(math.isclose(ret, 0.0, rel_tol=0.0, abs_tol=tolerance) for ret in returns):
        return "all_failure"
    return "partial_same"


def compute_rollout_diagnostics(
    grouped_rollouts: Sequence[Sequence[AppWorldTrainingRollout]],
    *,
    baseline: Baseline | str = Baseline.LOO,
    adv_normalization: bool = False,
    abs_adv_threshold: float = 0.01,
    pos_adv_only: bool = False,
    zero_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compute grouped return, advantage, API-diversity, and behavior diagnostics.

    Effective advantage follows the trainer's threshold semantics but intentionally excludes its
    distributed batch-rounding behavior.  Effective token ratio is measured over policy output
    tokens only.
    """
    if not grouped_rollouts:
        raise ValueError("grouped_rollouts must contain at least one scenario group")
    if abs_adv_threshold < 0:
        raise ValueError("abs_adv_threshold must be non-negative")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")

    group_rows: list[dict[str, Any]] = []
    all_rollout_rows: list[dict[str, Any]] = []
    total_effective_rollouts = 0
    total_output_tokens = 0
    effective_output_tokens = 0

    for scenario_idx, group in enumerate(grouped_rollouts):
        if len(group) < 2:
            raise ValueError(
                f"scenario group {scenario_idx} has {len(group)} rollout(s); at least two are required"
            )
        indexed_group = _indexed_group(group, scenario_idx=scenario_idx)
        ordered_group = [rollout for _rollout_idx, rollout in indexed_group]
        task_ids = {rollout.appworld_rollout_data.task.task_id for rollout in ordered_group}
        if len(task_ids) != 1:
            raise ValueError(
                f"scenario group {scenario_idx} contains multiple task ids: {sorted(task_ids)}"
            )

        returns = [float(rollout.ret) for rollout in ordered_group]
        successes = [
            bool(rollout.appworld_rollout_data.eval_result.success) for rollout in ordered_group
        ]
        advantages = compute_loop_advantages(
            [ordered_group], baseline=baseline, adv_normalization=adv_normalization
        )
        abs_advantages = advantages if pos_adv_only else np.abs(advantages)
        effective = [bool(value >= abs_adv_threshold) for value in abs_advantages]
        behaviors = [_rollout_behavior(rollout) for rollout in ordered_group]
        api_sequences = [tuple(row["api_sequence"]) for row in behaviors]
        business_sequences = [tuple(row["business_api_sequence"]) for row in behaviors]

        rollout_rows: list[dict[str, Any]] = []
        for (rollout_idx, rollout), advantage, is_effective, behavior in zip(
            indexed_group, advantages, effective, behaviors, strict=True
        ):
            output_tokens = int(behavior["output_token_count"])
            total_output_tokens += output_tokens
            if is_effective:
                total_effective_rollouts += 1
                effective_output_tokens += output_tokens
            row = {
                "scenario_idx": scenario_idx,
                "rollout_idx": rollout_idx,
                "task_id": rollout.appworld_rollout_data.task.task_id,
                "return": float(rollout.ret),
                "generation_seed": _generation_seed(rollout),
                "advantage": float(advantage),
                "effective_advantage": is_effective,
                **behavior,
            }
            rollout_rows.append(row)
            all_rollout_rows.append(row)

        return_std = float(np.std(returns))
        zero_class = _zero_return_class(returns, successes, tolerance=zero_tolerance)
        group_rows.append(
            {
                "scenario_idx": scenario_idx,
                "task_id": next(iter(task_ids)),
                "rollout_count": len(ordered_group),
                "returns": returns,
                "return_mean": _mean(returns),
                "return_std": return_std,
                "return_range": max(returns) - min(returns),
                "unique_return_count": _unique_float_count(returns, zero_tolerance),
                "zero_return_std": return_std <= zero_tolerance,
                "zero_return_class": zero_class,
                "strict_success_count": sum(successes),
                "effective_advantage_rollout_count": sum(effective),
                "effective_advantage_rollout_ratio": _safe_ratio(
                    sum(effective), len(ordered_group)
                ),
                "unique_api_sequence_count": len(set(api_sequences)),
                "unique_business_api_sequence_count": len(set(business_sequences)),
                "empty_business_api_sequence_count": sum(
                    not sequence for sequence in business_sequences
                ),
                "same_business_api_sequence_ratio": _same_sequence_pair_rate(business_sequences),
                "duplicated_business_api_sequence_rollout_ratio": (
                    _duplicated_sequence_rollout_ratio(business_sequences)
                ),
                "rollouts": rollout_rows,
            }
        )

    classification_counts = Counter(row["zero_return_class"] for row in group_rows)
    strict_success_rows = [row for row in all_rollout_rows if row["strict_success"]]
    error_rows = [row for row in all_rollout_rows if row["execution_failed_count"] > 0]
    behavior = {
        "execution_failed_count": sum(row["execution_failed_count"] for row in all_rollout_rows),
        "no_code_found_count": sum(row["no_code_found_count"] for row in all_rollout_rows),
        "http_401_count": sum(row["http_401_count"] for row in all_rollout_rows),
        "http_422_count": sum(row["http_422_count"] for row in all_rollout_rows),
        "name_error_count": sum(row["name_error_count"] for row in all_rollout_rows),
        "invalid_api_hits": sum(row["invalid_api_hits"] for row in all_rollout_rows),
        "api_doc_calls": sum(row["api_doc_calls"] for row in all_rollout_rows),
        "api_description_calls": sum(row["api_description_calls"] for row in all_rollout_rows),
        "consecutive_repeated_failed_action_count": sum(
            row["consecutive_repeated_failed_action_count"] for row in all_rollout_rows
        ),
        "average_turn_count": _mean([row["turn_count"] for row in all_rollout_rows]),
        "context_truncation_ratio": _mean(
            [int(row["context_truncated"]) for row in all_rollout_rows]
        ),
        "failed_api_calls": sum(row["failed_api_calls"] for row in all_rollout_rows),
        "recovered_api_calls": sum(row["recovered_api_calls"] for row in all_rollout_rows),
        "strict_success_count": len(strict_success_rows),
        "strict_success_rate": _safe_ratio(len(strict_success_rows), len(all_rollout_rows)),
        "average_partial_pass": _mean([row["partial_pass"] for row in all_rollout_rows]),
        "cancelled_count": sum(row["cancelled"] for row in all_rollout_rows),
        "average_execution_errors_before_strict_success": _mean(
            [row["execution_failed_count"] for row in strict_success_rows]
        ),
        "error_rollout_recovery_success_rate": _safe_ratio(
            sum(bool(row["strict_success"]) for row in error_rows), len(error_rows)
        ),
    }
    behavior["error_recovery_success_rate"] = _safe_ratio(
        behavior["recovered_api_calls"], behavior["failed_api_calls"]
    )
    num_rollouts = len(all_rollout_rows)
    behavior.update(
        {
            "execution_failed_per_rollout": _safe_ratio(
                behavior["execution_failed_count"], num_rollouts
            ),
            "no_code_found_per_rollout": _safe_ratio(behavior["no_code_found_count"], num_rollouts),
            "api_doc_calls_per_rollout": _safe_ratio(behavior["api_doc_calls"], num_rollouts),
            "api_description_calls_per_rollout": _safe_ratio(
                behavior["api_description_calls"], num_rollouts
            ),
        }
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "baseline": str(Baseline(baseline)),
            "adv_normalization": adv_normalization,
            "abs_adv_threshold": abs_adv_threshold,
            "pos_adv_only": pos_adv_only,
            "zero_tolerance": zero_tolerance,
            "effective_token_definition": "policy output tokens on effective-advantage rollouts",
            "recovery_definition": (
                "failed API endpoint called later in a non-execution-failed turn"
            ),
            "same_business_api_sequence_definition": "matching unordered rollout pairs",
        },
        "num_groups": len(group_rows),
        "num_rollouts": num_rollouts,
        "zero_return_std_group_count": sum(row["zero_return_std"] for row in group_rows),
        "zero_return_std_group_rate": _mean([int(row["zero_return_std"]) for row in group_rows]),
        "zero_return_group_class_counts": {
            key: int(classification_counts.get(key, 0))
            for key in ("all_success", "all_failure", "partial_same", "non_zero_variance")
        },
        "mean_unique_return_count_per_group": _mean(
            [row["unique_return_count"] for row in group_rows]
        ),
        "mean_return_range_per_group": _mean([row["return_range"] for row in group_rows]),
        "mean_unique_api_sequence_count_per_group": _mean(
            [row["unique_api_sequence_count"] for row in group_rows]
        ),
        "mean_unique_business_api_sequence_count_per_group": _mean(
            [row["unique_business_api_sequence_count"] for row in group_rows]
        ),
        "same_business_api_sequence_ratio": _mean(
            [row["same_business_api_sequence_ratio"] for row in group_rows]
        ),
        "duplicated_business_api_sequence_rollout_ratio": _mean(
            [row["duplicated_business_api_sequence_rollout_ratio"] for row in group_rows]
        ),
        "effective_advantage_rollout_count": total_effective_rollouts,
        "effective_advantage_rollout_ratio": _safe_ratio(total_effective_rollouts, num_rollouts),
        "total_output_tokens": total_output_tokens,
        "effective_advantage_output_tokens": effective_output_tokens,
        "effective_advantage_token_ratio": _safe_ratio(
            effective_output_tokens, total_output_tokens
        ),
        "behavior": behavior,
        "groups": group_rows,
    }


def _redact_sensitive_text(text: str) -> str:
    text = SECRET_HEADER_RE.sub(lambda match: match.group("prefix") + "[REDACTED]", text)
    text = QUOTED_SECRET_RE.sub(lambda match: match.group("prefix") + '"[REDACTED]"', text)
    return ASSIGNED_SECRET_RE.sub(lambda match: match.group("prefix") + '"[REDACTED]"', text)


def _sanitized_messages(rollout: AppWorldTrainingRollout) -> list[dict[str, str]]:
    return [
        {
            "role": str(getattr(message, "role", "unknown")),
            "content": _redact_sensitive_text(str(getattr(message, "content", "") or "")),
        }
        for message in rollout.messages
    ]


def sanitized_trajectory_payload(
    rollout: AppWorldTrainingRollout,
    *,
    iteration: int,
    scenario_idx: int,
    rollout_idx: int,
) -> dict[str, Any]:
    """Build a JSON-safe trajectory without evaluator details or token/log-probability arrays."""
    if min(iteration, scenario_idx, rollout_idx) < 0:
        raise ValueError("iteration, scenario_idx, and rollout_idx must be non-negative")
    data = rollout.appworld_rollout_data
    behavior = _rollout_behavior(rollout)
    return {
        "schema_version": SANITIZED_TRAJECTORY_SCHEMA_VERSION,
        "iteration": iteration,
        "scenario_idx": scenario_idx,
        "rollout_idx": rollout_idx,
        "task_id": data.task.task_id,
        "dataset_name": data.dataset_name,
        "messages": _sanitized_messages(rollout),
        "metadata": {
            "difficulty": data.eval_result.difficulty,
            "return": float(rollout.ret),
            "strict_success": bool(data.eval_result.success),
            "partial_pass": _partial_pass(rollout),
            "num_interactions": data.eval_result.num_interactions,
            "elapsed_seconds": float(rollout.elapsed),
            "cancelled": bool(rollout.cancelled),
            "execution_failed_count": behavior["execution_failed_count"],
            "no_code_found_count": behavior["no_code_found_count"],
            "http_401_count": behavior["http_401_count"],
            "http_422_count": behavior["http_422_count"],
            "name_error_count": behavior["name_error_count"],
            "invalid_api_hits": behavior["invalid_api_hits"],
            "api_doc_calls": behavior["api_doc_calls"],
            "api_description_calls": behavior["api_description_calls"],
            "consecutive_repeated_failed_action_count": behavior[
                "consecutive_repeated_failed_action_count"
            ],
            "context_truncated": behavior["context_truncated"],
            "output_token_count": behavior["output_token_count"],
            "visible_message_count": len(rollout.messages),
            "generation_seed": _generation_seed(rollout),
        },
    }


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
        file_handle.write("\n")


def save_sanitized_trajectories(
    grouped_rollouts: Sequence[Sequence[AppWorldTrainingRollout]],
    *,
    output_root: Path,
    iteration: int,
    overwrite: bool = False,
) -> list[Path]:
    """Save each rollout under iteration/scenario/rollout without cross-rollout overwrites."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    indexed_groups = [
        _indexed_group(group, scenario_idx=scenario_idx)
        for scenario_idx, group in enumerate(grouped_rollouts)
    ]
    paths = [
        output_root
        / f"iteration-{iteration:06d}"
        / f"scenario-{scenario_idx:04d}"
        / f"rollout-{rollout_idx:02d}"
        / "trajectory.json"
        for scenario_idx, indexed_group in enumerate(indexed_groups)
        for rollout_idx, _rollout in indexed_group
    ]
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing trajectory: {existing[0]}")

    saved_paths: list[Path] = []
    for scenario_idx, indexed_group in enumerate(indexed_groups):
        for rollout_idx, rollout in indexed_group:
            path = (
                output_root
                / f"iteration-{iteration:06d}"
                / f"scenario-{scenario_idx:04d}"
                / f"rollout-{rollout_idx:02d}"
                / "trajectory.json"
            )
            payload = sanitized_trajectory_payload(
                rollout,
                iteration=iteration,
                scenario_idx=scenario_idx,
                rollout_idx=rollout_idx,
            )
            _write_json(path, payload, overwrite=overwrite)
            saved_paths.append(path)
    return saved_paths


def save_rollout_diagnostics(
    diagnostics: dict[str, Any], *, output_path: Path, overwrite: bool = False
) -> Path:
    """Persist a diagnostics dictionary as JSON."""
    _write_json(output_path, diagnostics, overwrite=overwrite)
    return output_path
