#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

import requests
from jinja2 import Template
from openai import OpenAI

from phi_agents.agent.react_template import APP_DESCRIPTIONS, PROMPT_TEMPLATE
from phi_agents.appworld.interface import (
    AppWorldExecutionError,
    AppWorldInitializeTaskError,
    AppWorldInterface,
    AppWorldTaskEvalResult,
    Task,
    load_task_ids,
)
from phi_agents.utils.appworld import extract_code_format_output
from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_OUTPUT_ROOT = Path("artifacts/deepseek_appworld_dev")
DEV_SPLIT_NAME = "dev"
DEV_SPLIT_PATH = Path("data/appworld_splits/dev.txt")
SUPPORTED_SPLIT_PATHS: dict[str, Path] = {
    DEV_SPLIT_NAME: DEV_SPLIT_PATH,
    "test_normal": Path("data/appworld_splits/test_normal.txt"),
    "test_challenge": Path("data/appworld_splits/test_challenge.txt"),
}
DEFAULT_EXPERIMENT_PREFIX = "deepseek_v4_appworld_dev"
DEFAULT_COST_LIMIT_CNY = 20.0

ThinkingOverride = Literal["profile", "enabled", "disabled"]
ReasoningEffortOverride = Literal["profile", "high", "max", "none"]


class DeepSeekRequestError(RuntimeError):
    """Raised when a DeepSeek API request fails after retries."""


class CostLimitExceeded(RuntimeError):
    """Raised when the estimated DeepSeek spend reaches the configured run limit."""


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    today_date: date | None = None
    stopped_by_max_tokens_limit: bool = False

    def asdict(self) -> dict[str, Any]:
        message_dict: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.today_date is not None:
            message_dict["today_date"] = self.today_date.strftime("%Y-%m-%d")
        if self.role == "assistant":
            message_dict["stopped_by_max_tokens_limit"] = self.stopped_by_max_tokens_limit
        return message_dict


@dataclass(frozen=True)
class TaskEvalResult:
    success: bool
    difficulty: int
    num_tests: int
    passes: list[Any]
    failures: list[Any]
    num_interactions: int

    @classmethod
    def create(
        cls,
        appworld_task_eval_result: AppWorldTaskEvalResult,
        num_interactions: int,
    ) -> TaskEvalResult:
        return cls(
            success=appworld_task_eval_result.success,
            difficulty=appworld_task_eval_result.difficulty,
            num_tests=appworld_task_eval_result.num_tests,
            passes=appworld_task_eval_result.passes,
            failures=appworld_task_eval_result.failures,
            num_interactions=num_interactions,
        )


def execution_failed(appworld_output: str) -> bool:
    return "Execution failed." in appworld_output


def no_code_found(code: str) -> bool:
    return code == ""


@dataclass(frozen=True)
class DeepSeekPricingCny:
    input_cache_hit_per_million: float
    input_cache_miss_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class CostEstimate:
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    completion_tokens: int
    cost_cny: float

    def asdict(self) -> dict[str, int | float]:
        return asdict(self)


PRICING_CNY_BY_MODEL_FAMILY: dict[str, DeepSeekPricingCny] = {
    "flash": DeepSeekPricingCny(
        input_cache_hit_per_million=0.02,
        input_cache_miss_per_million=1.0,
        output_per_million=2.0,
    ),
    "pro": DeepSeekPricingCny(
        input_cache_hit_per_million=0.025,
        input_cache_miss_per_million=3.0,
        output_per_million=6.0,
    ),
}


@dataclass(frozen=True)
class DeepSeekProfile:
    name: str
    model: str
    thinking_enabled: bool | None
    reasoning_effort: Literal["high", "max"] | None
    base_url: str = DEFAULT_BASE_URL
    temperature: float | None = 0.1
    max_tokens: int = 4096
    max_retries: int = 5
    request_timeout_seconds: float = 180.0
    request_interval_seconds: float = 1.0
    retry_backoff_seconds: float = 2.0

    def request_extra_body(self) -> dict[str, Any]:
        extra_body: dict[str, Any] = {}
        if self.thinking_enabled is not None:
            thinking_type = "enabled" if self.thinking_enabled else "disabled"
            extra_body["thinking"] = {"type": thinking_type}
        return extra_body

    def redacted_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PROFILES: dict[str, DeepSeekProfile] = {
    "deepseek-v4-flash": DeepSeekProfile(
        name="deepseek-v4-flash",
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort=None,
    ),
    "deepseek-v4-flash-thinking": DeepSeekProfile(
        name="deepseek-v4-flash-thinking",
        model="deepseek-v4-flash",
        thinking_enabled=True,
        reasoning_effort="max",
    ),
    "deepseek-v4-pro": DeepSeekProfile(
        name="deepseek-v4-pro",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="max",
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "flash": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "flash-thinking": "deepseek-v4-flash-thinking",
    "flash_thinking": "deepseek-v4-flash-thinking",
    "deepseek-v4-flash-thinking": "deepseek-v4-flash-thinking",
    "deepseek-v4-flash_thinking": "deepseek-v4-flash-thinking",
    "deepseek-v4-flash thinking": "deepseek-v4-flash-thinking",
    "pro": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
}


@dataclass(frozen=True)
class DeepSeekProfileOverrides:
    base_url: str | None = None
    model: str | None = None
    thinking: ThinkingOverride = "profile"
    reasoning_effort: ReasoningEffortOverride = "profile"
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    request_timeout_seconds: float | None = None
    request_interval_seconds: float | None = None
    retry_backoff_seconds: float | None = None


@dataclass(frozen=True)
class DeepSeekChatResponse:
    content: str
    reasoning_content: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    cost_estimate: CostEstimate | None
    cumulative_cost_cny: float
    request_seconds: float
    response_id: str | None


@dataclass(frozen=True)
class AppWorldApiEvalSettings:
    model_profile: DeepSeekProfile
    output_dir: Path
    experiment_name: str
    split_name: str = DEV_SPLIT_NAME
    split_path: Path = DEV_SPLIT_PATH
    max_interactions: int = 50
    appworld_timeout_seconds: int = 100
    raise_on_unsafe_syntax: bool = False
    stdout_to_devnull: bool = True
    max_observation_chars: int | None = 24_000
    task_retries: int = 2
    resume: bool = True
    cost_limit_cny: float | None = DEFAULT_COST_LIMIT_CNY


@dataclass(frozen=True)
class TaskRunResult:
    task_id: str
    model_profile: str
    success: bool | None
    steps: int
    error_type: str
    output_path: Path
    status: Literal["completed", "failed"]
    cost_cny: float
    cumulative_cost_cny: float


def normalize_profile_name(profile_name: str) -> str:
    normalized = profile_name.strip().lower().replace("_", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in PROFILE_ALIASES:
        return PROFILE_ALIASES[normalized]
    normalized = normalized.replace(" ", "-")
    if normalized in PROFILE_ALIASES:
        return PROFILE_ALIASES[normalized]
    allowed = ", ".join(sorted(DEFAULT_PROFILES))
    raise ValueError(f"Unknown DeepSeek profile '{profile_name}'. Expected one of: {allowed}")


def build_profile(profile_name: str, overrides: DeepSeekProfileOverrides) -> DeepSeekProfile:
    profile = DEFAULT_PROFILES[normalize_profile_name(profile_name)]
    updates: dict[str, Any] = {}
    if overrides.base_url is not None:
        updates["base_url"] = overrides.base_url
    if overrides.model is not None:
        updates["model"] = overrides.model
    if overrides.thinking != "profile":
        updates["thinking_enabled"] = overrides.thinking == "enabled"
    if overrides.reasoning_effort != "profile":
        updates["reasoning_effort"] = (
            None if overrides.reasoning_effort == "none" else overrides.reasoning_effort
        )
    if overrides.temperature is not None:
        updates["temperature"] = overrides.temperature
    if overrides.max_tokens is not None:
        updates["max_tokens"] = overrides.max_tokens
    if overrides.max_retries is not None:
        updates["max_retries"] = overrides.max_retries
    if overrides.request_timeout_seconds is not None:
        updates["request_timeout_seconds"] = overrides.request_timeout_seconds
    if overrides.request_interval_seconds is not None:
        updates["request_interval_seconds"] = overrides.request_interval_seconds
    if overrides.retry_backoff_seconds is not None:
        updates["retry_backoff_seconds"] = overrides.retry_backoff_seconds
    return replace(profile, **updates)


def resolve_output_dir(
    output_root: Path,
    model_profile: str,
    run_name: str,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir
    return output_root / model_profile / run_name


def default_experiment_name(
    model_profile: str,
    run_name: str,
    split_name: str = DEV_SPLIT_NAME,
) -> str:
    safe_profile = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_profile)
    safe_run_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", run_name)
    safe_split_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", split_name)
    if split_name == DEV_SPLIT_NAME:
        return f"{DEFAULT_EXPERIMENT_PREFIX}_{safe_profile}_{safe_run_name}"
    return f"deepseek_v4_appworld_{safe_split_name}_{safe_profile}_{safe_run_name}"


def load_split_task_ids(
    split_name: str,
    limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[str]:
    if split_name not in SUPPORTED_SPLIT_PATHS:
        allowed = ", ".join(sorted(SUPPORTED_SPLIT_PATHS))
        raise ValueError(f"Unsupported AppWorld split '{split_name}'. Expected one of: {allowed}")
    split_task_ids = load_task_ids(split_name)
    if task_ids:
        split_task_id_set = set(split_task_ids)
        missing_task_ids = [task_id for task_id in task_ids if task_id not in split_task_id_set]
        if missing_task_ids:
            raise ValueError(
                f"These task IDs are not in AppWorld split '{split_name}' "
                f"({SUPPORTED_SPLIT_PATHS[split_name]}): {missing_task_ids}"
            )
        selected_task_ids = task_ids
    else:
        selected_task_ids = split_task_ids
    if limit is not None:
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        selected_task_ids = selected_task_ids[:limit]
    return selected_task_ids


def load_dev_task_ids(limit: int | None = None, task_ids: list[str] | None = None) -> list[str]:
    return load_split_task_ids(DEV_SPLIT_NAME, limit=limit, task_ids=task_ids)


def ensure_deepseek_api_key() -> str:
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{DEEPSEEK_API_KEY_ENV} is not set. Export it in the shell before running "
            "DeepSeek AppWorld eval."
        )
    return api_key


def model_family_for_pricing(model: str) -> Literal["flash", "pro"]:
    model_lower = model.lower()
    if "pro" in model_lower:
        return "pro"
    if "flash" in model_lower:
        return "flash"
    logger.warning(
        "Unknown DeepSeek model for pricing '%s'; using pro prices conservatively.", model
    )
    return "pro"


def usage_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_cost_cny(model: str, usage: dict[str, Any] | None) -> CostEstimate | None:
    if usage is None:
        return None

    prompt_tokens = usage_int(usage, "prompt_tokens") or 0
    completion_tokens = usage_int(usage, "completion_tokens") or 0
    prompt_cache_hit_tokens = usage_int(usage, "prompt_cache_hit_tokens")
    prompt_cache_miss_tokens = usage_int(usage, "prompt_cache_miss_tokens")

    if prompt_cache_hit_tokens is None and prompt_cache_miss_tokens is None:
        prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = prompt_tokens
    elif prompt_cache_hit_tokens is None:
        prompt_cache_hit_tokens = max(0, prompt_tokens - (prompt_cache_miss_tokens or 0))
    elif prompt_cache_miss_tokens is None:
        prompt_cache_miss_tokens = max(0, prompt_tokens - prompt_cache_hit_tokens)

    pricing = PRICING_CNY_BY_MODEL_FAMILY[model_family_for_pricing(model)]
    cost_cny = (
        prompt_cache_hit_tokens * pricing.input_cache_hit_per_million
        + prompt_cache_miss_tokens * pricing.input_cache_miss_per_million
        + completion_tokens * pricing.output_per_million
    ) / 1_000_000
    return CostEstimate(
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        completion_tokens=completion_tokens,
        cost_cny=cost_cny,
    )


def sanitize_api_key(value: Any) -> Any:
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
    if isinstance(value, str):
        return value.replace(api_key, "<DEEPSEEK_API_KEY>") if api_key else value
    if isinstance(value, list):
        return [sanitize_api_key(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_api_key(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_api_key(item) for key, item in value.items()}
    return value


def task_to_dict(task: Task) -> dict[str, Any]:
    task_dict = asdict(task)
    task_dict["datetime"] = task.datetime.isoformat()
    return task_dict


def eval_result_to_dict(
    eval_result: TaskEvalResult | AppWorldTaskEvalResult | None,
) -> dict[str, Any] | None:
    if eval_result is None:
        return None
    return asdict(eval_result)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    return str(value)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_api_key(jsonable(data)), f, indent=2, sort_keys=True)
        f.write("\n")


def message_to_openai(message: Message) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def message_to_trajectory_dict(message: Message) -> dict[str, Any]:
    return message.asdict()


def build_react_prompt_messages(task: Task) -> tuple[list[Message], Message]:
    dictionary = {
        "main_user": task.supervisor,
        "input_str": task.instruction,
        "app_descriptions": APP_DESCRIPTIONS,
        "date": task.datetime.isoformat(timespec="seconds"),
    }
    prompt = Template(PROMPT_TEMPLATE.lstrip()).render(dictionary)

    messages: list[Message] = []
    last_start = 0
    for match in re.finditer("(USER|ASSISTANT|SYSTEM):\n", prompt):
        last_end = match.span()[0]
        if len(messages) == 0:
            if last_end != 0:
                raise ValueError(f"Start of the prompt has no assigned role: {prompt[:last_end]}")
        else:
            messages[-1].content = prompt[last_start:last_end]

        mesg_type = match.group(1).lower()
        if mesg_type == "user":
            mesg = Message(role="user", content="")
        elif mesg_type == "system":
            mesg = Message(role="system", content="", today_date=task.datetime.date())
        elif mesg_type == "assistant":
            mesg = Message(role="assistant", content="")
        else:
            raise AssertionError(f"Unexpected prompt msg type: {mesg_type}. Prompt:\n{prompt}")
        messages.append(mesg)
        last_start = match.span()[1]
    messages[-1].content = prompt[last_start:]
    return messages, messages[-1]


def truncate_observation(observation: str, max_observation_chars: int | None) -> tuple[str, bool]:
    if max_observation_chars is None or max_observation_chars <= 0:
        return observation, False
    if len(observation) <= max_observation_chars:
        return observation, False
    truncation_msg = "\n...\nExecution output is too long and is not fully shown."
    remaining_chars = max(1, max_observation_chars - len(truncation_msg))
    return observation[:remaining_chars] + truncation_msg, True


class DeepSeekApiClient:
    def __init__(self, profile: DeepSeekProfile, cost_limit_cny: float | None):
        self.profile = profile
        self.cost_limit_cny = cost_limit_cny
        self.total_cost_cny = 0.0
        self.cost_limit_reached = False
        self.client = OpenAI(
            api_key=ensure_deepseek_api_key(),
            base_url=profile.base_url,
            timeout=profile.request_timeout_seconds,
        )
        self._last_request_at: float | None = None

    def _sleep_for_rate_limit(self) -> None:
        if self.profile.request_interval_seconds <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.profile.request_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def chat(self, messages: list[Message]) -> DeepSeekChatResponse:
        if self.cost_limit_reached:
            raise CostLimitExceeded(
                f"Estimated DeepSeek spend reached {self.total_cost_cny:.6f} CNY "
                f"(limit {self.cost_limit_cny:.6f} CNY)."
            )

        openai_messages = [message_to_openai(message) for message in messages]
        request_kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "messages": openai_messages,
            "max_tokens": self.profile.max_tokens,
        }
        if self.profile.temperature is not None:
            request_kwargs["temperature"] = self.profile.temperature
        if self.profile.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.profile.reasoning_effort
        extra_body = self.profile.request_extra_body()
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        last_exception: Exception | None = None
        for attempt_idx in range(self.profile.max_retries + 1):
            self._sleep_for_rate_limit()
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                self._last_request_at = time.monotonic()
                request_seconds = time.perf_counter() - start
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning_content = cast(str | None, getattr(message, "reasoning_content", None))
                if reasoning_content is None and hasattr(message, "model_extra"):
                    reasoning_content = cast(
                        str | None,
                        getattr(message, "model_extra", {}).get("reasoning_content"),
                    )
                usage = jsonable(getattr(response, "usage", None))
                usage_dict = cast(dict[str, Any] | None, usage)
                cost_estimate = estimate_cost_cny(self.profile.model, usage_dict)
                if cost_estimate is not None:
                    self.total_cost_cny += cost_estimate.cost_cny
                    if (
                        self.cost_limit_cny is not None
                        and self.total_cost_cny >= self.cost_limit_cny
                    ):
                        self.cost_limit_reached = True
                return DeepSeekChatResponse(
                    content=content,
                    reasoning_content=reasoning_content,
                    finish_reason=cast(str | None, getattr(choice, "finish_reason", None)),
                    usage=usage_dict,
                    cost_estimate=cost_estimate,
                    cumulative_cost_cny=self.total_cost_cny,
                    request_seconds=request_seconds,
                    response_id=cast(str | None, getattr(response, "id", None)),
                )
            except Exception as exc:
                self._last_request_at = time.monotonic()
                last_exception = exc
                if attempt_idx >= self.profile.max_retries:
                    break
                delay_seconds = self.profile.retry_backoff_seconds * (2**attempt_idx)
                delay_seconds += random.uniform(0.0, min(1.0, self.profile.retry_backoff_seconds))
                logger.warning(
                    "DeepSeek request failed on attempt %s/%s; retrying in %.1fs: %s",
                    attempt_idx + 1,
                    self.profile.max_retries + 1,
                    delay_seconds,
                    sanitize_api_key(str(exc)),
                )
                time.sleep(delay_seconds)

        assert last_exception is not None
        raise DeepSeekRequestError(sanitize_api_key(str(last_exception))) from last_exception


class DeepSeekReactAgent:
    def __init__(
        self,
        *,
        task: Task,
        client: DeepSeekApiClient,
        max_observation_chars: int | None,
    ):
        self.task = task
        self.client = client
        self.max_observation_chars = max_observation_chars
        self.messages, self.reminder_message = build_react_prompt_messages(task)

    def _append_observation(self, observation: str) -> tuple[str, bool]:
        observation_for_model, truncated = truncate_observation(
            observation, self.max_observation_chars
        )
        last_prompt_user_msg = self.reminder_message
        instruction = last_prompt_user_msg.content.split("solve the actual task:")[-1]
        if not instruction.startswith("\n"):
            instruction = "\n" + instruction
        observation_for_model += f"\nAs a reminder{instruction}"
        self.messages.append(Message(role="user", content=observation_for_model))
        return observation_for_model, truncated

    def next_code_block(
        self, last_execution_output: str | None
    ) -> tuple[str, DeepSeekChatResponse, str | None, bool]:
        observation_for_model = None
        observation_truncated = False
        if last_execution_output is not None:
            observation_for_model, observation_truncated = self._append_observation(
                last_execution_output
            )

        response = self.client.chat(self.messages)
        assistant_message = Message(
            role="assistant",
            content=response.content,
            stopped_by_max_tokens_limit=response.finish_reason == "length",
        )
        self.messages.append(assistant_message)
        code = extract_code_format_output(response.content)
        return code, response, observation_for_model, observation_truncated

    def chat_history(self) -> list[dict[str, Any]]:
        return [message_to_trajectory_dict(message) for message in self.messages]


def trajectory_path(output_dir: Path, task_id: str) -> Path:
    return output_dir / "tasks" / task_id / "trajectory.json"


def is_completed_trajectory(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("status") == "completed"


def read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        return []
    with open(summary_path, newline="") as f:
        return list(csv.DictReader(f))


def upsert_summary_row(summary_path: Path, result: TaskRunResult) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "model_profile",
        "success",
        "steps",
        "error_type",
        "cost_cny",
        "cumulative_cost_cny",
        "output_path",
    ]
    rows = read_summary_rows(summary_path)
    new_row = {
        "task_id": result.task_id,
        "model_profile": result.model_profile,
        "success": "" if result.success is None else str(result.success),
        "steps": str(result.steps),
        "error_type": result.error_type,
        "cost_cny": f"{result.cost_cny:.8f}",
        "cumulative_cost_cny": f"{result.cumulative_cost_cny:.8f}",
        "output_path": str(result.output_path.resolve()),
    }
    replaced = False
    for idx, row in enumerate(rows):
        if (
            row.get("task_id") == result.task_id
            and row.get("model_profile") == result.model_profile
        ):
            rows[idx] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run_config(
    settings: AppWorldApiEvalSettings,
    task_ids: list[str],
    summary_path: Path,
) -> None:
    save_json(
        settings.output_dir / "run_config.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "split": settings.split_name,
            "split_path": str(settings.split_path),
            "num_selected_tasks": len(task_ids),
            "task_ids": task_ids,
            "model_profile": settings.model_profile.redacted_dict(),
            "experiment_name": settings.experiment_name,
            "max_interactions": settings.max_interactions,
            "appworld_timeout_seconds": settings.appworld_timeout_seconds,
            "raise_on_unsafe_syntax": settings.raise_on_unsafe_syntax,
            "stdout_to_devnull": settings.stdout_to_devnull,
            "max_observation_chars": settings.max_observation_chars,
            "task_retries": settings.task_retries,
            "resume": settings.resume,
            "cost_limit_cny": settings.cost_limit_cny,
            "pricing_cny_per_1m_tokens": {
                family: asdict(pricing) for family, pricing in PRICING_CNY_BY_MODEL_FAMILY.items()
            },
            "summary_path": summary_path.resolve(),
        },
    )


def safe_close_world(world: AppWorldInterface) -> None:
    if world.clean:
        return
    try:
        world.close_world()
    except Exception:
        logger.exception("Failed to close AppWorld task world cleanly.")


def run_single_task_once(
    *,
    world: AppWorldInterface,
    task_id: str,
    settings: AppWorldApiEvalSettings,
    client: DeepSeekApiClient,
    output_path: Path,
) -> TaskRunResult:
    task: Task | None = None
    agent: DeepSeekReactAgent | None = None
    initial_prompt_messages: list[dict[str, Any]] | None = None
    steps: list[dict[str, Any]] = []
    eval_result: TaskEvalResult | None = None
    error_type = ""
    error_message = ""
    status: Literal["completed", "failed"] = "failed"

    started_at = datetime.now(UTC)
    try:
        task = world.initialize(
            task_id=task_id,
            experiment_name=settings.experiment_name,
            raise_on_unsafe_syntax=settings.raise_on_unsafe_syntax,
        )
        agent = DeepSeekReactAgent(
            task=task,
            client=client,
            max_observation_chars=settings.max_observation_chars,
        )
        initial_prompt_messages = agent.chat_history()
        last_execution_output: str | None = None

        for interaction_idx in range(settings.max_interactions):
            code, model_response, observation_for_model, observation_truncated = (
                agent.next_code_block(last_execution_output)
            )
            environment_result = world.execute(code)
            task_completed = world.task_completed()
            steps.append(
                {
                    "step_index": interaction_idx + 1,
                    "observation": last_execution_output,
                    "observation_sent_to_model": observation_for_model,
                    "observation_truncated": observation_truncated,
                    "model_output": model_response.content,
                    "reasoning_content": model_response.reasoning_content,
                    "finish_reason": model_response.finish_reason,
                    "usage": model_response.usage,
                    "cost_estimate": (
                        model_response.cost_estimate.asdict()
                        if model_response.cost_estimate is not None
                        else None
                    ),
                    "cumulative_cost_cny": model_response.cumulative_cost_cny,
                    "request_seconds": model_response.request_seconds,
                    "response_id": model_response.response_id,
                    "action_code": code,
                    "no_code_found": no_code_found(code),
                    "environment_result": environment_result,
                    "execution_failed": execution_failed(environment_result),
                    "task_completed": task_completed,
                }
            )
            last_execution_output = environment_result
            if client.cost_limit_reached:
                error_type = "cost_limit_reached"
                break
            if task_completed:
                break
        else:
            error_type = "max_interactions"

        appworld_eval_result = world.evaluate()
        eval_result = TaskEvalResult.create(
            appworld_eval_result,
            num_interactions=len(steps),
        )
        status = "completed"
        safe_close_world(world)
    except DeepSeekRequestError as exc:
        error_type = "deepseek_request_error"
        error_message = sanitize_api_key(str(exc))
        safe_close_world(world)
    except CostLimitExceeded as exc:
        error_type = "cost_limit_reached"
        error_message = sanitize_api_key(str(exc))
        safe_close_world(world)
    except AppWorldInitializeTaskError as exc:
        error_type = "appworld_initialize_error"
        error_message = sanitize_api_key(str(exc))
        if not world.clean:
            world.restart()
    except AppWorldExecutionError as exc:
        error_type = "appworld_execution_error"
        error_message = sanitize_api_key(str(exc))
        if not world.clean:
            world.restart()
    except (requests.HTTPError, requests.ConnectionError) as exc:
        error_type = "appworld_connection_error"
        error_message = sanitize_api_key(str(exc))
        if not world.clean:
            world.restart()
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = sanitize_api_key(str(exc))
        safe_close_world(world)

    finished_at = datetime.now(UTC)
    success = None if eval_result is None else eval_result.success
    trajectory = {
        "status": status,
        "task_id": task_id,
        "model_profile": settings.model_profile.name,
        "model_config": settings.model_profile.redacted_dict(),
        "experiment_name": settings.experiment_name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "task": task_to_dict(task) if task else None,
        "initial_prompt_messages": initial_prompt_messages,
        "chat_history": agent.chat_history() if agent is not None else None,
        "steps": steps,
        "success": success,
        "eval_result": eval_result_to_dict(eval_result),
        "steps_count": len(steps),
        "cost_cny": sum(
            step["cost_estimate"]["cost_cny"]
            for step in steps
            if step.get("cost_estimate") is not None
        ),
        "cumulative_cost_cny": client.total_cost_cny,
        "cost_limit_cny": settings.cost_limit_cny,
        "error_type": error_type,
        "error_message": error_message,
    }
    save_json(output_path, trajectory)

    return TaskRunResult(
        task_id=task_id,
        model_profile=settings.model_profile.name,
        success=success,
        steps=len(steps),
        error_type=error_type,
        output_path=output_path,
        status=status,
        cost_cny=sum(
            step["cost_estimate"]["cost_cny"]
            for step in steps
            if step.get("cost_estimate") is not None
        ),
        cumulative_cost_cny=client.total_cost_cny,
    )


def run_single_task(
    *,
    world: AppWorldInterface,
    task_id: str,
    settings: AppWorldApiEvalSettings,
    client: DeepSeekApiClient,
) -> TaskRunResult:
    output_path = trajectory_path(settings.output_dir, task_id)
    if settings.resume and is_completed_trajectory(output_path):
        logger.info(
            "Skipping completed task_id=%s for profile=%s", task_id, settings.model_profile.name
        )
        with open(output_path) as f:
            trajectory = json.load(f)
        return TaskRunResult(
            task_id=task_id,
            model_profile=settings.model_profile.name,
            success=cast(bool | None, trajectory.get("success")),
            steps=int(trajectory.get("steps_count", 0)),
            error_type=str(trajectory.get("error_type", "")),
            output_path=output_path,
            status="completed",
            cost_cny=float(trajectory.get("cost_cny", 0.0)),
            cumulative_cost_cny=float(trajectory.get("cumulative_cost_cny", 0.0)),
        )

    last_result: TaskRunResult | None = None
    for attempt_idx in range(settings.task_retries + 1):
        if attempt_idx:
            logger.info("Retrying task_id=%s after failed task attempt %s", task_id, attempt_idx)
            world.ensure_server()
        last_result = run_single_task_once(
            world=world,
            task_id=task_id,
            settings=settings,
            client=client,
            output_path=output_path,
        )
        if last_result.status == "completed":
            return last_result
        if last_result.error_type == "cost_limit_reached":
            return last_result
        if last_result.error_type == "deepseek_request_error":
            return last_result
    assert last_result is not None
    return last_result


def run_deepseek_appworld_dev_eval(
    *,
    settings: AppWorldApiEvalSettings,
    task_ids: list[str],
) -> list[TaskRunResult]:
    client = DeepSeekApiClient(settings.model_profile, settings.cost_limit_cny)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = settings.output_dir / "summary.csv"
    save_run_config(settings, task_ids, summary_path)

    results: list[TaskRunResult] = []
    with AppWorldInterface(
        stdout_to_devnull=settings.stdout_to_devnull,
        timeout_seconds=settings.appworld_timeout_seconds,
    ) as world:
        for idx, task_id in enumerate(task_ids, start=1):
            logger.info(
                "Running DeepSeek AppWorld %s task %s/%s: %s (%s)",
                settings.split_name,
                idx,
                len(task_ids),
                task_id,
                settings.model_profile.name,
            )
            result = run_single_task(
                world=world,
                task_id=task_id,
                settings=settings,
                client=client,
            )
            upsert_summary_row(summary_path, result)
            results.append(result)
            if client.cost_limit_reached or result.error_type == "cost_limit_reached":
                logger.warning(
                    "Stopping DeepSeek AppWorld %s eval because estimated spend reached "
                    "%.6f CNY (limit %.6f CNY).",
                    settings.split_name,
                    client.total_cost_cny,
                    settings.cost_limit_cny,
                )
                break
    return results
