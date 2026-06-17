# 中文注释：直接调用 DeepSeek V4 API 在 AppWorld split 上运行 ReAct 评测，默认只跑 dev。
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Run DeepSeek V4 API ReAct evaluation on an AppWorld split, defaulting to dev."""

from __future__ import annotations

import argparse
from pathlib import Path

from phi_agents.api_eval.deepseek_appworld import (
    DEFAULT_COST_LIMIT_CNY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROFILES,
    DEV_SPLIT_PATH,
    SUPPORTED_SPLIT_PATHS,
    AppWorldApiEvalSettings,
    DeepSeekProfileOverrides,
    build_profile,
    default_experiment_name,
    load_split_task_ids,
    resolve_output_dir,
    run_deepseek_appworld_dev_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek V4 API ReAct evaluation on AppWorld dev by default."
    )
    parser.add_argument(
        "--model-profile",
        "--profile",
        required=True,
        help=(
            "DeepSeek profile. Supported: "
            f"{', '.join(sorted(DEFAULT_PROFILES))}. "
            "Aliases include flash, flash-thinking, and pro."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N task IDs from the selected split.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SUPPORTED_SPLIT_PATHS),
        default="dev",
        help=(
            "AppWorld split to run. Defaults to dev. "
            "Use test_normal/test_challenge only for final evaluation, not tuning."
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=None,
        help=(
            "Run a specific task ID from the selected split. Can be passed multiple times. "
            f"Default split file is {DEV_SPLIT_PATH}."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks whose trajectory.json already has status=completed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for profile-separated results.",
    )
    parser.add_argument(
        "--run-name",
        default="dev",
        help="Run directory name under output-root/model-profile/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the full output directory.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="AppWorld experiment_name. Defaults to a DeepSeek name derived from split/profile/run-name.",
    )
    parser.add_argument("--base-url", default=None, help="DeepSeek-compatible OpenAI base URL.")
    parser.add_argument("--model", default=None, help="Override the API model ID for this profile.")
    parser.add_argument(
        "--thinking",
        choices=["profile", "enabled", "disabled"],
        default="profile",
        help="Override profile thinking mode.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["profile", "high", "max", "none"],
        default="profile",
        help="Override profile reasoning_effort.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--api-retries", type=int, default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument("--request-interval-seconds", type=float, default=None)
    parser.add_argument("--retry-backoff-seconds", type=float, default=None)
    parser.add_argument(
        "--cost-limit-cny",
        type=float,
        default=DEFAULT_COST_LIMIT_CNY,
        help=(
            "Stop launching further DeepSeek API requests after estimated run spend reaches "
            "this many CNY. Set <=0 to disable."
        ),
    )
    parser.add_argument("--max-interactions", type=int, default=50)
    parser.add_argument("--appworld-timeout-seconds", type=int, default=100)
    parser.add_argument(
        "--raise-on-unsafe-syntax",
        action="store_true",
        help="Forward raise_on_unsafe_syntax=True to AppWorld initialize.",
    )
    parser.add_argument(
        "--stdout-to-devnull",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide AppWorld server stdout.",
    )
    parser.add_argument(
        "--max-observation-chars",
        type=int,
        default=24_000,
        help="Approximate per-observation char cap sent back to the model. <=0 disables.",
    )
    parser.add_argument("--task-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = DeepSeekProfileOverrides(
        base_url=args.base_url,
        model=args.model,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.api_retries,
        request_timeout_seconds=args.request_timeout_seconds,
        request_interval_seconds=args.request_interval_seconds,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    profile = build_profile(args.model_profile, overrides)
    output_dir = resolve_output_dir(
        output_root=args.output_root,
        model_profile=profile.name,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    experiment_name = args.experiment_name or default_experiment_name(
        profile.name,
        args.run_name,
        args.split,
    )
    max_observation_chars = (
        None
        if args.max_observation_chars is not None and args.max_observation_chars <= 0
        else args.max_observation_chars
    )
    cost_limit_cny = (
        None
        if args.cost_limit_cny is not None and args.cost_limit_cny <= 0
        else args.cost_limit_cny
    )
    split_path = SUPPORTED_SPLIT_PATHS[args.split]
    task_ids = load_split_task_ids(args.split, limit=args.limit, task_ids=args.task_id)
    settings = AppWorldApiEvalSettings(
        model_profile=profile,
        output_dir=output_dir,
        experiment_name=experiment_name,
        split_name=args.split,
        split_path=split_path,
        max_interactions=args.max_interactions,
        appworld_timeout_seconds=args.appworld_timeout_seconds,
        raise_on_unsafe_syntax=args.raise_on_unsafe_syntax,
        stdout_to_devnull=args.stdout_to_devnull,
        max_observation_chars=max_observation_chars,
        task_retries=args.task_retries,
        resume=args.resume,
        cost_limit_cny=cost_limit_cny,
    )

    results = run_deepseek_appworld_dev_eval(settings=settings, task_ids=task_ids)
    completed = sum(result.status == "completed" for result in results)
    successes = sum(result.success is True for result in results)
    estimated_cost = max((result.cumulative_cost_cny for result in results), default=0.0)
    print(f"Profile: {profile.name}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Summary: {(output_dir / 'summary.csv').resolve()}")
    print(f"Completed rollouts: {completed}/{len(task_ids)}")
    print(f"Successful tasks: {successes}/{completed if completed else 0}")
    print(f"Estimated DeepSeek spend: {estimated_cost:.6f} CNY")
    if cost_limit_cny is not None and estimated_cost >= cost_limit_cny:
        print(f"Stopped because estimated spend reached the {cost_limit_cny:.6f} CNY limit.")


if __name__ == "__main__":
    main()
