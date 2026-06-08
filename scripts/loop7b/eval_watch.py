# 中文注释：监控 Qwen2.5-7B LOOP checkpoint 并自动跑 AppWorld dev 评测。
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Evaluate base/checkpoints on AppWorld dev splits and maintain trend summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

LOWER_BETTER_METRICS = (
    "execution_failed_count",
    "invalid_api_hits",
    "multiple_code_cells_per_turn",
    "execution_errors_per_turn",
    "assumption_words_per_rollout",
    "dummy_words_per_rollout",
    "failed_api_call_give_up_rate",
    "capitulation_after_error_count",
)
DOC_HIGHER_BETTER_METRICS = (
    "api_doc_calls_per_rollout",
    "api_description_calls_per_rollout",
    "api_doc_or_description_calls_per_rollout",
    "doc_before_api_call_rate",
)
PRIMARY_RATIO_METRICS = ("TGC", "SGC", "average_partial_pass_rate")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _checkpoint_iteration(path: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)|ckpt(\d+)", path.name)
    if not match:
        return None
    return int(next(group for group in match.groups() if group is not None))


def _is_complete_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / ".complete").exists():
        return True
    has_trainer_state = (path / "trainer_state.pt").exists()
    has_lora = (path / "lora" / "adapter_config.json").exists() or (
        path / "lora_vllmqwen25" / "adapter_config.json"
    ).exists()
    return has_trainer_state and has_lora


def _summary_paths(summary_dir: Path) -> dict[str, Path]:
    return {
        "json": summary_dir / "summary.json",
        "csv": summary_dir / "summary.csv",
        "history": summary_dir / "summary_history.jsonl",
        "best": summary_dir / "best_checkpoint.json",
        "config": summary_dir / "RUN_CONFIG.json",
    }


def _load_rows(summary_dir: Path) -> list[dict[str, Any]]:
    path = _summary_paths(summary_dir)["json"]
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _write_rows(summary_dir: Path, rows: list[dict[str, Any]]) -> None:
    paths = _summary_paths(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with paths["csv"].open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_history(summary_dir: Path, row: dict[str, Any]) -> None:
    path = _summary_paths(summary_dir)["history"]
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("split") or ""),
        str(row.get("experiment_name") or ""),
        str(row.get("repeat_index") or 0),
    )


def _upsert_row(summary_dir: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [old for old in _load_rows(summary_dir) if _row_key(old) != _row_key(row)]
    rows.append(row)
    rows.sort(
        key=lambda item: (
            str(item.get("split")),
            str(item.get("checkpoint_iteration")),
            str(item.get("experiment_name")),
        )
    )
    _write_rows(summary_dir, rows)
    _append_history(summary_dir, row)
    _write_best_checkpoint(summary_dir, rows, str(row.get("split") or ""))
    return rows


def _best_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        _safe_float(row.get("TGC")) or -1.0,
        _safe_float(row.get("SGC")) or -1.0,
        _safe_float(row.get("average_partial_pass_rate")) or -1.0,
        -(_safe_float(row.get("invalid_api_hits")) or 1e18),
        -(_safe_float(row.get("execution_failed_count")) or 1e18),
    )


def _write_best_checkpoint(summary_dir: Path, rows: list[dict[str, Any]], split: str) -> None:
    candidates = [
        row
        for row in rows
        if str(row.get("split")) == split
        and _safe_float(row.get("checkpoint_iteration")) is not None
    ]
    if not candidates:
        return
    best = max(candidates, key=_best_key)
    _summary_paths(summary_dir)["best"].write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n"
    )


def _add_base_comparison(row: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if row.get("checkpoint_name") == "base":
        return
    split = row.get("split")
    base_rows = [
        existing
        for existing in rows
        if existing.get("split") == split and existing.get("checkpoint_name") == "base"
    ]
    if not base_rows:
        return
    base = base_rows[-1]
    row["base_experiment_name"] = base.get("experiment_name")

    for metric in PRIMARY_RATIO_METRICS + LOWER_BETTER_METRICS + DOC_HIGHER_BETTER_METRICS:
        row[f"ratio_{metric}_checkpoint_over_base"] = _safe_div(row.get(metric), base.get(metric))
    for metric in LOWER_BETTER_METRICS:
        row[f"reduction_{metric}"] = _safe_div(base.get(metric), row.get(metric))
    for metric in DOC_HIGHER_BETTER_METRICS:
        row[f"increase_{metric}"] = _safe_div(row.get(metric), base.get(metric))


def _run_and_log(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    with log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n# command_start {datetime.now().isoformat(timespec='seconds')}\n")
        log_fh.write(printable + "\n")
        log_fh.flush()
        print(printable)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_fh.write(line)
        return_code = process.wait()
        log_fh.write(
            f"# command_end {datetime.now().isoformat(timespec='seconds')} "
            f"rc={return_code}\n"
        )
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def _split_counts(repo_root: Path, appworld_root: Path, split: str) -> dict[str, int | None]:
    repo_path = repo_root / "data" / "appworld_splits" / f"{split}.txt"
    appworld_path = appworld_root / "data" / "datasets" / f"{split}.txt"
    return {
        "repo_split_count": len(repo_path.read_text().splitlines()) if repo_path.exists() else None,
        "appworld_split_count": len(appworld_path.read_text().splitlines())
        if appworld_path.exists()
        else None,
    }


def _write_run_config(args: argparse.Namespace) -> None:
    paths = _summary_paths(args.summary_dir)
    if paths["config"].exists():
        return
    payload = {
        "created_at": _now(),
        "mode": args.mode,
        "split": args.split,
        "run_name": args.run_name,
        "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
        "summary_dir": str(args.summary_dir),
        "cuda_visible_devices": args.cuda_visible_devices,
        "num_scenario_runners": args.num_scenario_runners,
        "llm": args.llm,
        "max_gpu_mem_utilization": args.max_gpu_mem_utilization,
        "eager_mode": args.eager_mode,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "eval_every": args.eval_every,
        "repeat": args.repeat,
    }
    paths["config"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_env(args: argparse.Namespace, experiment_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["APPWORLD_ROOT"] = str(args.appworld_root)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["TMPDIR"] = str(args.tmpdir)
    env["RAY_TMPDIR"] = str(args.tmpdir)
    env["VLLM_CACHE_ROOT"] = str(args.summary_dir / "vllm_cache" / experiment_name)
    env["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.max_gpu_mem_utilization)
    env["PATH"] = f"{args.appworld_env_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _experiment_name(
    *, args: argparse.Namespace, checkpoint_name: str, iteration: int | None, repeat_index: int
) -> str:
    suffix = f"_r{repeat_index}" if args.repeat > 1 else ""
    if checkpoint_name == "base":
        return _sanitize(f"eval_base_qwen25_7b_{args.split}_stage2{suffix}")
    iteration_text = f"ckpt{iteration}" if iteration is not None else checkpoint_name
    return _sanitize(f"eval_{args.run_name}_{iteration_text}_{args.split}{suffix}")


def _run_eval_once(
    args: argparse.Namespace,
    *,
    adapter_path: str,
    checkpoint_name: str,
    checkpoint_iteration: int | None,
    repeat_index: int,
) -> dict[str, Any]:
    experiment_name = _experiment_name(
        args=args,
        checkpoint_name=checkpoint_name,
        iteration=checkpoint_iteration,
        repeat_index=repeat_index,
    )
    run_dir = args.summary_dir / "runs" / experiment_name
    log_path = args.summary_dir / "logs" / f"{experiment_name}.log"
    all_results_path = run_dir / "appworld-results" / "all_results.txt"
    episode_summary_path = run_dir / "episode_summary.txt"
    behavior_json_path = run_dir / "behavior_summary.json"
    behavior_csv_path = run_dir / "behavior_summary.csv"
    env = _prepare_env(args, experiment_name)

    start_time = _now()
    common_overrides = [
        f"experiment_name={experiment_name}",
        f"llm={args.llm}",
        f"llm.adapter_path={adapter_path}",
        f"scenario_sampler.dataset_name={args.split}",
        f"num_scenario_runners={args.num_scenario_runners}",
        f"llm.max_gpu_mem_utilization={args.max_gpu_mem_utilization}",
        "llm.vllm_server.gpus_per_vllm_server=1",
        f"llm.vllm_server.max_model_len={args.max_model_len}",
        f"llm.vllm_class.max_new_tokens={args.max_new_tokens}",
        f"llm.vllm_server.eager_mode={args.eager_mode}",
        *args.hydra_override,
    ]

    _run_and_log(
        [args.python_bin, "-m", "scripts.appworld.run_inference", *common_overrides],
        env,
        log_path,
    )
    _run_and_log(
        [
            args.python_bin,
            "-m",
            "scripts.appworld.eval_parse_and_log",
            f"experiment_name={experiment_name}",
            f"llm={args.llm}",
            f"scenario_sampler.dataset_name={args.split}",
            f"log_dir={run_dir / 'appworld-results'}",
        ],
        env,
        log_path,
    )
    _run_and_log(
        [
            args.python_bin,
            "-m",
            "scripts.loop7b.summarize_appworld_episodes",
            experiment_name,
            "--appworld-root",
            str(args.appworld_root),
        ],
        env,
        episode_summary_path,
    )
    _run_and_log(
        [
            args.python_bin,
            "-m",
            "scripts.loop7b.analyze_appworld_behavior",
            "--experiment-name",
            experiment_name,
            "--appworld-root",
            str(args.appworld_root),
            "--all-results",
            str(all_results_path),
            "--checkpoint-name",
            checkpoint_name,
            "--eval-result-path",
            str(all_results_path),
            "--eval-log-path",
            str(log_path),
            "--output-json",
            str(behavior_json_path),
            "--output-csv",
            str(behavior_csv_path),
            *(
                ["--checkpoint-iteration", str(checkpoint_iteration)]
                if checkpoint_iteration
                else []
            ),
        ],
        env,
        log_path,
    )
    end_time = _now()

    row = json.loads(behavior_json_path.read_text())
    row.update(
        {
            "split": args.split,
            "adapter_path": adapter_path,
            "repeat_index": repeat_index,
            "eval_start_time": start_time,
            "eval_end_time": end_time,
            "episode_summary_path": str(episode_summary_path),
            "summary_dir": str(args.summary_dir),
            **_split_counts(args.repo_root, args.appworld_root, args.split),
        }
    )
    rows = _load_rows(args.summary_dir)
    _add_base_comparison(row, rows)
    _upsert_row(args.summary_dir, row)
    return row


def _already_evaluated(rows: list[dict[str, Any]], split: str, checkpoint_name: str) -> bool:
    return any(
        row.get("split") == split
        and row.get("checkpoint_name") == checkpoint_name
        and row.get("repeat_index") == 0
        for row in rows
    )


def _run_repeats(
    args: argparse.Namespace,
    *,
    adapter_path: str,
    checkpoint_name: str,
    checkpoint_iteration: int | None,
) -> None:
    for repeat_index in range(args.repeat):
        _run_eval_once(
            args,
            adapter_path=adapter_path,
            checkpoint_name=checkpoint_name,
            checkpoint_iteration=checkpoint_iteration,
            repeat_index=repeat_index,
        )


def _run_base_if_needed(args: argparse.Namespace) -> None:
    rows = _load_rows(args.summary_dir)
    if _already_evaluated(rows, args.split, "base"):
        print(f"base already evaluated for split={args.split}; skip")
        return
    _run_repeats(args, adapter_path="null", checkpoint_name="base", checkpoint_iteration=None)


def _checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_root is None:
        return []
    candidates = sorted(
        args.checkpoint_root.glob("checkpoint-*"),
        key=lambda item: _checkpoint_iteration(item) or -1,
    )
    paths = []
    for path in candidates:
        iteration = _checkpoint_iteration(path)
        if iteration is None or iteration % args.eval_every != 0:
            continue
        if _is_complete_checkpoint(path):
            paths.append(path)
        else:
            print(f"skip incomplete checkpoint: {path}")
    return paths


def _watch(args: argparse.Namespace) -> None:
    if args.run_base_if_missing:
        _run_base_if_needed(args)

    while True:
        rows = _load_rows(args.summary_dir)
        for checkpoint_path in _checkpoint_paths(args):
            checkpoint_name = checkpoint_path.name
            if _already_evaluated(rows, args.split, checkpoint_name):
                continue
            iteration = _checkpoint_iteration(checkpoint_path)
            _run_repeats(
                args,
                adapter_path=str(checkpoint_path),
                checkpoint_name=checkpoint_name,
                checkpoint_iteration=iteration,
            )
            rows = _load_rows(args.summary_dir)

        if args.once:
            break
        time.sleep(args.poll_seconds)


def _validate_args(args: argparse.Namespace) -> None:
    if args.split.startswith("test_") and not args.allow_test_split:
        raise ValueError(
            "This watcher is for train/dev analysis. "
            "Refusing test split without --allow-test-split."
        )
    if args.appworld_root is None:
        raise ValueError("APPWORLD_ROOT is required unless --appworld-root is passed.")
    args.repo_root = args.repo_root.resolve()
    args.summary_dir = args.summary_dir.resolve()
    args.appworld_root = args.appworld_root.resolve()
    args.tmpdir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "checkpoint", "watch"), default="watch")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--appworld-root",
        type=Path,
        default=Path(os.environ["APPWORLD_ROOT"]) if os.environ.get("APPWORLD_ROOT") else None,
    )
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--run-name", default="qwen25_7b_loop_200x24x6_lora16")
    parser.add_argument(
        "--summary-dir", type=Path, default=Path("artifacts/loop7b_stage2_dev_eval")
    )
    parser.add_argument("--split", default="dev_small64")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--run-base-if-missing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--appworld-env-bin", default=str(Path.cwd() / "appworld-env" / "bin"))
    parser.add_argument("--tmpdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument(
        "--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1,2,3")
    )
    parser.add_argument("--num-scenario-runners", type=int, default=12)
    parser.add_argument("--llm", default="qwen_2_5_7b_lora16_eval")
    parser.add_argument("--max-gpu-mem-utilization", type=float, default=0.82)
    parser.add_argument("--eager-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--hydra-override", action="append", default=[])
    parser.add_argument("--allow-test-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    if args.mode == "base":
        _run_repeats(args, adapter_path="null", checkpoint_name="base", checkpoint_iteration=None)
    elif args.mode == "checkpoint":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for --mode checkpoint")
        if not _is_complete_checkpoint(args.checkpoint_path):
            raise ValueError(f"Checkpoint appears incomplete: {args.checkpoint_path}")
        _run_repeats(
            args,
            adapter_path=str(args.checkpoint_path),
            checkpoint_name=args.checkpoint_path.name,
            checkpoint_iteration=_checkpoint_iteration(args.checkpoint_path),
        )
    else:
        if args.checkpoint_root is None:
            raise ValueError("--checkpoint-root is required for --mode watch")
        _watch(args)


if __name__ == "__main__":
    main()
