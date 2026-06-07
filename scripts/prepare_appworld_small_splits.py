#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Create small AppWorld split files for lightweight LOOP experiments."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


DEFAULT_TRAIN_SOURCES = ("train_difficulty_1", "train_difficulty_2")


def _read_split(splits_dir: Path, split_name: str) -> list[str]:
    split_path = splits_dir / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def _write_split(splits_dir: Path, split_name: str, task_ids: list[str]) -> Path:
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{split_name}.txt"
    split_path.write_text("\n".join(task_ids) + "\n")
    return split_path


def _sample(task_ids: list[str], size: int, rng: random.Random, label: str) -> list[str]:
    shuffled = list(task_ids)
    rng.shuffle(shuffled)

    if len(shuffled) < size:
        print(f"WARNING: requested {size} {label} tasks, but only {len(shuffled)} are available.")

    return shuffled[: min(size, len(shuffled))]


def _dedupe_preserving_order(task_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for task_id in task_ids:
        if task_id not in seen:
            deduped.append(task_id)
            seen.add(task_id)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--repo-splits-dir", type=Path, default=Path("data/appworld_splits"))
    parser.add_argument("--appworld-root", type=Path, default=None)
    parser.add_argument("--no-appworld-mirror", action="store_true")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--dev-size", type=int, default=64)
    parser.add_argument("--train-output", default="train_small128")
    parser.add_argument("--dev-output", default="dev_small64")
    parser.add_argument("--train-source", action="append", default=None)
    parser.add_argument("--fallback-train-source", default="train")
    parser.add_argument("--dev-source", default="dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    repo_splits_dir = args.repo_splits_dir

    train_sources = tuple(args.train_source) if args.train_source else DEFAULT_TRAIN_SOURCES
    train_ids: list[str] = []
    missing_train_sources: list[str] = []
    for split_name in train_sources:
        try:
            train_ids.extend(_read_split(repo_splits_dir, split_name))
        except FileNotFoundError:
            missing_train_sources.append(split_name)

    if not train_ids:
        if missing_train_sources:
            print(f"WARNING: missing train source splits: {', '.join(missing_train_sources)}")
        train_ids = _read_split(repo_splits_dir, args.fallback_train_source)

    train_ids = _dedupe_preserving_order(train_ids)
    dev_ids = _dedupe_preserving_order(_read_split(repo_splits_dir, args.dev_source))

    train_small = _sample(train_ids, args.train_size, rng, "train")
    dev_small = _sample(dev_ids, args.dev_size, rng, "dev")

    written_paths = [
        _write_split(repo_splits_dir, args.train_output, train_small),
        _write_split(repo_splits_dir, args.dev_output, dev_small),
    ]

    if not args.no_appworld_mirror:
        appworld_root = args.appworld_root or os.environ.get("APPWORLD_ROOT")
        if not appworld_root:
            raise RuntimeError(
                "APPWORLD_ROOT is required unless --no-appworld-mirror is passed."
            )
        appworld_dataset_dir = Path(appworld_root) / "data" / "datasets"
        written_paths.extend(
            [
                _write_split(appworld_dataset_dir, args.train_output, train_small),
                _write_split(appworld_dataset_dir, args.dev_output, dev_small),
            ]
        )

    print(f"{args.train_output}: {len(train_small)} tasks")
    print(f"{args.dev_output}: {len(dev_small)} tasks")
    for path in written_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
