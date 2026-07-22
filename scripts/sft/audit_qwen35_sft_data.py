from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from phi_agents.sft.trainer import SFTConfig, train

EXPECTED_COUNTS = {
    "d12": (509, 129),
    "d3": (81, 16),
}
SECRET_PATTERNS = {
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}", re.IGNORECASE),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_task_ids(repo_root: Path) -> set[str]:
    result: set[str] = set()
    split_root = repo_root / "data" / "appworld_splits"
    for path in split_root.glob("*.txt"):
        if path.stem == "dev" or path.stem.startswith("test"):
            result.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return result


def _audit_source(
    *,
    name: str,
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    forbidden_task_ids: set[str],
) -> dict[str, Any]:
    train_path = data_dir / "qwen_sft_train.jsonl"
    validation_path = data_dir / "qwen_sft_validation.jsonl"
    train_rows = _read_jsonl(train_path)
    validation_rows = _read_jsonl(validation_path)
    expected_train, expected_validation = EXPECTED_COUNTS[name]
    if (len(train_rows), len(validation_rows)) != (expected_train, expected_validation):
        raise ValueError(
            f"Unexpected {name} row counts: actual={(len(train_rows), len(validation_rows))} "
            f"expected={(expected_train, expected_validation)}"
        )

    task_ids = {
        str((row.get("metadata") or {}).get("task_id") or "")
        for row in train_rows + validation_rows
    }
    leaked = sorted(task_ids & forbidden_task_ids)
    if leaked:
        raise ValueError(f"{name} contains dev/test task IDs: {leaked[:10]}")

    raw_text = train_path.read_text() + validation_path.read_text()
    secret_hits = {
        label: len(pattern.findall(raw_text)) for label, pattern in SECRET_PATTERNS.items()
    }
    if any(secret_hits.values()):
        raise ValueError(f"{name} contains credential-like strings: {secret_hits}")

    manifest_path = data_dir / "data_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("dev_or_test_tasks_present") is not False:
        raise ValueError(f"{manifest_path} does not certify train-only data")
    if source_manifest.get("api_key_present_in_output") is not False:
        raise ValueError(f"{manifest_path} reports an API key")

    tokenized = train(
        SFTConfig(
            train_jsonl=train_path,
            validation_jsonl=validation_path,
            output_dir=output_dir / name,
            model_name="Qwen/Qwen3.5-4B",
            model_path=model_path,
            max_length=16_384,
            epochs=1 if name == "d12" else 2,
            gradient_accumulation_steps=4 if name == "d12" else 2,
            sparse_assistant_logits=True,
            attention_backend="sdpa",
        ),
        tokenize_only=True,
    )
    stats = tokenized["data_stats"]
    if stats["token_truncation_ratio"] != 0.0:
        raise ValueError(f"{name} silently truncated training data")
    if stats["other_anomaly_masked_steps"] != 0:
        raise ValueError(f"{name} contains unexpected action masks")
    return {
        "name": name,
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "task_count": len(task_ids),
        "dev_test_task_overlap": len(leaked),
        "secret_hits": secret_hits,
        "source_manifest": str(manifest_path.resolve()),
        "data_sha256": tokenized["data_sha256"],
        "data_stats": stats,
        "schedule_preflight": tokenized["schedule_preflight"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = _split_task_ids(args.repo_root)
    rows = [
        _audit_source(
            name="d12",
            data_dir=args.data_root / "d12_supervision_v2",
            model_path=args.model_path,
            output_dir=args.output_dir,
            forbidden_task_ids=forbidden,
        ),
        _audit_source(
            name="d3",
            data_dir=args.data_root / "d3_supervision_v2",
            model_path=args.model_path,
            output_dir=args.output_dir,
            forbidden_task_ids=forbidden,
        ),
    ]
    report = {
        "status": "passed",
        "max_length": 16_384,
        "forbidden_dev_test_task_count": len(forbidden),
        "stages": rows,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
