from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class SFTConfig:
    train_jsonl: Path
    validation_jsonl: Path
    output_dir: Path
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_path: Path | None = None
    max_length: int = 16_384
    turn_overlap: int = 0
    preserve_history: bool = True
    audit_supervision: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    attention_backend: str = "auto"
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 20_260_713
    resume_from_checkpoint: str | None = None
    initial_adapter: Path | None = None
    max_steps: int = -1
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    use_cpu: bool = False


def training_schedule_preflight(config: SFTConfig, train_windows: int) -> dict[str, int | float]:
    if train_windows < 1:
        raise ValueError("At least one training window is required")
    micro_batches_per_epoch = math.ceil(train_windows / config.per_device_train_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / config.gradient_accumulation_steps
    )
    scheduler_total_steps = (
        config.max_steps
        if config.max_steps > 0
        else math.ceil(optimizer_steps_per_epoch * config.epochs)
    )
    warmup_steps = math.ceil(scheduler_total_steps * config.warmup_ratio)
    return {
        "train_windows": train_windows,
        "epochs": config.epochs,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "expected_optimizer_steps": scheduler_total_steps,
        "warmup_steps": warmup_steps,
        "scheduler_total_steps": scheduler_total_steps,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _data_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _directory_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _attention_backend(config: SFTConfig) -> str:
    if config.attention_backend != "auto":
        return config.attention_backend
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def tokenize_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    require_supervision: bool = True,
) -> dict[str, Any]:
    """Tokenize Qwen chat messages and label only visible trainable assistant content."""
    chat_messages = [{"role": row["role"], "content": row["content"]} for row in messages]
    rendered = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    labels = [-100] * len(input_ids)
    step_token_stats: list[dict[str, Any]] = []

    cursor = 0
    for message_index, message in enumerate(messages):
        content = str(message["content"])
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError("Message content was not preserved by the Qwen chat template")
        end = start + len(content)
        cursor = end
        token_indices = [
            idx
            for idx, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if message.get("loss") is True:
            if message["role"] != "assistant":
                raise ValueError("Only assistant messages may have loss=true")
            for idx in token_indices:
                labels[idx] = input_ids[idx]
        if message["role"] == "assistant" and message.get("step_id") is not None:
            step_token_stats.append(
                {
                    "step_id": str(message["step_id"]),
                    "message_index": message_index,
                    "assistant_tokens": len(token_indices),
                    "supervised_tokens": sum(labels[idx] != -100 for idx in token_indices),
                    "loss": message.get("loss") is True,
                    "mask_reason": message.get("mask_reason"),
                    "window_role": message.get("window_role", "target"),
                }
            )
    if require_supervision and not any(label != -100 for label in labels):
        raise ValueError("SFT window has no assistant action tokens")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "step_token_stats": step_token_stats,
    }


def _action_message_indices(messages: list[dict[str, Any]]) -> list[int]:
    indices = [
        idx
        for idx, message in enumerate(messages)
        if message.get("role") == "assistant"
        and (message.get("message_type") == "appworld_action" or message.get("step_id") is not None)
    ]
    if indices:
        return indices
    indices = sorted(
        {
            idx - 1
            for idx, message in enumerate(messages)
            if idx > 0
            and message.get("message_type") == "appworld_observation"
            and messages[idx - 1].get("role") == "assistant"
        }
    )
    if indices:
        return indices
    first_trainable = next(
        (idx for idx, message in enumerate(messages) if message.get("loss") is True), None
    )
    if first_trainable is None:
        raise ValueError("Sample has no identifiable AppWorld assistant actions")
    return [
        idx
        for idx in range(first_trainable, len(messages))
        if messages[idx].get("role") == "assistant"
    ]


def _normalize_sample_steps(sample: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Add deterministic IDs to legacy JSONL without pretending its old masks are reliable."""
    messages = [dict(message) for message in sample["messages"]]
    action_indices = _action_message_indices(messages)
    metadata = dict(sample.get("metadata") or {})
    task_id = str(metadata.get("task_id") or "unknown-task")
    trajectory_id = metadata.get("trajectory_id")
    legacy = any(messages[idx].get("step_id") is None for idx in action_indices)
    if trajectory_id in (None, ""):
        fingerprint = metadata.get("action_fingerprint")
        if fingerprint:
            trajectory_id = f"legacy-{str(fingerprint)[:20]}"
        else:
            identity = json.dumps(
                messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            trajectory_id = f"legacy-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    metadata["trajectory_id"] = str(trajectory_id)
    metadata["legacy_step_metadata"] = legacy
    for ordinal, message_index in enumerate(action_indices):
        message = messages[message_index]
        message.setdefault("message_type", "appworld_action")
        message.setdefault("step_index", ordinal)
        message.setdefault("step_id", f"{task_id}:{trajectory_id}:{message['step_index']}")
        if message.get("loss") is not True:
            message.setdefault("mask_reason", "legacy_masked")
        for following in range(message_index + 1, len(messages)):
            if following in action_indices:
                break
            if messages[following].get("message_type") == "appworld_observation":
                messages[following].setdefault("step_id", message["step_id"])
                messages[following].setdefault("observation_for_step_id", message["step_id"])
    return {**sample, "messages": messages, "metadata": metadata}, legacy


def _prompt_and_turns(messages: list[dict[str, Any]]) -> tuple[list[Any], list[list[Any]]]:
    action_indices = _action_message_indices(messages)
    if not action_indices:
        raise ValueError("Sample has no AppWorld assistant steps")
    prompt = messages[: action_indices[0]]
    turns: list[list[dict[str, Any]]] = []
    for ordinal, start in enumerate(action_indices):
        end = action_indices[ordinal + 1] if ordinal + 1 < len(action_indices) else len(messages)
        turns.append(messages[start:end])
    return prompt, turns


def _window_messages(
    tokenizer: Any,
    prompt: list[dict[str, Any]],
    turns: list[list[dict[str, Any]]],
    target_start: int,
    target_end: int,
    max_length: int,
    turn_overlap: int,
    preserve_history: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    target = [
        {**message, "window_role": "target"}
        for turn in turns[target_start:target_end]
        for message in turn
    ]

    def render(history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        candidate = [dict(message) for message in prompt] + history + target
        tokenized = tokenize_messages(tokenizer, candidate)
        if len(tokenized["input_ids"]) > max_length:
            return None
        return candidate, tokenized

    if target_start == 0 or not preserve_history:
        return render([])

    earliest = 0 if turn_overlap <= 0 else max(0, target_start - turn_overlap)
    for history_start in range(earliest, target_start):
        history = [
            {
                **message,
                "loss": False,
                "window_role": "history",
                "mask_reason": message.get("mask_reason") or "history_context",
            }
            for turn in turns[history_start:target_start]
            for message in turn
        ]
        rendered = render(history)
        if rendered is not None:
            return rendered

    previous_observations = [
        {
            **message,
            "loss": False,
            "window_role": "history_observation",
        }
        for message in turns[target_start - 1][1:]
        if message.get("role") != "assistant"
    ]
    if not previous_observations:
        raise ValueError(
            f"Turn {target_start} has no preceding observation available for its first target action"
        )
    return render(previous_observations)


def window_sample(
    tokenizer: Any,
    sample: dict[str, Any],
    max_length: int,
    turn_overlap: int = 0,
    preserve_history: bool = True,
) -> list[dict[str, Any]]:
    """Split targets once while retaining maximal masked history at complete boundaries."""
    normalized, _ = _normalize_sample_steps(sample)
    prompt, turns = _prompt_and_turns(normalized["messages"])
    for turn_index, turn in enumerate(turns):
        single_turn = tokenize_messages(tokenizer, prompt + turn, require_supervision=False)
        if len(single_turn["input_ids"]) > max_length:
            step_id = turn[0].get("step_id", turn_index)
            raise ValueError(
                f"One complete prompt+turn exceeds max_length={max_length}: "
                f"turn={turn_index}, step_id={step_id}"
            )
    windows: list[dict[str, Any]] = []
    start = 0
    while start < len(turns):
        best_end: int | None = None
        best_messages: list[dict[str, Any]] | None = None
        best_tokens: dict[str, Any] | None = None
        for end in range(start + 1, len(turns) + 1):
            if not any(
                message.get("loss") is True
                for turn in turns[start:end]
                for message in turn
                if message.get("role") == "assistant"
            ):
                continue
            rendered = _window_messages(
                tokenizer,
                prompt,
                turns,
                start,
                end,
                max_length,
                turn_overlap,
                preserve_history,
            )
            if rendered is None:
                break
            best_messages, best_tokens = rendered
            best_end = end
        if best_end is None or best_tokens is None or best_messages is None:
            raise ValueError(
                f"No complete supervised target range starting at turn {start} fits "
                f"max_length={max_length} with its required preceding observation"
            )
        windows.append(
            {
                **best_tokens,
                "messages": best_messages,
                "metadata": normalized.get("metadata") or {},
                "window_index": len(windows),
                "window_start_turn": start,
                "window_end_turn": best_end,
                "source_turn_count": len(turns),
            }
        )
        if best_end == len(turns):
            break
        start = best_end
    return windows


class TokenizedSFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in features)
        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for row in features:
            padding = max_length - len(row["input_ids"])
            batch["input_ids"].append(
                torch.nn.functional.pad(row["input_ids"], (0, padding), value=self.pad_token_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(row["attention_mask"], (0, padding), value=0)
            )
            batch["labels"].append(torch.nn.functional.pad(row["labels"], (0, padding), value=-100))
        return {key: torch.stack(values) for key, values in batch.items()}


def audit_sample_windows(
    sample: dict[str, Any], windows: list[dict[str, Any]], *, strict: bool = True
) -> dict[str, int]:
    """Verify that target supervision is exactly-once and known bad steps stay masked."""
    normalized, _ = _normalize_sample_steps(sample)
    action_indices = _action_message_indices(normalized["messages"])
    expected = {
        str(normalized["messages"][idx]["step_id"]): normalized["messages"][idx]
        for idx in action_indices
    }
    occurrences: Counter[str] = Counter()
    supervision: Counter[str] = Counter()
    locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    effective_supervised_tokens = 0
    masked_assistant_tokens = 0
    history_occurrences = 0
    for window in windows:
        for record in window.get("step_token_stats") or []:
            step_id = str(record["step_id"])
            occurrences[step_id] += 1
            if int(record["supervised_tokens"]) > 0:
                supervision[step_id] += 1
            if str(record.get("window_role") or "").startswith("history"):
                history_occurrences += 1
            effective_supervised_tokens += int(record["supervised_tokens"])
            masked_assistant_tokens += int(record["assistant_tokens"]) - int(
                record["supervised_tokens"]
            )
            locations[step_id].append(
                {
                    "window": window["window_index"],
                    "role": record.get("window_role"),
                    "loss": record.get("loss"),
                    "supervised_tokens": record.get("supervised_tokens"),
                }
            )

    errors: list[str] = []
    for step_id, message in expected.items():
        expected_supervision = 1 if message.get("loss") is True else 0
        if occurrences[step_id] < 1 or supervision[step_id] != expected_supervision:
            errors.append(
                f"step_id={step_id} occurrences={occurrences[step_id]} "
                f"supervision={supervision[step_id]} expected={expected_supervision} "
                f"windows={locations[step_id]}"
            )
    unknown = sorted(set(occurrences) - set(expected))
    if unknown:
        errors.append(f"unknown step_ids in windows={unknown}")
    history_supervised = [
        step_id
        for step_id, rows in locations.items()
        if any(str(row.get("role") or "").startswith("history") and row.get("loss") for row in rows)
    ]
    if history_supervised:
        errors.append(f"history actions unexpectedly supervised={history_supervised}")
    if errors and strict:
        metadata = normalized.get("metadata") or {}
        raise ValueError(
            "Supervision audit failed: "
            f"task_id={metadata.get('task_id')} trajectory_id={metadata.get('trajectory_id')}; "
            + "; ".join(errors)
        )

    mask_reasons = Counter(
        str(message.get("mask_reason") or "")
        for message in expected.values()
        if message.get("loss") is not True
    )
    other_anomalies = sum(
        count
        for reason, count in mask_reasons.items()
        if reason not in {"execution_failed", "no_code"}
    )
    return {
        "assistant_steps": len(expected),
        "supervised_steps": sum(message.get("loss") is True for message in expected.values()),
        "execution_failed_masked_steps": mask_reasons["execution_failed"],
        "no_code_masked_steps": mask_reasons["no_code"],
        "other_anomaly_masked_steps": other_anomalies,
        "history_repeated_steps": sum(count > 1 for count in occurrences.values()),
        "history_step_occurrences": history_occurrences,
        "effective_supervised_tokens": effective_supervised_tokens,
        "masked_assistant_tokens": masked_assistant_tokens,
    }


def artifact_manifest_callback(manifest: dict[str, Any]) -> Any:
    from transformers import TrainerCallback

    class ArtifactManifestCallback(TrainerCallback):
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            logs = logs or {}
            if "loss" in logs:
                logger.info(
                    "SFT training progress (step=%s, epoch=%s, loss=%s)",
                    state.global_step,
                    state.epoch,
                    logs["loss"],
                    extra={"event": "training_progress"},
                )
            if "eval_loss" in logs:
                logger.info(
                    "SFT validation completed (step=%s, epoch=%s, validation_loss=%s)",
                    state.global_step,
                    state.epoch,
                    logs["eval_loss"],
                    extra={"event": "validation_completed"},
                )

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            checkpoint_manifest = {
                **manifest,
                "checkpoint": {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "path": str(checkpoint.resolve()),
                },
            }
            (checkpoint / "sft_artifact.json").write_text(
                json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n"
            )
            lora_dir = checkpoint / "lora"
            lora_dir.mkdir(exist_ok=True)
            for filename in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "README.md",
            ):
                source = checkpoint / filename
                if source.is_file():
                    shutil.copy2(source, lora_dir / filename)
            if state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-6:
                epoch_link = Path(args.output_dir) / f"epoch-{round(state.epoch)}"
                if epoch_link.is_symlink():
                    epoch_link.unlink()
                if not epoch_link.exists():
                    epoch_link.symlink_to(checkpoint.name, target_is_directory=True)
            logger.info(
                "SFT checkpoint saved (step=%s, epoch=%s)",
                state.global_step,
                state.epoch,
                extra={"event": "training_checkpoint_saved"},
            )

    return ArtifactManifestCallback()


def prepare_windows(
    config: SFTConfig, tokenizer: Any
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_samples = _read_jsonl(config.train_jsonl)
    validation_samples = _read_jsonl(config.validation_jsonl)
    source_train_samples = len(train_samples)
    source_validation_samples = len(validation_samples)
    if config.max_train_samples is not None:
        train_samples = train_samples[: config.max_train_samples]
    if config.max_validation_samples is not None:
        validation_samples = validation_samples[: config.max_validation_samples]
    if not config.preserve_history:
        warnings.warn(
            "preserve_history=False is a risky legacy mode: later windows may not contain the "
            "observation that caused their first target action.",
            stacklevel=2,
        )
    if not config.audit_supervision:
        warnings.warn(
            "audit_supervision=False disables exactly-once step supervision checks.", stacklevel=2
        )

    def prepare_split(samples: list[dict[str, Any]]) -> tuple[list[Any], dict[str, int], int, int]:
        rows: list[dict[str, Any]] = []
        totals: Counter[str] = Counter()
        split_trajectories = 0
        legacy_samples = 0
        for sample in samples:
            normalized, legacy = _normalize_sample_steps(sample)
            legacy_samples += int(legacy)
            windows = window_sample(
                tokenizer,
                normalized,
                config.max_length,
                config.turn_overlap,
                preserve_history=config.preserve_history,
            )
            split_trajectories += int(len(windows) > 1)
            totals.update(
                audit_sample_windows(normalized, windows, strict=config.audit_supervision)
            )
            rows.extend(windows)
        return rows, dict(totals), split_trajectories, legacy_samples

    train, train_audit, train_split_count, train_legacy = prepare_split(train_samples)
    validation, validation_audit, validation_split_count, validation_legacy = prepare_split(
        validation_samples
    )
    legacy_samples = train_legacy + validation_legacy
    if legacy_samples:
        warnings.warn(
            f"Loaded {legacy_samples} legacy SFT samples without per-step failure metadata. "
            "Stable step IDs were synthesized, but execution_failed actions cannot be recovered "
            "reliably from these JSONL files; rebuild them from raw trajectory.json files before training.",
            stacklevel=2,
        )
    combined_audit = Counter(train_audit) + Counter(validation_audit)
    trajectory_count = len(train_samples) + len(validation_samples)
    stats = {
        "source_train_samples": source_train_samples,
        "source_validation_samples": source_validation_samples,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "original_trajectories": trajectory_count,
        "windows": len(train) + len(validation),
        "assistant_steps": combined_audit["assistant_steps"],
        "supervised_steps": combined_audit["supervised_steps"],
        "execution_failed_masked_steps": combined_audit["execution_failed_masked_steps"],
        "no_code_masked_steps": combined_audit["no_code_masked_steps"],
        "other_anomaly_masked_steps": combined_audit["other_anomaly_masked_steps"],
        "history_repeated_steps": combined_audit["history_repeated_steps"],
        "history_step_occurrences": combined_audit["history_step_occurrences"],
        "assistant_tokens": train_audit.get("effective_supervised_tokens", 0),
        "effective_supervised_tokens": combined_audit["effective_supervised_tokens"],
        "masked_assistant_tokens": combined_audit["masked_assistant_tokens"],
        "total_tokens": sum(len(row["input_ids"]) for row in train),
        "trajectory_split_ratio": (
            (train_split_count + validation_split_count) / trajectory_count
            if trajectory_count
            else 0.0
        ),
        "token_truncation_ratio": 0.0,
        "legacy_samples_without_step_metadata": legacy_samples,
        "train_audit": train_audit,
        "validation_audit": validation_audit,
    }
    return train, validation, stats


def train(config: SFTConfig, tokenize_only: bool = False) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Preparing AppWorld SFT data (output_dir=%s)",
        config.output_dir,
        extra={"event": "training_data_preparation_started"},
    )
    model_source = str(config.model_path or config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, validation_rows, stats = prepare_windows(config, tokenizer)
    schedule = training_schedule_preflight(config, len(train_rows))
    schedule["effective_supervised_tokens_per_epoch"] = stats["assistant_tokens"]
    schedule["expected_supervised_token_exposures"] = int(stats["assistant_tokens"] * config.epochs)
    selected_attention_backend = _attention_backend(config)
    manifest = {
        "sft_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "data_sha256": _data_digest((config.train_jsonl, config.validation_jsonl)),
        "train_data_sha256": hashlib.sha256(config.train_jsonl.read_bytes()).hexdigest(),
        "validation_data_sha256": hashlib.sha256(config.validation_jsonl.read_bytes()).hexdigest(),
        "data_stats": stats,
        "base_model": config.model_name,
        "base_model_path": str(config.model_path.resolve()) if config.model_path else None,
        "adapter_kind": "lora",
        "initial_adapter_sha256": _directory_digest(config.initial_adapter),
        "attention_backend": selected_attention_backend,
        "packing": False,
        "lora_target_modules": LORA_TARGET_MODULES,
        "loop_compatible": True,
        "schedule_preflight": schedule,
    }
    (config.output_dir / "sft_artifact.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "AppWorld SFT supervision audit passed "
        "(train_windows=%s, validation_windows=%s, supervised_tokens=%s, legacy_samples=%s)",
        stats["train_windows"],
        stats["validation_windows"],
        stats["assistant_tokens"],
        stats["legacy_samples_without_step_metadata"],
        extra={"event": "training_data_audited"},
    )
    print("SFT_SCHEDULE_PREFLIGHT " + json.dumps(schedule, sort_keys=True), flush=True)
    logger.info(
        "AppWorld SFT schedule confirmed "
        "(train_windows=%s, optimizer_steps=%s, supervised_tokens=%s, warmup_steps=%s, "
        "scheduler_total_steps=%s)",
        schedule["train_windows"],
        schedule["expected_optimizer_steps"],
        schedule["expected_supervised_token_exposures"],
        schedule["warmup_steps"],
        schedule["scheduler_total_steps"],
        extra={"event": "training_schedule_confirmed"},
    )
    if tokenize_only:
        return manifest

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    model_kwargs = {
        "dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "attn_implementation": selected_attention_backend,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    except (ImportError, RuntimeError, ValueError):
        if selected_attention_backend != "flash_attention_2":
            raise
        selected_attention_backend = "sdpa"
        model_kwargs["attn_implementation"] = selected_attention_backend
        manifest["attention_backend"] = selected_attention_backend
        (config.output_dir / "sft_artifact.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        logger.warning(
            "Flash Attention 2 was unavailable; falling back to SDPA.",
            extra={"event": "attention_backend_fallback"},
        )
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if config.initial_adapter is None:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, config.initial_adapter, is_trainable=True)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = False
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise RuntimeError("A non-LoRA model parameter is unexpectedly trainable")

    enable_tf32 = config.tf32 and torch.cuda.is_available() and not config.use_cpu
    if torch.cuda.is_available() and not config.use_cpu:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32

    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        optim="adamw_torch",
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        tf32=enable_tf32,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="epoch" if validation_rows else "no",
        save_strategy="epoch",
        save_total_limit=3,
        report_to="wandb" if os.environ.get("WANDB_PROJECT") else "none",
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        use_cpu=config.use_cpu,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=TokenizedSFTDataset(train_rows),
        eval_dataset=TokenizedSFTDataset(validation_rows) if validation_rows else None,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[artifact_manifest_callback(manifest)],
    )
    if torch.cuda.is_available() and not config.use_cpu:
        torch.cuda.reset_peak_memory_stats()
    logger.info(
        "AppWorld LoRA SFT started "
        "(train_windows=%s, epochs=%s, accumulation=%s, optimizer_steps=%s, "
        "warmup_steps=%s, attention_backend=%s)",
        len(train_rows),
        config.epochs,
        config.gradient_accumulation_steps,
        schedule["expected_optimizer_steps"],
        schedule["warmup_steps"],
        selected_attention_backend,
        extra={"event": "training_started"},
    )
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    eval_metrics = next(
        (
            {key: value for key, value in entry.items() if key.startswith("eval_")}
            for entry in reversed(trainer.state.log_history)
            if "eval_loss" in entry
        ),
        {},
    )
    if validation_rows and not eval_metrics:
        eval_metrics = trainer.evaluate()
    trainer.save_model(str(config.output_dir / "final_adapter"))
    trainer.save_model(str(config.output_dir / "lora"))
    trainer.save_state()
    metrics = {**result.metrics, **eval_metrics, **stats}
    runtime = float(result.metrics.get("train_runtime") or 0.0)
    completed_epochs = float(trainer.state.epoch or 0.0)
    metrics["optimizer_steps"] = trainer.state.global_step
    if (
        config.resume_from_checkpoint is None
        and metrics["optimizer_steps"] != schedule["expected_optimizer_steps"]
    ):
        raise RuntimeError(
            "Optimizer-step count did not match the preflight schedule: "
            f"actual={metrics['optimizer_steps']} expected={schedule['expected_optimizer_steps']}"
        )
    metrics["completed_epochs"] = completed_epochs
    metrics["effective_supervised_tokens_per_epoch"] = stats["assistant_tokens"]
    metrics["effective_supervised_token_exposures"] = int(
        stats["assistant_tokens"] * completed_epochs
    )
    metrics["peak_gpu_memory_bytes"] = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() and not config.use_cpu else 0
    )
    metrics["attention_backend"] = selected_attention_backend
    metrics["expected_optimizer_steps"] = schedule["expected_optimizer_steps"]
    metrics["optimizer_steps_per_epoch"] = schedule["optimizer_steps_per_epoch"]
    metrics["warmup_steps"] = schedule["warmup_steps"]
    metrics["scheduler_total_steps"] = schedule["scheduler_total_steps"]
    metrics["final_learning_rate"] = float(trainer.lr_scheduler.get_last_lr()[0])
    metrics["train_tokens_per_second"] = (
        stats["total_tokens"] * completed_epochs / runtime if runtime else 0.0
    )
    (config.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "AppWorld LoRA SFT completed "
        "(optimizer_steps=%s, train_loss=%s, validation_loss=%s, runtime_seconds=%s)",
        metrics["optimizer_steps"],
        metrics.get("train_loss"),
        metrics.get("eval_loss"),
        runtime,
        extra={"event": "training_completed"},
    )
    return {**manifest, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-7B AppWorld actions.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-length", type=int, default=16_384)
    parser.add_argument("--turn-overlap", type=int, default=0)
    parser.add_argument(
        "--disable-history-context",
        action="store_true",
        help="Risky legacy mode: do not carry masked prior turns into later windows.",
    )
    parser.add_argument(
        "--disable-supervision-audit",
        action="store_true",
        help="Disable exactly-once step supervision validation (not recommended).",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--attention-backend", choices=("auto", "sdpa", "flash_attention_2"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=20_260_713)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


class TrainingTerminated(RuntimeError):
    pass


def _install_termination_handlers() -> dict[signal.Signals, Any]:
    previous: dict[signal.Signals, Any] = {}

    def terminate(signum: int, frame: Any) -> None:
        del frame
        raise TrainingTerminated(signal.Signals(signum).name)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    return previous


def _restore_termination_handlers(previous: dict[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main() -> None:
    args = parse_args()
    config = SFTConfig(
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        model_path=args.model_path,
        max_length=args.max_length,
        turn_overlap=args.turn_overlap,
        preserve_history=not args.disable_history_context,
        audit_supervision=not args.disable_supervision_audit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        tf32=not args.disable_tf32,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        attention_backend=args.attention_backend,
        seed=args.seed,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        initial_adapter=args.initial_adapter,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        use_cpu=args.use_cpu,
    )
    previous_handlers = _install_termination_handlers()
    try:
        result = train(config, tokenize_only=args.tokenize_only)
        print(json.dumps(result, indent=2, sort_keys=True))
    except torch.cuda.OutOfMemoryError:
        logger.critical(
            "AppWorld SFT failed because CUDA ran out of memory.",
            exc_info=True,
            extra={"event": "cuda_oom"},
        )
        raise
    except (KeyboardInterrupt, TrainingTerminated) as exc:
        logger.critical(
            "AppWorld SFT training was interrupted (%s).",
            type(exc).__name__,
            extra={"event": "training_failed"},
        )
        raise
    except BaseException:
        logger.critical(
            "AppWorld SFT training failed.",
            exc_info=True,
            extra={"event": "training_failed"},
        )
        raise
    finally:
        _restore_termination_handlers(previous_handlers)


if __name__ == "__main__":
    main()
