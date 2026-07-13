from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SFTConfig:
    train_jsonl: Path
    validation_jsonl: Path
    output_dir: Path
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 16_384
    turn_overlap: int = 0
    preserve_history: bool = True
    audit_supervision: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    learning_rate: float = 2e-5
    epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42
    resume_from_checkpoint: str | None = None
    use_cpu: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _data_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "sft_artifact.json"
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return ArtifactManifestCallback()


def prepare_windows(
    config: SFTConfig, tokenizer: Any
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_samples = _read_jsonl(config.train_jsonl)
    validation_samples = _read_jsonl(config.validation_jsonl)
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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, validation_rows, stats = prepare_windows(config, tokenizer)
    manifest = {
        "sft_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "data_sha256": _data_digest((config.train_jsonl, config.validation_jsonl)),
        "data_stats": stats,
        "base_model": config.model_name,
        "adapter_kind": "lora",
        "loop_compatible": True,
    }
    (config.output_dir / "sft_artifact.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if tokenize_only:
        return manifest

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = False
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ):
        raise RuntimeError("A non-LoRA model parameter is unexpectedly trainable")

    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if validation_rows else "no",
        save_strategy="steps",
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
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    eval_metrics = trainer.evaluate() if validation_rows else {}
    trainer.save_model(str(config.output_dir / "final_adapter"))
    trainer.save_state()
    metrics = {**result.metrics, **eval_metrics, **stats}
    runtime = float(result.metrics.get("train_runtime") or 0.0)
    metrics["train_tokens_per_second"] = (
        stats["total_tokens"] * config.epochs / runtime if runtime else 0.0
    )
    (config.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return {**manifest, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-7B AppWorld actions.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
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
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SFTConfig(
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        turn_overlap=args.turn_overlap,
        preserve_history=not args.disable_history_context,
        audit_supervision=not args.disable_supervision_audit,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_cpu=args.use_cpu,
    )
    print(json.dumps(train(config, tokenize_only=args.tokenize_only), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
