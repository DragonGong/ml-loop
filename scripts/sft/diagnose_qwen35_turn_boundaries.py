from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_TASK_IDS = (
    "0d8a4ee_1",
    "23cf851_1",
    "383cbac_1",
    "4fab96f_1",
    "fac291d_1",
)
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token_id, int) for token_id in value):
        raise TypeError(f"Expected flat token IDs, got {type(value).__name__}")
    return value


def exact_prompt_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Mirror VLLMQwen3's initial-turn tokenization, including thinking-disabled suffix."""
    template_messages = [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]
    prompt_without_suffix = _flat_token_ids(
        tokenizer.apply_chat_template(
            template_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    dummy_messages = [
        {"role": "system", "content": "dummy"},
        {"role": "user", "content": "dummy"},
    ]
    dummy_without_suffix = _flat_token_ids(
        tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    dummy_with_suffix = _flat_token_ids(
        tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if dummy_without_suffix != dummy_with_suffix[: len(dummy_without_suffix)]:
        raise ValueError("Qwen3.5 generation prompt is not an append-only suffix")
    generation_suffix = dummy_with_suffix[len(dummy_without_suffix) :]
    prompt_token_ids = prompt_without_suffix + generation_suffix

    direct = _flat_token_ids(
        tokenizer.apply_chat_template(
            template_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if prompt_token_ids != direct:
        raise ValueError("VLLMQwen3-compatible prompt differs from direct chat-template output")
    return prompt_token_ids


def python_code_block_count(text: str) -> int:
    return len(re.findall(r"```(?:python|py)\s*(?:\r?\n)", text, flags=re.IGNORECASE))


def summarize_generation(
    *,
    state: str,
    task_id: str,
    text: str,
    token_ids: list[int],
    finish_reason: str | None,
    stop_reason: int | str | None,
    stop_token_ids: set[int],
) -> dict[str, Any]:
    code_blocks = python_code_block_count(text)
    stop_token_hit = finish_reason == "stop" and (
        isinstance(stop_reason, int)
        and stop_reason in stop_token_ids
        or isinstance(stop_reason, str)
        and stop_reason.isdigit()
        and int(stop_reason) in stop_token_ids
    )
    return {
        "state": state,
        "task_id": task_id,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "stop_token_hit": stop_token_hit,
        "generated_tokens": len(token_ids),
        "tail_token_ids": token_ids[-16:],
        "python_code_blocks": code_blocks,
        "stopped_after_exactly_one_action": finish_reason == "stop" and code_blocks == 1,
        "text": text,
    }


def aggregate_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty generation set")
    return {
        "samples": len(rows),
        "finish_stop": sum(row["finish_reason"] == "stop" for row in rows),
        "finish_length": sum(row["finish_reason"] == "length" for row in rows),
        "stop_token_hit": sum(bool(row["stop_token_hit"]) for row in rows),
        "exactly_one_code_block": sum(row["python_code_blocks"] == 1 for row in rows),
        "multiple_code_blocks": sum(row["python_code_blocks"] > 1 for row in rows),
        "stopped_after_exactly_one_action": sum(
            bool(row["stopped_after_exactly_one_action"]) for row in rows
        ),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in rows) / len(rows), 2
        ),
    }


def classify_result(aggregates: dict[str, dict[str, Any]]) -> str:
    base = aggregates["base"]
    adapters = [aggregates["d12"], aggregates["d3"]]
    base_normal = base["stopped_after_exactly_one_action"] >= max(1, base["samples"] - 1)
    adapters_broken = all(
        row["multiple_code_blocks"] + row["finish_length"] >= max(1, row["samples"] - 1)
        for row in adapters
    )
    if base_normal and adapters_broken:
        return "adapter_specific_turn_boundary_failure"
    if base["multiple_code_blocks"] + base["finish_length"] >= max(1, base["samples"] - 1):
        return "shared_inference_or_prompt_failure"
    return "mixed_or_inconclusive"


def _load_reference_prompts(
    episodes_root: Path, task_ids: list[str]
) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    prompts: list[list[dict[str, str]]] = []
    metadata: list[dict[str, Any]] = []
    for task_id in task_ids:
        episode_path = episodes_root / task_id / "logs" / "episode.json"
        episode = json.loads(episode_path.read_text())
        if episode["task"]["task_id"] != task_id:
            raise ValueError(f"Task mismatch in {episode_path}")
        prompt_message_count = int(episode["num_prompt_messages"])
        messages = episode["chat_history"][:prompt_message_count]
        if len(messages) != prompt_message_count:
            raise ValueError(f"Incomplete saved prompt in {episode_path}")
        compact = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
        if compact[-1]["role"] != "user":
            raise ValueError(f"Expected final prompt message to be user in {episode_path}")
        prompts.append(compact)
        metadata.append(
            {
                "task_id": task_id,
                "episode_path": str(episode_path.resolve()),
                "prompt_messages": prompt_message_count,
                "instruction": episode["task"]["instruction"],
            }
        )
    return prompts, metadata


def _fingerprint_directory(
    path: Path, *, metadata_only: bool, adapter: bool = False
) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    names = (
        ("adapter_config.json", "adapter_model.safetensors", "README.md")
        if adapter
        else MODEL_METADATA_FILES
    )
    files = [path / name for name in names if (path / name).is_file()]
    if not metadata_only and not adapter:
        files.extend(sorted(path.glob("*.safetensors")))
    unique_files = sorted(set(files))
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "files": {
            child.name: {"bytes": child.stat().st_size, "sha256": _sha256(child)}
            for child in unique_files
        },
    }
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        config = json.loads(adapter_config.read_text())
        result["base_model_name_or_path"] = config.get("base_model_name_or_path")
        result["peft_type"] = config.get("peft_type")
        result["rank"] = config.get("r")
    return result


def _tokens_prompt(prompt_token_ids: list[int]) -> Any:
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=prompt_token_ids)
    except ImportError:
        return {"prompt_token_ids": prompt_token_ids}


def _write_report(
    output: Path,
    aggregates: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    classification: str,
) -> None:
    lines = [
        "# Qwen3.5 First-Turn Boundary Diagnostic",
        "",
        "All three states used the same vLLM instance, tokenizer, saved formal-eval prompts, "
        "sampling parameters, stop-token IDs, and 16K context. Generated code was not executed.",
        "",
        "## Aggregate",
        "",
        "| State | Stop | Length | Stop token | One block | Multi-block | "
        "Clean one-action stop | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for state in ("base", "d12", "d3"):
        row = aggregates[state]
        lines.append(
            f"| {state} | {row['finish_stop']}/{row['samples']} | "
            f"{row['finish_length']}/{row['samples']} | "
            f"{row['stop_token_hit']}/{row['samples']} | "
            f"{row['exactly_one_code_block']}/{row['samples']} | "
            f"{row['multiple_code_blocks']}/{row['samples']} | "
            f"{row['stopped_after_exactly_one_action']}/{row['samples']} | "
            f"{row['mean_generated_tokens']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Automatic classification: `{classification}`.",
            "",
            "## Per Task",
            "",
            "| Task | State | Finish | Stop reason | Tokens | Python blocks | "
            "Clean one-action stop |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['state']} | {row['finish_reason']} | "
            f"{row['stop_reason']} | {row['generated_tokens']} | "
            f"{row['python_code_blocks']} | "
            f"{'yes' if row['stopped_after_exactly_one_action'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Raw first-turn text, generated token tails, prompt hashes, and fingerprints are in "
            "`diagnostic.json`.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--d12-adapter", type=Path, required=True)
    parser.add_argument("--d3-adapter", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--max-new-tokens", type=int, default=1_200)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2_026_072_200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--eager-mode", action="store_true")
    parser.add_argument("--hash-model-shards", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    task_ids = [value.strip() for value in args.task_ids.split(",") if value.strip()]
    if not task_ids:
        raise ValueError("At least one task ID is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for adapter_path in (args.d12_adapter, args.d3_adapter):
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_path / filename).is_file():
                raise FileNotFoundError(adapter_path / filename)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    prompts, prompt_metadata = _load_reference_prompts(args.episodes_root, task_ids)
    prompt_token_ids = [exact_prompt_token_ids(tokenizer, prompt) for prompt in prompts]
    for metadata, token_ids in zip(prompt_metadata, prompt_token_ids, strict=True):
        metadata["prompt_tokens"] = len(token_ids)
        metadata["prompt_token_sha256"] = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest()

    special_tokens = {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
    }
    if tokenizer.unk_token_id in special_tokens.values():
        raise ValueError(f"Missing required Qwen stop token: {special_tokens}")
    stop_token_ids = set(special_tokens.values())

    llm = LLM(
        model=str(args.model_path),
        enable_lora=True,
        max_lora_rank=32,
        max_loras=2,
        max_cpu_loras=2,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager_mode,
        enable_prefix_caching=True,
        language_model_only=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        stop_token_ids=sorted(stop_token_ids),
        skip_special_tokens=False,
    )
    vllm_prompts = [_tokens_prompt(token_ids) for token_ids in prompt_token_ids]
    states = (
        ("base", None),
        ("d12", LoRARequest("qwen35_d12_boundary_diagnostic", 1, str(args.d12_adapter))),
        ("d3", LoRARequest("qwen35_d3_boundary_diagnostic", 2, str(args.d3_adapter))),
    )
    rows: list[dict[str, Any]] = []
    for state, lora_request in states:
        outputs = llm.generate(vllm_prompts, sampling, lora_request=lora_request)
        if len(outputs) != len(task_ids):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(task_ids)} tasks")
        for task_id, output in zip(task_ids, outputs, strict=True):
            candidate = output.outputs[0]
            rows.append(
                summarize_generation(
                    state=state,
                    task_id=task_id,
                    text=candidate.text,
                    token_ids=list(candidate.token_ids),
                    finish_reason=candidate.finish_reason,
                    stop_reason=candidate.stop_reason,
                    stop_token_ids=stop_token_ids,
                )
            )

    aggregates = {
        state: aggregate_state([row for row in rows if row["state"] == state])
        for state in ("base", "d12", "d3")
    }
    classification = classify_result(aggregates)
    result = {
        "status": "passed",
        "classification": classification,
        "inference": {
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "max_num_seqs": args.max_num_seqs,
            "temperature": args.temperature,
            "seed": args.seed,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "eager_mode": args.eager_mode,
            "enable_thinking": False,
            "stop_tokens": special_tokens,
            "generated_code_executed": False,
        },
        "prompts": prompt_metadata,
        "fingerprints": {
            "base": _fingerprint_directory(
                args.model_path, metadata_only=not args.hash_model_shards
            ),
            "d12": _fingerprint_directory(args.d12_adapter, metadata_only=False, adapter=True),
            "d3": _fingerprint_directory(args.d3_adapter, metadata_only=False, adapter=True),
        },
        "aggregates": aggregates,
        "generations": rows,
    }
    (args.output_dir / "diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "raw_generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_report(args.output_dir / "report.md", aggregates, rows, classification)
    print(json.dumps({**aggregates, "classification": classification}, indent=2))


if __name__ == "__main__":
    main()
