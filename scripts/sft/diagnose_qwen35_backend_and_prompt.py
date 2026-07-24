from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from scripts.sft.diagnose_qwen35_turn_boundaries import (
    DEFAULT_TASK_IDS,
    _fingerprint_directory,
    _load_reference_prompts,
    _tokens_prompt,
    exact_prompt_token_ids,
    python_code_block_count,
    summarize_generation,
)

PROMPT_STYLES = ("eval_disabled_thinking", "training_no_thinking_scaffold")
_ASSISTANT_MARKER = "__APPWORLD_QWEN35_ASSISTANT_MARKER_7B72E5__"
_FOLLOWUP_MARKER = "__APPWORLD_QWEN35_FOLLOWUP_MARKER_27F8A1__"


def _flat_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token_id, int) for token_id in value):
        raise TypeError(f"Expected flat token IDs, got {type(value).__name__}")
    return value


def training_no_thinking_prompt_token_ids(
    tokenizer: Any, messages: list[dict[str, str]]
) -> list[int]:
    """Build the assistant prefix produced for actions followed by a user observation."""
    if any(
        marker in str(message.get("content") or "")
        for marker in (_ASSISTANT_MARKER, _FOLLOWUP_MARKER)
        for message in messages
    ):
        raise ValueError("Diagnostic marker unexpectedly occurs in the prompt")
    probe_messages = [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]
    probe_messages.extend(
        [
            {"role": "assistant", "content": _ASSISTANT_MARKER},
            {"role": "user", "content": _FOLLOWUP_MARKER},
        ]
    )
    rendered = str(
        tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    marker_index = rendered.find(_ASSISTANT_MARKER)
    if marker_index < 0 or rendered.find(_ASSISTANT_MARKER, marker_index + 1) >= 0:
        raise ValueError("Could not uniquely locate the synthetic assistant content")
    prefix = rendered[:marker_index]
    if not prefix.endswith("<|im_start|>assistant\n"):
        raise ValueError(f"Unexpected training-style assistant prefix: {prefix[-96:]!r}")
    return _flat_token_ids(tokenizer(prefix, add_special_tokens=False))


def prompt_token_ids(
    tokenizer: Any, messages: list[dict[str, str]], prompt_style: str
) -> list[int]:
    if prompt_style == "eval_disabled_thinking":
        return exact_prompt_token_ids(tokenizer, messages)
    if prompt_style == "training_no_thinking_scaffold":
        return training_no_thinking_prompt_token_ids(tokenizer, messages)
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def _summarize_generation(
    *,
    state: str,
    prompt_style: str,
    task_id: str,
    token_ids: list[int],
    tokenizer: Any,
    stop_token_ids: set[int],
) -> dict[str, Any]:
    stopped = bool(token_ids) and token_ids[-1] in stop_token_ids
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "state": state,
        "prompt_style": prompt_style,
        "task_id": task_id,
        "finish_reason": "stop" if stopped else "length",
        "stop_token_hit": stopped,
        "generated_tokens": len(token_ids),
        "tail_token_ids": token_ids[-16:],
        "python_code_blocks": python_code_block_count(text),
        "text": text,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "finish_stop": sum(row["finish_reason"] == "stop" for row in rows),
        "finish_length": sum(row["finish_reason"] == "length" for row in rows),
        "one_code_block": sum(row["python_code_blocks"] == 1 for row in rows),
        "multiple_code_blocks": sum(row["python_code_blocks"] > 1 for row in rows),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in rows) / len(rows), 2
        ),
    }


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    stop_token_ids: set[int],
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> list[int]:
    device = next(model.parameters()).device
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(inputs)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generated = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        max_new_tokens=max_new_tokens,
        eos_token_id=sorted(stop_token_ids),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    return generated[0, inputs.shape[1] :].tolist()


def _generate_with_transformers(
    *,
    model_path: Path,
    adapter_paths: dict[str, Path],
    requested_states: list[str],
    styles: list[str],
    task_ids: list[str],
    all_prompt_ids: dict[str, list[list[int]]],
    tokenizer: Any,
    stop_token_ids: set[int],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    attention_backend: str,
) -> list[dict[str, Any]]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    )
    adapter_states = [state for state in requested_states if state != "base"]
    first_adapter_state = adapter_states[0] if adapter_states else None
    if first_adapter_state is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_paths[first_adapter_state],
            adapter_name=first_adapter_state,
        )
        for state in adapter_states[1:]:
            model.load_adapter(adapter_paths[state], adapter_name=state)
    model.eval().cuda()

    rows: list[dict[str, Any]] = []
    for state in requested_states:
        if first_adapter_state is None:
            adapter_context = nullcontext()
        else:
            model.set_adapter(first_adapter_state if state == "base" else state)
            adapter_context = model.disable_adapter() if state == "base" else nullcontext()
        with adapter_context, torch.inference_mode():
            for style in styles:
                for task_index, (task_id, input_ids) in enumerate(
                    zip(task_ids, all_prompt_ids[style], strict=True)
                ):
                    generated_ids = _generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        stop_token_ids=stop_token_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=seed + task_index,
                    )
                    row = _summarize_generation(
                        state=state,
                        prompt_style=style,
                        task_id=task_id,
                        token_ids=generated_ids,
                        tokenizer=tokenizer,
                        stop_token_ids=stop_token_ids,
                    )
                    rows.append(row)
                    _print_progress(row)
    return rows


def _generate_with_vllm(
    *,
    model_path: Path,
    adapter_paths: dict[str, Path],
    requested_states: list[str],
    styles: list[str],
    task_ids: list[str],
    all_prompt_ids: dict[str, list[list[int]]],
    stop_token_ids: set[int],
    max_model_len: int,
    max_new_tokens: int,
    max_num_seqs: int,
    temperature: float,
    seed: int,
    gpu_memory_utilization: float,
    eager_mode: bool,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_states = [state for state in requested_states if state != "base"]
    llm = LLM(
        model=str(model_path),
        enable_lora=bool(adapter_states),
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=max(1, len(adapter_states)),
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=eager_mode,
        enable_prefix_caching=True,
        language_model_only=True,
        seed=seed,
    )
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=sorted(stop_token_ids),
        skip_special_tokens=False,
    )
    requests = {
        state: LoRARequest(f"qwen35_{state}_engineering_diagnostic", index, str(path))
        for index, (state, path) in enumerate(
            (
                (state, adapter_paths[state])
                for state in adapter_states
            ),
            start=1,
        )
    }
    rows: list[dict[str, Any]] = []
    for state in requested_states:
        for style in styles:
            prompts = [_tokens_prompt(token_ids) for token_ids in all_prompt_ids[style]]
            outputs = llm.generate(
                prompts,
                sampling,
                lora_request=requests.get(state),
            )
            for task_id, output in zip(task_ids, outputs, strict=True):
                candidate = output.outputs[0]
                row = summarize_generation(
                    state=state,
                    task_id=task_id,
                    text=candidate.text,
                    token_ids=list(candidate.token_ids),
                    finish_reason=candidate.finish_reason,
                    stop_reason=candidate.stop_reason,
                    stop_token_ids=stop_token_ids,
                )
                row["prompt_style"] = style
                rows.append(row)
                _print_progress(row)
    return rows


def _print_progress(row: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                key: row[key]
                for key in (
                    "state",
                    "prompt_style",
                    "task_id",
                    "finish_reason",
                    "generated_tokens",
                    "python_code_blocks",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _write_report(
    output_path: Path,
    aggregates: dict[str, dict[str, Any]],
    prompt_suffixes: dict[str, dict[str, Any]],
    backend: str,
) -> None:
    backend_description = (
        "Transformers + PEFT on one model instance"
        if backend == "transformers"
        else "one vLLM instance with dynamically loaded LoRA adapters"
    )
    lines = [
        "# Qwen3.5 Backend and Prompt-Scaffold Diagnostic",
        "",
        f"All generations use {backend_description}. No generated code is executed. "
        "The two prompt styles differ only in the assistant-generation suffix.",
        "",
        "## Prompt Suffixes",
        "",
        "| Style | Suffix tokens | Decoded suffix |",
        "|---|---:|---|",
    ]
    for style in prompt_suffixes:
        suffix = prompt_suffixes[style]
        decoded = str(suffix["decoded"]).replace("\n", "\\n").replace("|", "\\|")
        lines.append(f"| {style} | {suffix['tokens']} | `{decoded}` |")
    lines.extend(
        [
            "",
            "## Generation Results",
            "",
            "| State | Prompt style | Stop | Length | One block | Multi-block | Mean tokens |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in sorted(aggregates):
        row = aggregates[key]
        lines.append(
            f"| {row['state']} | {row['prompt_style']} | "
            f"{row['finish_stop']}/{row['samples']} | "
            f"{row['finish_length']}/{row['samples']} | "
            f"{row['one_code_block']}/{row['samples']} | "
            f"{row['multiple_code_blocks']}/{row['samples']} | "
            f"{row['mean_generated_tokens']:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--d12-adapter", type=Path, required=True)
    parser.add_argument("--d3-adapter", type=Path, required=True)
    parser.add_argument("--d12-step10-adapter", type=Path)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--prompt-styles", default=",".join(PROMPT_STYLES))
    parser.add_argument("--states", default="base,d12_step10,d12,d3")
    parser.add_argument("--max-new-tokens", type=int, default=1_200)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2_026_072_200)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--eager-mode", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    task_ids = [value.strip() for value in args.task_ids.split(",") if value.strip()]
    styles = [value.strip() for value in args.prompt_styles.split(",") if value.strip()]
    requested_states = [value.strip() for value in args.states.split(",") if value.strip()]
    unknown_styles = sorted(set(styles) - set(PROMPT_STYLES))
    if unknown_styles:
        raise ValueError(f"Unknown prompt styles: {unknown_styles}")

    adapter_paths = {
        "d12": args.d12_adapter,
        "d3": args.d3_adapter,
    }
    if args.d12_step10_adapter is not None:
        adapter_paths["d12_step10"] = args.d12_step10_adapter
    for state in requested_states:
        if state == "base":
            continue
        if state not in adapter_paths:
            raise ValueError(f"No adapter path was provided for requested state {state!r}")
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_paths[state] / filename).is_file():
                raise FileNotFoundError(adapter_paths[state] / filename)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, prompt_metadata = _load_reference_prompts(args.episodes_root, task_ids)
    all_prompt_ids = {
        style: [prompt_token_ids(tokenizer, messages, style) for messages in prompts]
        for style in styles
    }
    prompt_suffixes: dict[str, dict[str, Any]] = {}
    for style in styles:
        no_suffix = [
            tokenizer.apply_chat_template(
                [
                    {"role": message["role"], "content": message["content"]}
                    for message in prompts[0]
                ],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        ]
        base_ids = _flat_token_ids(no_suffix[0])
        suffix_ids = all_prompt_ids[style][0][len(base_ids) :]
        prompt_suffixes[style] = {
            "token_ids": suffix_ids,
            "tokens": len(suffix_ids),
            "decoded": tokenizer.decode(
                suffix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }

    special_tokens = {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<|im_end|>", "<|endoftext|>")
    }
    stop_token_ids = set(special_tokens.values())
    generation_kwargs = {
        "model_path": args.model_path,
        "adapter_paths": adapter_paths,
        "requested_states": requested_states,
        "styles": styles,
        "task_ids": task_ids,
        "all_prompt_ids": all_prompt_ids,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.backend == "transformers":
        rows = _generate_with_transformers(
            **generation_kwargs,
            tokenizer=tokenizer,
            attention_backend=args.attention_backend,
        )
    else:
        rows = _generate_with_vllm(
            **generation_kwargs,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            eager_mode=args.eager_mode,
        )

    aggregates: dict[str, dict[str, Any]] = {}
    for state in requested_states:
        for style in styles:
            selected = [
                row
                for row in rows
                if row["state"] == state and row["prompt_style"] == style
            ]
            aggregate = _aggregate(selected)
            aggregates[f"{state}:{style}"] = {
                "state": state,
                "prompt_style": style,
                **aggregate,
            }
    result = {
        "backend": "transformers_peft" if args.backend == "transformers" else "vllm",
        "generated_code_executed": False,
        "inference": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "attention_backend": args.attention_backend,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "eager_mode": args.eager_mode,
            "stop_tokens": special_tokens,
        },
        "prompt_suffixes": prompt_suffixes,
        "prompts": [
            {
                **metadata,
                "token_counts": {
                    style: len(all_prompt_ids[style][index]) for style in styles
                },
                "token_sha256": {
                    style: hashlib.sha256(
                        json.dumps(
                            all_prompt_ids[style][index], separators=(",", ":")
                        ).encode()
                    ).hexdigest()
                    for style in styles
                },
            }
            for index, metadata in enumerate(prompt_metadata)
        ],
        "fingerprints": {
            "base": _fingerprint_directory(args.model_path, metadata_only=True),
            **{
                state: _fingerprint_directory(path, metadata_only=False, adapter=True)
                for state, path in adapter_paths.items()
                if state in requested_states
            },
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
    _write_report(args.output_dir / "report.md", aggregates, prompt_suffixes, args.backend)


if __name__ == "__main__":
    main()
