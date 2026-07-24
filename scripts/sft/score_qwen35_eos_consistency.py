from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from scripts.sft.diagnose_qwen35_backend_and_prompt import (
    PROMPT_STYLES,
    prompt_token_ids,
)
from scripts.sft.diagnose_qwen35_turn_boundaries import (
    DEFAULT_TASK_IDS,
    _load_reference_prompts,
    _tokens_prompt,
)


def _load_base_actions(
    diagnostic_path: Path,
    task_ids: list[str],
    stop_token_ids: set[int],
    tokenizer: Any,
    validate_saved_tail: bool,
) -> dict[str, list[int]]:
    diagnostic = json.loads(diagnostic_path.read_text())
    source_rows = {
        str(row["task_id"]): row
        for row in diagnostic["generations"]
        if row["state"] == "base"
    }
    rows: dict[str, list[int]] = {}
    for task_id, row in source_rows.items():
        token_ids = list(
            tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"]
        )
        expected_tail = [
            int(token_id)
            for token_id in row.get("tail_token_ids", [])
            if int(token_id) not in stop_token_ids
        ]
        if (
            validate_saved_tail
            and expected_tail
            and token_ids[-len(expected_tail) :] != expected_tail
        ):
            raise ValueError(
                f"Re-tokenized Base action does not match saved generated tail for {task_id}"
            )
        rows[task_id] = token_ids
    missing = sorted(set(task_ids) - set(rows))
    if missing:
        raise ValueError(f"Base actions are missing from {diagnostic_path}: {missing}")
    for task_id in task_ids:
        while rows[task_id] and rows[task_id][-1] in stop_token_ids:
            rows[task_id].pop()
        if not rows[task_id]:
            raise ValueError(f"Base action for {task_id} is empty")
    return rows


def _score_transformers_batch(
    *,
    model: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    task_ids: list[str],
    eos_token_id: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    lengths = [len(sequence) for sequence in sequences]
    max_length = max(lengths)
    input_ids = torch.full(
        (len(sequences), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, device=device)
        attention_mask[index, : len(sequence)] = 1
    positions = sorted({length - 1 for length in lengths})
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=position_tensor,
            use_cache=False,
        )
    logits = outputs.logits.float()
    results: list[dict[str, Any]] = []
    for row_index, (task_id, length) in enumerate(zip(task_ids, lengths, strict=True)):
        position_index = positions.index(length - 1)
        row_logits = logits[row_index, position_index]
        log_probs = torch.log_softmax(row_logits, dim=-1)
        eos_logprob = float(log_probs[eos_token_id].item())
        top_values, top_ids = torch.topk(log_probs, k=5)
        results.append(
            {
                "task_id": task_id,
                "context_tokens": length,
                "eos_logprob": eos_logprob,
                "eos_probability": math.exp(eos_logprob),
                "eos_rank": int((row_logits > row_logits[eos_token_id]).sum().item() + 1),
                "top_tokens": [
                    {
                        "token_id": int(token_id),
                        "text": tokenizer.decode(
                            [int(token_id)],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "logprob": float(value),
                    }
                    for value, token_id in zip(
                        top_values.tolist(), top_ids.tolist(), strict=True
                    )
                ],
            }
        )
    return results


def _run_transformers(
    *,
    model_path: Path,
    adapter_paths: dict[str, Path],
    states: list[str],
    tokenizer: Any,
    sequences_by_style: dict[str, list[list[int]]],
    task_ids: list[str],
    eos_token_id: int,
    attention_backend: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    model_config = AutoConfig.from_pretrained(model_path)
    model_loader = (
        AutoModelForImageTextToText
        if model_config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = model_loader.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attention_backend,
    )
    adapter_states = [state for state in states if state != "base"]
    first_adapter = adapter_states[0] if adapter_states else None
    if first_adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_paths[first_adapter],
            adapter_name=first_adapter,
        )
        for state in adapter_states[1:]:
            model.load_adapter(adapter_paths[state], adapter_name=state)
    model.eval().cuda()

    rows: list[dict[str, Any]] = []
    for state in states:
        if first_adapter is None:
            adapter_context = nullcontext()
        else:
            model.set_adapter(first_adapter if state == "base" else state)
            adapter_context = model.disable_adapter() if state == "base" else nullcontext()
        with adapter_context:
            for style, sequences in sequences_by_style.items():
                for start in range(0, len(sequences), batch_size):
                    batch_rows = _score_transformers_batch(
                        model=model,
                        tokenizer=tokenizer,
                        sequences=sequences[start : start + batch_size],
                        task_ids=task_ids[start : start + batch_size],
                        eos_token_id=eos_token_id,
                    )
                    for row in batch_rows:
                        row.update({"state": state, "prompt_style": style})
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def _vllm_actual_token_logprob(entry: Any, token_id: int) -> tuple[float, int | None]:
    if entry is None or token_id not in entry:
        raise RuntimeError(f"vLLM did not return the requested prompt token {token_id}")
    value = entry[token_id]
    return float(value.logprob), int(value.rank) if value.rank is not None else None


def _run_vllm(
    *,
    model_path: Path,
    adapter_paths: dict[str, Path],
    states: list[str],
    sequences_by_style: dict[str, list[list[int]]],
    task_ids: list[str],
    eos_token_id: int,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    seed: int,
    eager_mode: bool,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_states = [state for state in states if state != "base"]
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
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    requests = {
        state: LoRARequest(f"qwen35_{state}_eos_consistency", index, str(adapter_paths[state]))
        for index, state in enumerate(adapter_states, start=1)
    }
    rows: list[dict[str, Any]] = []
    for state in states:
        for style, sequences in sequences_by_style.items():
            eos_sequences = [sequence + [eos_token_id] for sequence in sequences]
            outputs = llm.generate(
                [_tokens_prompt(sequence) for sequence in eos_sequences],
                sampling,
                lora_request=requests.get(state),
            )
            for task_id, sequence, output in zip(
                task_ids, sequences, outputs, strict=True
            ):
                eos_logprob, eos_rank = _vllm_actual_token_logprob(
                    output.prompt_logprobs[-1],
                    eos_token_id,
                )
                row = {
                    "state": state,
                    "prompt_style": style,
                    "task_id": task_id,
                    "context_tokens": len(sequence),
                    "eos_logprob": eos_logprob,
                    "eos_probability": math.exp(eos_logprob),
                    "eos_rank": eos_rank,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    keys = sorted({(row["state"], row["prompt_style"]) for row in rows})
    for state, style in keys:
        selected = [
            row
            for row in rows
            if row["state"] == state and row["prompt_style"] == style
        ]
        aggregates[f"{state}:{style}"] = {
            "state": state,
            "prompt_style": style,
            "samples": len(selected),
            "mean_eos_logprob": sum(row["eos_logprob"] for row in selected)
            / len(selected),
            "mean_eos_probability": sum(row["eos_probability"] for row in selected)
            / len(selected),
            "median_eos_rank": sorted(row["eos_rank"] for row in selected)[
                len(selected) // 2
            ],
        }
    return aggregates


def _write_report(path: Path, backend: str, aggregates: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Qwen3.5 End-of-Turn Consistency",
        "",
        f"Backend: `{backend}`. Each row scores `<|im_end|>` immediately after the same "
        "clean Base-generated one-action response. Generated code is not executed.",
        "",
        "| State | Prompt style | Samples | Mean EOS logprob | Mean EOS probability | "
        "Median EOS rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(aggregates):
        row = aggregates[key]
        lines.append(
            f"| {row['state']} | {row['prompt_style']} | {row['samples']} | "
            f"{row['mean_eos_logprob']:.6f} | {row['mean_eos_probability']:.6g} | "
            f"{row['median_eos_rank']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("transformers", "vllm"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--d12-adapter", type=Path, required=True)
    parser.add_argument("--d3-adapter", type=Path)
    parser.add_argument("--d12-step10-adapter", type=Path)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional named adapter; may be repeated.",
    )
    parser.add_argument("--base-generation-diagnostic", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--prompt-styles", default=",".join(PROMPT_STYLES))
    parser.add_argument("--states", default="base,d12_step10,d12,d3")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=2_026_072_200)
    parser.add_argument("--eager-mode", action="store_true")
    parser.add_argument("--allow-retokenized-base-actions", action="store_true")
    parser.add_argument("--append-token-ids", default="")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=False)
    task_ids = [value.strip() for value in args.task_ids.split(",") if value.strip()]
    styles = [value.strip() for value in args.prompt_styles.split(",") if value.strip()]
    states = [value.strip() for value in args.states.split(",") if value.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token_id = int(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    endoftext_id = int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    base_actions = _load_base_actions(
        args.base_generation_diagnostic,
        task_ids,
        {eos_token_id, endoftext_id},
        tokenizer,
        validate_saved_tail=not args.allow_retokenized_base_actions,
    )
    prompts, prompt_metadata = _load_reference_prompts(args.episodes_root, task_ids)
    appended_token_ids = [
        int(value.strip()) for value in args.append_token_ids.split(",") if value.strip()
    ]
    sequences_by_style = {
        style: [
            prompt_token_ids(tokenizer, prompt, style)
            + base_actions[task_id]
            + appended_token_ids
            for task_id, prompt in zip(task_ids, prompts, strict=True)
        ]
        for style in styles
    }
    adapter_paths = {"d12": args.d12_adapter}
    if args.d3_adapter is not None:
        adapter_paths["d3"] = args.d3_adapter
    if args.d12_step10_adapter is not None:
        adapter_paths["d12_step10"] = args.d12_step10_adapter
    for specification in args.adapter:
        name, separator, path = specification.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"Expected --adapter NAME=PATH, got {specification!r}")
        name = name.strip()
        if name == "base" or name in adapter_paths:
            raise ValueError(f"Duplicate or reserved adapter name: {name}")
        adapter_paths[name] = Path(path.strip())
    for state in states:
        if state != "base" and state not in adapter_paths:
            raise ValueError(f"Missing adapter path for {state}")

    common = {
        "model_path": args.model_path,
        "adapter_paths": adapter_paths,
        "states": states,
        "sequences_by_style": sequences_by_style,
        "task_ids": task_ids,
        "eos_token_id": eos_token_id,
    }
    if args.backend == "transformers":
        rows = _run_transformers(
            **common,
            tokenizer=tokenizer,
            attention_backend=args.attention_backend,
            batch_size=args.batch_size,
        )
    else:
        rows = _run_vllm(
            **common,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            eager_mode=args.eager_mode,
        )
    aggregates = _aggregate(rows)
    result = {
        "backend": args.backend,
        "eos_token_id": eos_token_id,
        "appended_token_ids": appended_token_ids,
        "appended_text": tokenizer.decode(
            appended_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "source_base_generation_diagnostic": str(
            args.base_generation_diagnostic.resolve()
        ),
        "prompts": prompt_metadata,
        "aggregates": aggregates,
        "rows": rows,
    }
    (args.output_dir / "scores.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_report(args.output_dir / "report.md", args.backend, aggregates)


if __name__ == "__main__":
    main()
