#!/usr/bin/env python3
"""Merge an SFT LoRA into Qwen and verify the initialization used by LOOP.

The heavyweight model imports are intentionally kept inside ``prepare`` so the
manifest, hashing, and comparison helpers can be unit-tested without loading a
model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "sft_loop_merge_manifest.json"
SCHEMA_VERSION = "appworld-sft-loop-merge-v1"
LOOP_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    messages: tuple[dict[str, str], ...]


@dataclass
class ProbeSnapshot:
    prompt_hashes: tuple[str, ...]
    next_token_logits: tuple[Any, ...]
    generated_token_ids: tuple[tuple[int, ...], ...]


DEFAULT_PROBES = (
    ProbeSpec(
        name="appworld_api_action",
        messages=(
            {
                "role": "system",
                "content": "You are an AppWorld agent. Respond with the next executable action only.",
            },
            {
                "role": "user",
                "content": "Inspect the available APIs needed to find a saved music item.",
            },
        ),
    ),
    ProbeSpec(
        name="appworld_recovery",
        messages=(
            {
                "role": "system",
                "content": "You are an AppWorld agent. Recover from errors using valid API calls.",
            },
            {
                "role": "user",
                "content": "The previous API call used an invalid argument. Produce the corrected next action.",
            },
        ),
    ),
    ProbeSpec(
        name="appworld_completion",
        messages=(
            {
                "role": "system",
                "content": "You are an AppWorld agent. Complete tasks only after all required actions succeed.",
            },
            {
                "role": "user",
                "content": "All requested changes are now verified. Produce the completion action.",
            },
        ),
    ),
)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_weight_files(output_dir: Path) -> dict[str, Any]:
    weight_paths = sorted(output_dir.glob("*.safetensors"))
    if not weight_paths:
        raise ValueError(f"No safetensors weights found in output directory: {output_dir}")

    combined = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in weight_paths:
        relative_path = path.relative_to(output_dir).as_posix()
        file_digest = sha256_file(path)
        size_bytes = path.stat().st_size
        files.append(
            {
                "path": relative_path,
                "size_bytes": size_bytes,
                "sha256": file_digest,
            }
        )
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(file_digest))

    return {
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "combined_sha256": combined.hexdigest(),
        "combined_sha256_definition": "sha256(relative_path + NUL + raw_file_sha256), sorted",
    }


def _prompt_hash(spec: ProbeSpec) -> str:
    canonical = json.dumps(spec.messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compare_probe_snapshots(
    reference: ProbeSnapshot,
    candidate: ProbeSnapshot,
) -> dict[str, Any]:
    """Compare compact probe outputs without serializing full vocabulary logits."""
    import torch
    import torch.nn.functional as functional

    if reference.prompt_hashes != candidate.prompt_hashes:
        raise ValueError("Probe prompt hashes do not match")
    if len(reference.next_token_logits) != len(candidate.next_token_logits):
        raise ValueError("Probe snapshot lengths do not match")
    if len(reference.generated_token_ids) != len(candidate.generated_token_ids):
        raise ValueError("Generated sequence counts do not match")

    per_prompt: list[dict[str, Any]] = []
    total_absolute_difference = 0.0
    total_elements = 0
    for index, (reference_logits, candidate_logits) in enumerate(
        zip(reference.next_token_logits, candidate.next_token_logits, strict=True)
    ):
        reference_tensor = torch.as_tensor(reference_logits).detach().float().cpu().flatten()
        candidate_tensor = torch.as_tensor(candidate_logits).detach().float().cpu().flatten()
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(
                f"Probe logit shape mismatch at index {index}: "
                f"{tuple(reference_tensor.shape)} != {tuple(candidate_tensor.shape)}"
            )

        difference = (candidate_tensor - reference_tensor).abs()
        finite = bool(
            torch.isfinite(reference_tensor).all() and torch.isfinite(candidate_tensor).all()
        )
        cosine = float(
            functional.cosine_similarity(
                reference_tensor.unsqueeze(0), candidate_tensor.unsqueeze(0), dim=1
            ).item()
        )
        generated_equal = (
            reference.generated_token_ids[index] == candidate.generated_token_ids[index]
        )
        total_absolute_difference += float(difference.sum().item())
        total_elements += difference.numel()
        per_prompt.append(
            {
                "prompt_sha256": reference.prompt_hashes[index],
                "finite": finite,
                "exact_logits": bool(torch.equal(reference_tensor, candidate_tensor)),
                "max_abs_logit_difference": float(difference.max().item()),
                "mean_abs_logit_difference": float(difference.mean().item()),
                "logit_cosine_similarity": cosine,
                "next_token_equal": bool(
                    reference_tensor.argmax().item() == candidate_tensor.argmax().item()
                ),
                "generation_equal": generated_equal,
                "generated_token_count": len(candidate.generated_token_ids[index]),
            }
        )

    return {
        "prompt_count": len(per_prompt),
        "finite": all(item["finite"] for item in per_prompt),
        "exact_logits": all(item["exact_logits"] for item in per_prompt),
        "max_abs_logit_difference": max(
            (item["max_abs_logit_difference"] for item in per_prompt), default=0.0
        ),
        "mean_abs_logit_difference": (
            total_absolute_difference / total_elements if total_elements else 0.0
        ),
        "min_logit_cosine_similarity": min(
            (item["logit_cosine_similarity"] for item in per_prompt), default=1.0
        ),
        "next_token_equal": all(item["next_token_equal"] for item in per_prompt),
        "generation_equal": all(item["generation_equal"] for item in per_prompt),
        "per_prompt": per_prompt,
    }


def evaluate_verification(
    premerge_to_merged: dict[str, Any],
    merged_to_reloaded: dict[str, Any],
    reloaded_to_zero_loop_lora: dict[str, Any],
    *,
    min_merge_cosine: float,
    zero_lora_b_tensors: int,
    nonzero_lora_b_values: int,
) -> dict[str, Any]:
    merge_passed = bool(
        premerge_to_merged["finite"]
        and premerge_to_merged["next_token_equal"]
        and premerge_to_merged["min_logit_cosine_similarity"] >= min_merge_cosine
    )
    reload_passed = bool(
        merged_to_reloaded["finite"]
        and merged_to_reloaded["exact_logits"]
        and merged_to_reloaded["generation_equal"]
    )
    zero_loop_lora_passed = bool(
        zero_lora_b_tensors > 0
        and nonzero_lora_b_values == 0
        and reloaded_to_zero_loop_lora["finite"]
        and reloaded_to_zero_loop_lora["exact_logits"]
        and reloaded_to_zero_loop_lora["generation_equal"]
    )
    return {
        "passed": merge_passed and reload_passed and zero_loop_lora_passed,
        "thresholds": {
            "min_merge_cosine_similarity": min_merge_cosine,
            "merge_requires_all_next_tokens_equal": True,
            "merge_requires_full_greedy_generations_equal": False,
            "merge_generation_note": (
                "Full greedy continuations are diagnostic only because BF16 weight merging "
                "can change a later near-tied token after identical first-token argmaxes."
            ),
        },
        "checks": {
            "premerge_to_merged": {
                "passed": merge_passed,
                **premerge_to_merged,
            },
            "merged_to_reloaded": {
                "passed": reload_passed,
                **merged_to_reloaded,
            },
            "reloaded_to_zero_initialized_loop_lora": {
                "passed": zero_loop_lora_passed,
                "lora_b_tensor_count": zero_lora_b_tensors,
                "nonzero_lora_b_values": nonzero_lora_b_values,
                **reloaded_to_zero_loop_lora,
            },
        },
    }


def validate_paths(base: Path, adapter: Path, output: Path) -> None:
    required = (
        base / "config.json",
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ValueError(f"Required model files are missing: {formatted}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {output}")
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(base.resolve()) or resolved_output.is_relative_to(
        adapter.resolve()
    ):
        raise ValueError("Output directory cannot be the Base/adapter directory or a child of it")


def build_manifest(
    *,
    base: Path,
    adapter: Path,
    output: Path,
    dtype: str,
    max_shard_size: str,
    max_new_tokens: int,
    min_merge_cosine: float,
    adapter_config: dict[str, Any],
    source_adapter_sha256: str,
    output_weights: dict[str, Any],
    verification: dict[str, Any],
    package_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "status": "verified" if verification["passed"] else "verification_failed",
        "base_model": {
            "path": str(base.resolve()),
            "config_sha256": sha256_file(base / "config.json"),
        },
        "sft_adapter": {
            "path": str(adapter.resolve()),
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_model_sha256": source_adapter_sha256,
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "lora_dropout": adapter_config.get("lora_dropout"),
            "target_modules": sorted(adapter_config.get("target_modules", [])),
        },
        "merged_model": {
            "path": str(output.resolve()),
            "dtype": dtype,
            "safe_merge": True,
            "max_shard_size": max_shard_size,
            "tokenizer_saved_and_reloaded": True,
            "weights": output_weights,
        },
        "loop_adapter_initialization": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "target_modules": list(LOOP_LORA_TARGET_MODULES),
            "init_lora_weights": True,
        },
        "probe": {
            "names": [spec.name for spec in DEFAULT_PROBES],
            "prompt_sha256": [_prompt_hash(spec) for spec in DEFAULT_PROBES],
            "max_new_tokens": max_new_tokens,
            "min_merge_cosine_similarity": min_merge_cosine,
        },
        "verification": verification,
        "package_versions": package_versions,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "peft", "safetensors")
    }


def _dtype(name: str, torch_module: Any) -> Any:
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[name]


def _model_load_kwargs(dtype: Any, device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device != "cpu":
        kwargs["device_map"] = {"": device}
    return kwargs


def _run_probes(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> ProbeSnapshot:
    import torch

    model.eval()
    model_device = next(model.parameters()).device
    prompt_hashes: list[str] = []
    next_token_logits: list[Any] = []
    generated_token_ids: list[tuple[int, ...]] = []
    for spec in DEFAULT_PROBES:
        rendered = tokenizer.apply_chat_template(
            list(spec.messages), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].detach().float().cpu().squeeze(0)
            generated = model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist()
        prompt_hashes.append(_prompt_hash(spec))
        next_token_logits.append(logits)
        generated_token_ids.append(tuple(int(token) for token in new_tokens))

    return ProbeSnapshot(
        prompt_hashes=tuple(prompt_hashes),
        next_token_logits=tuple(next_token_logits),
        generated_token_ids=tuple(generated_token_ids),
    )


def _cleanup_model_memory(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_args(args)
    base = args.base.expanduser()
    adapter = args.adapter.expanduser()
    output = args.output.expanduser()
    validate_paths(base, adapter, output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")

    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dtype = _dtype(args.dtype, torch)
    load_kwargs = _model_load_kwargs(model_dtype, args.device)
    tokenizer = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(base, **load_kwargs)
    sft_model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
    premerge_probe = _run_probes(sft_model, tokenizer, max_new_tokens=args.probe_max_new_tokens)

    merged_model = sft_model.merge_and_unload(safe_merge=args.safe_merge)
    merged_model.eval()
    merged_probe = _run_probes(merged_model, tokenizer, max_new_tokens=args.probe_max_new_tokens)
    premerge_to_merged = compare_probe_snapshots(premerge_probe, merged_probe)

    merged_model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output)
    del sft_model, merged_model, base_model, tokenizer
    _cleanup_model_memory(torch)

    reloaded_tokenizer = AutoTokenizer.from_pretrained(output)
    reloaded_model = AutoModelForCausalLM.from_pretrained(output, **load_kwargs)
    reloaded_probe = _run_probes(
        reloaded_model, reloaded_tokenizer, max_new_tokens=args.probe_max_new_tokens
    )
    merged_to_reloaded = compare_probe_snapshots(merged_probe, reloaded_probe)

    loop_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(LOOP_LORA_TARGET_MODULES),
        init_lora_weights=True,
    )
    loop_model = get_peft_model(reloaded_model, loop_lora_config)
    lora_b_parameters = [
        parameter for name, parameter in loop_model.named_parameters() if ".lora_B." in name
    ]
    nonzero_lora_b_values = sum(
        int(torch.count_nonzero(parameter.detach()).item()) for parameter in lora_b_parameters
    )
    zero_loop_probe = _run_probes(
        loop_model, reloaded_tokenizer, max_new_tokens=args.probe_max_new_tokens
    )
    reloaded_to_zero_loop_lora = compare_probe_snapshots(reloaded_probe, zero_loop_probe)

    verification = evaluate_verification(
        premerge_to_merged,
        merged_to_reloaded,
        reloaded_to_zero_loop_lora,
        min_merge_cosine=args.min_merge_cosine,
        zero_lora_b_tensors=len(lora_b_parameters),
        nonzero_lora_b_values=nonzero_lora_b_values,
    )
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    manifest = build_manifest(
        base=base,
        adapter=adapter,
        output=output,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        max_new_tokens=args.probe_max_new_tokens,
        min_merge_cosine=args.min_merge_cosine,
        adapter_config=adapter_config,
        source_adapter_sha256=sha256_file(adapter / "adapter_model.safetensors"),
        output_weights=hash_weight_files(output),
        verification=verification,
        package_versions=_package_versions(),
    )
    write_manifest(output / MANIFEST_NAME, manifest)
    del loop_model, reloaded_model, reloaded_tokenizer
    _cleanup_model_memory(torch)

    if not verification["passed"]:
        raise RuntimeError(
            f"SFT merge verification failed; inspect {output / MANIFEST_NAME} before training"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely merge an SFT LoRA and verify a fresh rank-16 LOOP initialization."
    )
    parser.add_argument("--base", type=Path, required=True, help="Local Base model directory")
    parser.add_argument("--adapter", type=Path, required=True, help="SFT LoRA adapter directory")
    parser.add_argument("--output", type=Path, required=True, help="New merged model directory")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--safe-merge",
        action="store_true",
        default=True,
        help="Enable PEFT NaN-checked safe merge (mandatory and enabled by default)",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--probe-max-new-tokens", type=int, default=16)
    parser.add_argument("--min-merge-cosine", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.safe_merge:
        raise ValueError("Safe merge is mandatory for SFT→LOOP initialization")
    if args.probe_max_new_tokens < 1:
        raise ValueError("--probe-max-new-tokens must be positive")
    if not math.isfinite(args.min_merge_cosine) or not 0.0 <= args.min_merge_cosine <= 1.0:
        raise ValueError("--min-merge-cosine must be finite and between 0 and 1")
    if not args.max_shard_size.strip():
        raise ValueError("--max-shard-size cannot be empty")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    manifest = prepare(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(args.output.expanduser() / MANIFEST_NAME),
                "combined_weight_sha256": manifest["merged_model"]["weights"]["combined_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
