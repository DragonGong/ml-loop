from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ABLATIONS = {
    "without_linear_attention": lambda key: ".linear_attn." in key,
    "linear_attention_only": lambda key: ".linear_attn." not in key,
    "mlp_only": lambda key: ".mlp." not in key,
    "attention_only": lambda key: ".mlp." in key,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_safetensors(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def _build_variant(
    *,
    source_dir: Path,
    output_dir: Path,
    name: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    should_zero: Any,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=False)
    for filename in ("adapter_config.json", "README.md"):
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, variant_dir / filename)

    variant: dict[str, torch.Tensor] = {}
    zeroed: list[str] = []
    for key, tensor in tensors.items():
        if key.endswith(".lora_B.weight") and should_zero(key):
            variant[key] = torch.zeros_like(tensor)
            zeroed.append(key)
        else:
            variant[key] = tensor
    if not zeroed:
        raise ValueError(f"Ablation {name} did not select any LoRA B tensors")
    target = variant_dir / "adapter_model.safetensors"
    save_file(variant, target, metadata=metadata)
    return {
        "name": name,
        "path": str(variant_dir.resolve()),
        "zeroed_lora_b_tensors": len(zeroed),
        "retained_lora_b_tensors": sum(
            key.endswith(".lora_B.weight") for key in tensors
        )
        - len(zeroed),
        "zeroed_keys": zeroed,
        "adapter_sha256": _sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(ABLATIONS))
    args = parser.parse_args()

    source_model = args.source_adapter / "adapter_model.safetensors"
    source_config = args.source_adapter / "adapter_config.json"
    for path in (source_model, source_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}")

    tensors, metadata = _load_safetensors(source_model)
    variants = [
        _build_variant(
            source_dir=args.source_adapter,
            output_dir=args.output_dir,
            name=name,
            tensors=tensors,
            metadata=metadata,
            should_zero=ABLATIONS[name],
        )
        for name in requested
    ]
    manifest = {
        "source_adapter": str(args.source_adapter.resolve()),
        "source_adapter_sha256": _sha256(source_model),
        "source_tensor_count": len(tensors),
        "variants": variants,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
