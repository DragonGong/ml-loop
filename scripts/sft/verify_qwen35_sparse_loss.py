from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phi_agents.sft.trainer import tokenize_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="sdpa")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    row = tokenize_messages(
        tokenizer,
        [
            {"role": "system", "content": "You are an AppWorld assistant.", "loss": False},
            {
                "role": "user",
                "content": "Use the API documentation to inspect the phone app.",
                "loss": False,
            },
            {
                "role": "assistant",
                "content": (
                    "Code:\n```python\n"
                    "print(apis.api_docs.show_api_descriptions(app_name='phone'))\n```"
                ),
                "loss": True,
                "step_id": "sparse-loss-check:0",
            },
        ],
    )
    input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda")
    prediction_positions = labels[:, 1:].ne(-100)[0].nonzero().flatten()
    targets = labels[:, 1:][0, prediction_positions]

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attention_backend,
    ).eval().cuda()
    with torch.inference_mode():
        full_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        ).logits[:, prediction_positions, :]
        sparse_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=prediction_positions,
        ).logits
    full_loss = torch.nn.functional.cross_entropy(
        full_logits.reshape(-1, full_logits.shape[-1]).float(),
        targets,
        reduction="mean",
    )
    sparse_loss = torch.nn.functional.cross_entropy(
        sparse_logits.reshape(-1, sparse_logits.shape[-1]).float(),
        targets,
        reduction="mean",
    )
    result = {
        "model_path": str(args.model_path.resolve()),
        "sequence_tokens": input_ids.shape[1],
        "supervised_targets": prediction_positions.numel(),
        "full_logits_shape": list(full_logits.shape),
        "sparse_logits_shape": list(sparse_logits.shape),
        "max_absolute_logit_difference": float(
            (full_logits.float() - sparse_logits.float()).abs().max().item()
        ),
        "full_loss": float(full_loss.item()),
        "sparse_loss": float(sparse_loss.item()),
        "absolute_loss_difference": float((full_loss - sparse_loss).abs().item()),
    }
    if result["max_absolute_logit_difference"] != 0:
        raise RuntimeError(f"Sparse logits differ from full logits: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
