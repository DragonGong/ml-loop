from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = parser.parse_args()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.adapter_path / filename).is_file():
            raise FileNotFoundError(args.adapter_path / filename)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are an AppWorld agent."},
            {
                "role": "user",
                "content": "Reply with one short Python code block that prints the number 7.",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    llm = LLM(
        model=str(args.model_path),
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=16_384,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        language_model_only=True,
        seed=20260722,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.1, max_tokens=64, seed=2026072200),
        lora_request=LoRARequest("qwen35_sft_smoke", 1, str(args.adapter_path)),
    )
    generated = outputs[0].outputs[0].text
    if not generated.strip():
        raise RuntimeError("Qwen3.5 LoRA smoke generation was empty")
    result = {
        "status": "passed",
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "prompt_tokens": len(outputs[0].prompt_token_ids),
        "generated_tokens": len(outputs[0].outputs[0].token_ids),
        "generated_text": generated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
