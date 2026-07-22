from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from phi_agents.sft.trainer import (
    AssistantOnlyCollator,
    QWEN35_LORA_TARGET_MODULES,
    SFTConfig,
    TokenizedSFTDataset,
    model_profile,
    sparse_assistant_causal_loss,
    sparse_assistant_trainer_class,
    tokenize_messages,
    training_schedule_preflight,
)


class _TrimThinkingTokenizer:
    chat_template = "{% if enable_thinking %}thinking{% endif %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        assert enable_thinking is False
        rendered: list[str] = []
        for message in messages:
            content = message["content"].strip()
            if message["role"] == "assistant":
                content = "<think>\n\n</think>\n\n" + content
            rendered.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        return "".join(rendered)

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, Any]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_qwen35_trimmed_template_supervises_only_action_content() -> None:
    tokenizer = _TrimThinkingTokenizer()
    action = "Code:\n```python\nprint('ok')\n```"
    result = tokenize_messages(
        tokenizer,
        [
            {"role": "system", "content": "  SYS  ", "loss": False},
            {"role": "user", "content": "  TASK  ", "loss": False},
            {
                "role": "assistant",
                "content": f"  {action}  ",
                "loss": True,
                "step_id": "task:trajectory:0",
            },
            {
                "role": "user",
                "content": "OBSERVATION",
                "loss": False,
                "step_id": "task:trajectory:0",
            },
            {
                "role": "assistant",
                "content": "   ",
                "loss": False,
                "step_id": "task:trajectory:1",
            },
        ],
    )

    supervised_text = "".join(
        chr(token_id)
        for token_id, label in zip(result["input_ids"], result["labels"], strict=True)
        if label != -100
    )
    assert supervised_text == action
    assert "think" not in supervised_text
    assert "im_start" not in supervised_text
    assert "TASK" not in supervised_text
    assert result["step_token_stats"][0]["supervised_tokens"] == len(action)
    assert result["step_token_stats"][1]["supervised_tokens"] == 0


class _SparseLogitsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(23, 7)
        self.projection = torch.nn.Linear(7, 23, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: torch.Tensor | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        assert use_cache is False
        hidden = self.embedding(input_ids)
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep, :]
        return SimpleNamespace(logits=self.projection(hidden))


def test_sparse_assistant_loss_and_gradients_match_full_causal_loss() -> None:
    torch.manual_seed(20260722)
    sparse_model = _SparseLogitsModel()
    full_model = _SparseLogitsModel()
    full_model.load_state_dict(sparse_model.state_dict())
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 3, 4, -100, 6, 7]]),
    }
    target_count = int(inputs["labels"][:, 1:].ne(-100).sum())

    sparse_loss, outputs = sparse_assistant_causal_loss(
        sparse_model,
        inputs,
        num_items_in_batch=target_count,
    )
    sparse_loss.backward()

    full_logits = full_model(inputs["input_ids"]).logits
    full_loss = torch.nn.functional.cross_entropy(
        full_logits[:, :-1, :].reshape(-1, full_logits.shape[-1]).float(),
        inputs["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / target_count
    full_loss.backward()

    assert outputs.logits.shape == (1, target_count, 23)
    torch.testing.assert_close(sparse_loss, full_loss)
    for sparse_parameter, full_parameter in zip(
        sparse_model.parameters(), full_model.parameters(), strict=True
    ):
        torch.testing.assert_close(sparse_parameter.grad, full_parameter.grad)


def test_sparse_trainer_accumulation_matches_token_weighted_batch(tmp_path) -> None:
    from transformers import TrainingArguments

    torch.manual_seed(20260722)
    accumulated_model = _SparseLogitsModel()
    reference_model = _SparseLogitsModel()
    reference_model.load_state_dict(accumulated_model.state_dict())
    rows = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1] * 5,
            "labels": [-100, -100, 3, 4, -100],
        },
        {
            "input_ids": [6, 7, 8, 9, 10, 11, 12],
            "attention_mask": [1] * 7,
            "labels": [-100, -100, 8, 9, 10, 11, 12],
        },
    ]
    collator = AssistantOnlyCollator(pad_token_id=0)
    reference_batch = collator([TokenizedSFTDataset(rows)[index] for index in range(2)])
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    logits = reference_model(reference_batch["input_ids"]).logits
    reference_loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]).float(),
        reference_batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    ) / reference_batch["labels"][:, 1:].ne(-100).sum()
    reference_loss.backward()
    reference_optimizer.step()

    trainer_class = sparse_assistant_trainer_class()
    trainer = trainer_class(
        model=accumulated_model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=0.05,
            lr_scheduler_type="constant",
            optim="sgd",
            max_grad_norm=0.0,
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            average_tokens_across_devices=True,
            use_cpu=True,
            seed=17,
            data_seed=17,
        ),
        train_dataset=TokenizedSFTDataset(rows),
        data_collator=collator,
    )
    trainer.train()

    for accumulated_parameter, reference_parameter in zip(
        accumulated_model.parameters(), reference_model.parameters(), strict=True
    ):
        torch.testing.assert_close(accumulated_parameter, reference_parameter)


def test_qwen35_profile_has_all_hybrid_lora_targets(monkeypatch) -> None:
    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda _source: SimpleNamespace(model_type="qwen3_5"),
    )
    profile = model_profile("Qwen/Qwen3.5-4B")

    assert list(profile.lora_target_modules) == QWEN35_LORA_TARGET_MODULES
    assert profile.image_text_model
    assert profile.sparse_assistant_logits_supported
    assert profile.fsdp_transformer_layer_cls == "Qwen3_5DecoderLayer"


def test_two_gpu_schedule_counts_global_optimizer_steps(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    d12 = training_schedule_preflight(
        SFTConfig(
            train_jsonl=tmp_path / "train.jsonl",
            validation_jsonl=tmp_path / "validation.jsonl",
            output_dir=tmp_path / "d12",
            epochs=1,
            gradient_accumulation_steps=4,
        ),
        509,
    )
    d3 = training_schedule_preflight(
        SFTConfig(
            train_jsonl=tmp_path / "train.jsonl",
            validation_jsonl=tmp_path / "validation.jsonl",
            output_dir=tmp_path / "d3",
            epochs=2,
            gradient_accumulation_steps=2,
        ),
        81,
    )

    assert d12["world_size"] == 2
    assert d12["global_batch_size"] == 8
    assert d12["optimizer_steps_per_epoch"] == 64
    assert d12["expected_optimizer_steps"] == 64
    assert d3["global_batch_size"] == 4
    assert d3["optimizer_steps_per_epoch"] == 21
    assert d3["expected_optimizer_steps"] == 42
