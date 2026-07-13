from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any

import pytest
import torch

from phi_agents.sft.trainer import (
    AssistantOnlyCollator,
    SFTConfig,
    TokenizedSFTDataset,
    train,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _force_cpu_without_cuda_driver(monkeypatch: Any) -> Iterator[None]:
    """Keep this regression test independent of workstation CUDA driver health."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    from phi_agents.utils.logger import get_phi_logger

    logger = get_phi_logger()
    production_handlers = list(logger.handlers)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    yield
    logger.handlers.clear()
    logger.handlers.extend(production_handlers)


def _gradient_capture() -> Any:
    from transformers import TrainerCallback

    class GradientCapture(TrainerCallback):
        def __init__(self) -> None:
            self.gradients: dict[str, torch.Tensor] = {}

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            model = kwargs["model"]
            self.gradients = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            }

    return GradientCapture()


class _TinyCausalLM(torch.nn.Module):
    accepts_loss_kwargs = True

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 12)
        self.projection = torch.nn.Linear(12, 32, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del attention_mask, kwargs
        logits = self.projection(self.embedding(input_ids))
        shifted_logits = logits[:, :-1, :].contiguous().float()
        shifted_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="sum" if num_items_in_batch is not None else "mean",
        )
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch.to(loss.device)
        return {"loss": loss, "logits": logits}


def _tiny_model(initial_state: dict[str, torch.Tensor] | None = None) -> torch.nn.Module:
    model = _TinyCausalLM()
    if initial_state is not None:
        model.load_state_dict(initial_state)
    return model


def _rows() -> list[dict[str, list[int]]]:
    return [
        {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "attention_mask": [1] * 6,
            "labels": [-100, -100, -100, 4, 5, -100],
        },
        {
            "input_ids": [1, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "attention_mask": [1] * 10,
            "labels": [-100, -100, 8, 9, 10, 11, 12, 13, 14, 15],
        },
    ]


def test_reference_token_weighted_accumulation_matches_single_batch() -> None:
    torch.random.default_generator.manual_seed(1234)
    initial = copy.deepcopy(_tiny_model().state_dict())
    collator = AssistantOnlyCollator(pad_token_id=0)
    features = [TokenizedSFTDataset(_rows())[index] for index in range(2)]
    total_items = sum(int(feature["labels"].ne(-100).sum()) for feature in features)

    batch_model = _tiny_model(initial)
    batch_optimizer = torch.optim.SGD(batch_model.parameters(), lr=0.05)
    batch = collator(features)
    batch_loss = batch_model(**batch, num_items_in_batch=torch.tensor(total_items))["loss"]
    batch_loss.backward()
    batch_gradients = {
        name: parameter.grad.detach().clone() for name, parameter in batch_model.named_parameters()
    }
    batch_optimizer.step()

    accumulated_model = _tiny_model(initial)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.05)
    accumulated_loss = torch.tensor(0.0)
    for feature in features:
        micro_batch = collator([feature])
        micro_loss = accumulated_model(**micro_batch, num_items_in_batch=torch.tensor(total_items))[
            "loss"
        ]
        accumulated_loss = accumulated_loss + micro_loss.detach()
        micro_loss.backward()
    accumulated_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in accumulated_model.named_parameters()
    }
    accumulated_optimizer.step()

    loss_error = float((batch_loss.detach() - accumulated_loss).abs())
    gradient_error = max(
        float((batch_gradients[name] - accumulated_gradients[name]).abs().max())
        for name in batch_gradients
    )
    parameter_error = max(
        float((batch_model.state_dict()[name] - accumulated_model.state_dict()[name]).abs().max())
        for name in batch_model.state_dict()
    )
    print(
        json.dumps(
            {
                "reference_loss_abs_error": loss_error,
                "reference_gradient_max_abs_error": gradient_error,
                "reference_parameter_max_abs_error": parameter_error,
            },
            sort_keys=True,
        )
    )
    assert loss_error < 1e-6
    assert gradient_error < 1e-6
    assert parameter_error < 1e-6


def _one_update(
    tmp_path: Path,
    initial_state: dict[str, torch.Tensor],
    *,
    batch_size: int,
    accumulation: int,
) -> tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    from transformers import Trainer, TrainingArguments

    model = _tiny_model(initial_state)
    capture = _gradient_capture()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            learning_rate=0.05,
            lr_scheduler_type="constant",
            optim="sgd",
            max_grad_norm=0.0,
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=17,
            data_seed=17,
            use_cpu=True,
        ),
        train_dataset=TokenizedSFTDataset(_rows()),
        data_collator=AssistantOnlyCollator(pad_token_id=0),
        callbacks=[capture],
    )
    result = trainer.train()
    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return float(result.training_loss), capture.gradients, parameters


def test_trainer_gradient_accumulation_is_supervised_token_normalized(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    initial_model = _tiny_model()
    initial_state = copy.deepcopy(initial_model.state_dict())
    batch_loss, batch_gradients, batch_parameters = _one_update(
        tmp_path / "batch", initial_state, batch_size=2, accumulation=1
    )
    accumulated_loss, accumulated_gradients, accumulated_parameters = _one_update(
        tmp_path / "accumulated", initial_state, batch_size=1, accumulation=2
    )

    assert batch_gradients.keys() == accumulated_gradients.keys()
    assert batch_parameters.keys() == accumulated_parameters.keys()
    gradient_error = max(
        float((batch_gradients[name] - accumulated_gradients[name]).abs().max())
        for name in batch_gradients
    )
    parameter_error = max(
        float((batch_parameters[name] - accumulated_parameters[name]).abs().max())
        for name in batch_parameters
    )
    loss_error = abs(batch_loss - accumulated_loss)
    print(
        json.dumps(
            {
                "batch_loss": batch_loss,
                "accumulated_loss": accumulated_loss,
                "loss_abs_error": loss_error,
                "gradient_max_abs_error": gradient_error,
                "parameter_max_abs_error": parameter_error,
            },
            sort_keys=True,
        )
    )
    assert loss_error < 1e-6
    assert gradient_error < 1e-6
    assert parameter_error < 1e-6


class _TinyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize and not add_generation_prompt
        return "".join(f"<{row['role']}>{row['content']}</{row['role']}>" for row in messages)

    def __call__(
        self, text: str, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, Any]:
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [(ord(character) % 29) + 1 for character in text],
            "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
        }


def _sft_sample(index: int) -> dict[str, Any]:
    step_id = f"task-{index}:trajectory-{index}:0"
    return {
        "messages": [
            {"role": "system", "content": "SYS", "loss": False},
            {"role": "user", "content": f"TASK-{index}", "loss": False},
            {
                "role": "assistant",
                "content": f"ACTION-{index}",
                "loss": True,
                "message_type": "appworld_action",
                "step_id": step_id,
                "step_index": 0,
                "mask_reason": None,
            },
            {
                "role": "user",
                "content": f"OBS-{index}",
                "loss": False,
                "message_type": "appworld_observation",
                "step_id": step_id,
            },
        ],
        "metadata": {"task_id": f"task-{index}", "trajectory_id": f"trajectory-{index}"},
    }


def test_tiny_sft_train_backward_and_checkpoint_smoke(tmp_path: Path, monkeypatch: Any) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = _TinyChatTokenizer()
    tiny_base = tmp_path / "tiny-base"
    tiny_base.mkdir()

    def tiny_model() -> Qwen2ForCausalLM:
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        config._name_or_path = str(tiny_base)
        config.to_json_file(tiny_base / "config.json")
        return Qwen2ForCausalLM(config)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: tiny_model(),
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("\n".join(json.dumps(_sft_sample(index)) for index in range(2)) + "\n")
    validation_path.write_text(json.dumps(_sft_sample(2)) + "\n")
    output_dir = tmp_path / "output"
    result = train(
        SFTConfig(
            train_jsonl=train_path,
            validation_jsonl=validation_path,
            output_dir=output_dir,
            model_name="tiny-qwen2",
            max_length=256,
            lora_rank=2,
            lora_alpha=4,
            learning_rate=1e-3,
            epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False,
            bf16=False,
            logging_steps=1,
            save_steps=1,
            eval_steps=1,
            use_cpu=True,
        )
    )
    assert result["metrics"]["train_loss"] > 0
    assert result["metrics"]["effective_supervised_tokens"] > 0
    checkpoint = output_dir / "checkpoint-1"
    assert (checkpoint / "sft_artifact.json").is_file()
    assert (checkpoint / "trainer_state.json").is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "rng_state.pth").is_file()
    assert (checkpoint / "lora" / "adapter_config.json").is_file()
    assert (output_dir / "final_adapter" / "adapter_config.json").is_file()
    assert (output_dir / "lora" / "adapter_config.json").is_file()
    assert (output_dir / "train_metrics.json").is_file()
