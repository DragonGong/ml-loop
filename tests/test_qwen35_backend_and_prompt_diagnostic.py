from __future__ import annotations

from typing import Any

from scripts.sft.diagnose_qwen35_backend_and_prompt import (
    prompt_token_ids,
    training_no_thinking_prompt_token_ids,
)


class _TemplateTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        assert enable_thinking is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            + (
                "<think>\n\n</think>\n\n"
                if message["role"] == "assistant"
                and message is messages[-2]
                and messages[-1]["role"] == "user"
                and messages[-1]["content"].startswith("__APPWORLD_QWEN35_FOLLOWUP")
                is False
                else ""
            )
            + f"{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return [ord(character) for character in rendered] if tokenize else rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


def test_training_prompt_matches_action_context_without_empty_think() -> None:
    tokenizer = _TemplateTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]

    token_ids = training_no_thinking_prompt_token_ids(tokenizer, messages)
    rendered = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    assert rendered.endswith("<|im_start|>assistant\n")
    assert "<think>" not in rendered


def test_eval_and_training_prompt_styles_differ_only_by_empty_think_scaffold() -> None:
    tokenizer = _TemplateTokenizer()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]

    training_ids = prompt_token_ids(tokenizer, messages, "training_no_thinking_scaffold")
    eval_ids = prompt_token_ids(tokenizer, messages, "eval_disabled_thinking")

    assert eval_ids[: len(training_ids)] == training_ids
    assert tokenizer.decode(
        eval_ids[len(training_ids) :],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == "<think>\n\n</think>\n\n"
