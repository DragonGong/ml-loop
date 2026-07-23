from __future__ import annotations

from collections import UserDict
from typing import Any

from scripts.sft.audit_qwen35_turn_boundaries import audit_rows, supervised_spans
from scripts.sft.diagnose_qwen35_turn_boundaries import (
    aggregate_state,
    classify_result,
    exact_prompt_token_ids,
    summarize_generation,
)


class _PromptTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> UserDict[str, list[int]]:
        assert tokenize is True
        assert enable_thinking is False
        ids = [
            value
            for message in messages
            for value in (len(message["role"]), len(message["content"]))
        ]
        return UserDict({"input_ids": ids + ([91, 92] if add_generation_prompt else [])})


class _BoundaryTokenizer:
    _ids = {
        "<|im_start|>": 10_001,
        "<|im_end|>": 10_002,
        "<|endoftext|>": 10_003,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        inverse = {value: key for key, value in self._ids.items()}
        return "".join(inverse.get(token_id, chr(token_id)) for token_id in token_ids)


def test_exact_prompt_token_ids_matches_qwen_wrapper_suffix_logic() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    assert exact_prompt_token_ids(_PromptTokenizer(), messages) == [6, 3, 4, 4, 91, 92]


def test_boundary_audit_detects_masked_im_end_after_each_target() -> None:
    tokenizer = _BoundaryTokenizer()
    first = "```python\nprint(1)\n```"
    second = "```python\nprint(2)\n```"
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    observation = "\nOBS\n"
    input_ids = (
        [ord(character) for character in first]
        + [im_end]
        + [ord(character) for character in observation]
        + [ord(character) for character in second]
        + [im_end]
    )
    labels = (
        [ord(character) for character in first]
        + [-100] * (1 + len(observation))
        + [ord(character) for character in second]
        + [-100]
    )
    rows: list[dict[str, Any]] = [
        {
            "input_ids": input_ids,
            "labels": labels,
            "messages": [
                {"role": "assistant", "content": first, "loss": True},
                {
                    "role": "user",
                    "content": observation,
                    "loss": False,
                    "message_type": "appworld_observation",
                },
                {"role": "assistant", "content": second, "loss": True},
            ],
        }
    ]

    assert supervised_spans(labels) == [
        (0, len(first)),
        (len(first) + 1 + len(observation), len(input_ids) - 1),
    ]
    result = audit_rows(rows, tokenizer)
    assert result["supervised_spans"] == 2
    assert result["spans_followed_by_im_end"] == 2
    assert result["spans_followed_by_masked_im_end"] == 2
    assert result["<|im_end|>_supervised_count"] == 0
    assert result["supervised_assistant_messages_with_multiple_python_blocks"] == 0
    assert result["target_pairs_without_intervening_observation"] == 0


def test_first_turn_classifier_separates_base_from_broken_adapters() -> None:
    stop_tokens = {10_002}
    rows = []
    for task_id in ("a", "b", "c"):
        rows.append(
            summarize_generation(
                state="base",
                task_id=task_id,
                text="```python\nprint(1)\n```",
                token_ids=[1, 2],
                finish_reason="stop",
                stop_reason=10_002,
                stop_token_ids=stop_tokens,
            )
        )
        for state in ("d12", "d3"):
            rows.append(
                summarize_generation(
                    state=state,
                    task_id=task_id,
                    text="```python\nprint(1)\n```\n```python\nprint(2)\n```",
                    token_ids=list(range(1_200)),
                    finish_reason="length",
                    stop_reason=None,
                    stop_token_ids=stop_tokens,
                )
            )
    aggregates = {
        state: aggregate_state([row for row in rows if row["state"] == state])
        for state in ("base", "d12", "d3")
    }
    assert classify_result(aggregates) == "adapter_specific_turn_boundary_failure"
