#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from phi_agents.rl.type_defs import Message, PolicyMessage, PolicyTokenInfo


@dataclass
class SpecialToken:
    id: int  # like 128000
    content: str  # like <|begin_of_text>|


class LLM(ABC):
    @abstractmethod
    def get_tokens(
        self, messages: list[Message], is_output: bool = False, log_probs: bool = False
    ) -> tuple[list[int], list[bool] | None, list[float] | None]:
        """Get tokenization of the chat, keeping generated tokens for PolicyMessages.

        Returns:
            Tuple of (tokens, is_output, log_probs).
        """
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizerBase:
        pass

    @property
    @abstractmethod
    def special_tokens(self) -> dict[str, SpecialToken]:
        """Mapping from arbitrary str label to SpecialToken.

        Each SpecialToken should only occur once in the dict (there are no checks for this)
        """
        pass

    @abstractmethod
    def generate(self, messages: list[Message], *, seed: int | None = None) -> PolicyMessage:
        pass


class TrainableLLM(LLM):
    @abstractmethod
    def get_policy_token_info(self, messages: list[Message]) -> PolicyTokenInfo:
        pass


class SeededTrainableLLM(TrainableLLM):
    """Per-rollout deterministic view over a shared trainable LLM client.

    AppWorld runners share one HTTP client per vLLM server.  Keeping the seed and
    turn counter in this lightweight view avoids mutable cross-thread state while
    ensuring that the same rollout seed produces the same request-seed sequence
    for every compared model state.
    """

    def __init__(self, llm: TrainableLLM, rollout_seed: int):
        self._llm = llm
        self._rollout_seed = int(rollout_seed)
        self._generation_index = 0

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._llm.tokenizer

    @property
    def special_tokens(self) -> dict[str, SpecialToken]:
        return self._llm.special_tokens

    def get_tokens(
        self, messages: list[Message], is_output: bool = False, log_probs: bool = False
    ) -> tuple[list[int], list[bool] | None, list[float] | None]:
        return self._llm.get_tokens(messages, is_output=is_output, log_probs=log_probs)

    def generate(self, messages: list[Message], *, seed: int | None = None) -> PolicyMessage:
        if seed is None:
            seed = (self._rollout_seed + self._generation_index) % (2**32)
        self._generation_index += 1
        return self._llm.generate(messages, seed=seed)

    def get_policy_token_info(self, messages: list[Message]) -> PolicyTokenInfo:
        return self._llm.get_policy_token_info(messages)


def messages_str(messages: list[Message]) -> str:
    output_str = ""
    for m in messages:
        output_str += f"--------------{m.role} n_characters {len(m.content)} ---------------\n"
        output_str += m.content + "\n"

    return output_str
