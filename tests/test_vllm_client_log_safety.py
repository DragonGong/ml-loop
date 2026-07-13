import io
import logging
from typing import Any

import pytest
import requests

import phi_agents.rl.vllm_client as vllm_client_module
from phi_agents.rl.vllm_client import MaxSeqLenExceeded, VLLMClient


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        text: str,
        json_data: dict[str, Any] | None = None,
        chunks: list[str | bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}
        self._chunks = chunks or []

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"unsafe upstream error: {self.text}")

    def json(self) -> dict[str, Any]:
        return self._json_data

    def iter_content(self, **kwargs: Any) -> list[str | bytes]:
        del kwargs
        return self._chunks


def _capture_module_logger(monkeypatch) -> io.StringIO:
    stream = io.StringIO()
    logger = logging.Logger("vllm-client-safety-test", level=logging.DEBUG)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    monkeypatch.setattr(vllm_client_module, "logger", logger)
    return stream


def test_http_error_omits_response_and_prompt(monkeypatch, capsys) -> None:
    secret = "sk-live-http-error-secret"
    prompt_tokens = [901234567, 876543210]
    response = FakeResponse(
        status_code=500,
        text=f"upstream payload contains {secret}",
        json_data={"message": f"payload contains {secret}"},
    )
    monkeypatch.setattr(vllm_client_module.requests, "post", lambda *args, **kwargs: response)
    log_stream = _capture_module_logger(monkeypatch)

    with pytest.raises(requests.HTTPError) as exc_info:
        VLLMClient().get_completion("model", prompt_tokens)

    captured = capsys.readouterr()
    combined_output = log_stream.getvalue() + captured.out + captured.err
    assert secret not in combined_output
    assert str(prompt_tokens) not in combined_output
    assert secret not in str(exc_info.value)
    assert str(prompt_tokens) not in str(exc_info.value)
    assert "status_code=500" in combined_output


def test_invalid_sse_error_omits_chunk_and_prompt(monkeypatch, capsys) -> None:
    secret_chunk = "invalid SSE sk-live-sse-error-secret"
    prompt_tokens = [112233445, 998877665]
    response = FakeResponse(status_code=200, text="", chunks=[secret_chunk])
    monkeypatch.setattr(vllm_client_module.requests, "post", lambda *args, **kwargs: response)
    log_stream = _capture_module_logger(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        VLLMClient().get_completion("model", prompt_tokens)

    captured = capsys.readouterr()
    combined_output = log_stream.getvalue() + captured.out + captured.err
    assert secret_chunk not in combined_output
    assert str(prompt_tokens) not in combined_output
    assert secret_chunk not in str(exc_info.value)
    assert str(prompt_tokens) not in str(exc_info.value)
    assert "chunk_chars=" in combined_output


def test_max_sequence_error_detection_does_not_expose_payload(monkeypatch, capsys) -> None:
    secret = "sk-live-context-error-secret"
    response = FakeResponse(
        status_code=400,
        text=f"context response contains {secret}",
        json_data={
            "message": f"This model's maximum context length is 4096 tokens; {secret}"
        },
    )
    monkeypatch.setattr(vllm_client_module.requests, "post", lambda *args, **kwargs: response)
    log_stream = _capture_module_logger(monkeypatch)

    with pytest.raises(MaxSeqLenExceeded) as exc_info:
        VLLMClient().get_completion("model", [123, 456])

    captured = capsys.readouterr()
    combined_output = log_stream.getvalue() + captured.out + captured.err
    assert secret not in combined_output
    assert secret not in str(exc_info.value)
