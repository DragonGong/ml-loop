from argparse import Namespace
from pathlib import Path

from phi_agents.rl.config import get_config
from phi_agents.rl.llm import qwen_3
from phi_agents.vllm import vllm_server
from phi_agents.utils.file_utils import lora_path
from scripts.loop7b.eval_watch import _experiment_name, _is_adapter_directory


class _FakeTokenizer:
    unk_token_id = 0

    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []
        self.message_roles: list[list[str]] = []
        self._tokens = {
            "<|endoftext|>": 248044,
            "<|im_start|>": 248045,
            "<|im_end|>": 248046,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens.get(token, self.unk_token_id)

    def decode(self, token_ids: list[int]) -> str:
        inverse = {value: key for key, value in self._tokens.items()}
        return "".join(inverse[token_id] for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> dict[str, list[int]]:
        assert messages
        assert tokenize is True
        self.enable_thinking_values.append(enable_thinking)
        self.message_roles.append([message["role"] for message in messages])
        tokens = [41, 42]
        if add_generation_prompt:
            tokens.extend([self._tokens["<|im_start|>"], 99])
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

    def encode(self, text: str) -> list[int]:
        assert text == "\n"
        return [198]


def test_qwen35_resolves_special_tokens_and_disables_thinking(monkeypatch) -> None:
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        qwen_3.AutoTokenizer,
        "from_pretrained",
        lambda _path: tokenizer,
    )

    llm = qwen_3.VLLMQwen3(
        host="localhost",
        port=8000,
        base_model_path=Path("Qwen3.5-4B"),
        model_id=None,
        temperature=0.1,
        max_new_tokens=1200,
        top_p=None,
        min_p=None,
        top_k=None,
        frequency_penalty=None,
        max_model_len=16384,
        enable_thinking=False,
    )

    assert {name: token.id for name, token in llm.special_tokens.items()} == {
        "bom": 248045,
        "eom": 248046,
        "eot": 248044,
    }
    assert llm.generation_prompt_tokens == [248045, 99]
    assert tokenizer.enable_thinking_values == [False, False]
    assert tokenizer.message_roles == [
        ["system", "user"],
        ["system", "user"],
    ]


def test_vllm_server_passes_qwen35_text_only_and_cache_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _PopenResult:
        pid = 12345

    class _Process:
        pass

    def _popen(args, *, env):
        captured["args"] = args
        captured["env"] = env
        return _PopenResult()

    monkeypatch.setattr(vllm_server, "get_path_to_python_env_bin", lambda *_args, **_kwargs: "vllm")
    monkeypatch.setattr(vllm_server, "get_cuda_architecture_for_devices", lambda _devices: 8)
    monkeypatch.setattr(vllm_server, "kill_process_by_port", lambda _port: False)
    monkeypatch.setattr(vllm_server.subprocess, "Popen", _popen)
    monkeypatch.setattr(vllm_server.psutil, "Process", lambda _pid: _Process())

    conf = vllm_server.VLLMServer.Conf(
        enable_lora=False,
        allow_connect_to_existing=False,
        language_model_only=True,
        kv_cache_memory_bytes="4G",
        max_num_seqs=8,
        modern_cli=True,
        max_model_len=16384,
        seed=20260722,
    )
    server = vllm_server.VLLMServer(
        conf,
        port=8123,
        cuda_visible_devices=["1"],
        max_gpu_mem_utilization=0.82,
    )
    server.start_server(Path(".model_cache/Qwen/Qwen3.5-4B"))

    args = captured["args"]
    assert isinstance(args, list)
    assert "--language-model-only" in args
    assert "--no-enable-log-requests" in args
    assert "--disable-log-requests" not in args
    assert not any(str(arg).startswith("--swap-space") for arg in args)
    assert not any(str(arg).startswith("--max-seq-len-to-capture") for arg in args)
    assert args[args.index("--kv-cache-memory-bytes") + 1] == "4G"
    assert args[args.index("--max-num-seqs") + 1] == "8"
    assert "--enable-lora" not in args
    assert not any(str(arg).startswith("--max-lora-rank") for arg in args)
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["VLLM_NO_USAGE_STATS"] == "1"
    assert env["VLLM_PORT"] == str(vllm_server._vllm_internal_port_base(8123))


def test_concurrent_vllm_servers_get_disjoint_internal_port_blocks() -> None:
    bases = [
        vllm_server._vllm_internal_port_base(port)
        for port in (5555, 5557, 5559, 5561)
    ]

    assert len(set(bases)) == 4
    assert all(30_000 <= base <= 61_936 for base in bases)
    assert min(abs(left - right) for left in bases for right in bases if left != right) >= 64


def test_qwen35_base_experiment_name_uses_run_name() -> None:
    args = Namespace(run_name="qwen35_4b", split="dev", repeat=1)

    assert (
        _experiment_name(
            args=args,
            checkpoint_name="base",
            iteration=None,
            repeat_index=0,
        )
        == "eval_qwen35_4b_base_dev"
    )


def test_qwen35_hydra_config_composes_for_eval() -> None:
    cfg = get_config(
        "eval",
        [
            "llm=qwen_3_5_4b_eval",
            "experiment_name=qwen35_config_test",
        ],
    )

    assert cfg.llm.vllm_class._target_ == "phi_agents.rl.llm.VLLMQwen3"
    assert cfg.llm.vllm_class.enable_thinking is False
    assert cfg.llm.vllm_server.language_model_only is True
    assert cfg.llm.vllm_server.enable_lora is False
    assert cfg.llm.vllm_server.max_num_seqs == 8
    assert cfg.llm.vllm_server.max_model_len == 16384
    assert cfg.llm.vllm_server.max_tries == 900
    assert cfg.llm.vllm_server.modern_cli is True


def test_qwen35_lora_config_composes_for_direct_adapter_eval() -> None:
    cfg = get_config(
        "eval",
        [
            "llm=qwen_3_5_4b_lora32_eval",
            "experiment_name=qwen35_lora_config_test",
        ],
    )

    assert cfg.llm.vllm_server.enable_lora is True
    assert cfg.llm.vllm_server.max_lora_rank == 32
    assert cfg.llm.lora_rank == 32
    assert cfg.llm.vllm_class.enable_thinking is False


def test_direct_adapter_path_is_not_rewritten_as_checkpoint(tmp_path) -> None:
    adapter = tmp_path / "final_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    assert _is_adapter_directory(adapter)
    assert lora_path(adapter) == adapter
    assert lora_path(tmp_path / "checkpoint-1") == tmp_path / "checkpoint-1" / "lora"
