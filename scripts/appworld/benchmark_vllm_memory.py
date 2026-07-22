from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil
import requests
from transformers import AutoTokenizer


GPU_QUERY = (
    "index,memory.used,memory.total,utilization.gpu,utilization.memory,"
    "power.draw,temperature.gpu"
)
SAMPLE_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "phase",
    "gpu_index",
    "memory_used_mib",
    "memory_total_mib",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "power_w",
    "temperature_c",
    "system_memory_available_mib",
    "swap_used_mib",
]


def _number(value: str) -> float:
    text = value.strip().split()[0]
    return float(text) if text not in {"N/A", "[N/A]"} else float("nan")


class ResourceMonitor:
    def __init__(self, *, gpu_index: int, output_path: Path, interval: float) -> None:
        self.gpu_index = gpu_index
        self.output_path = output_path
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._phase = "initializing"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at = time.perf_counter()

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                sample = self._sample()
                if sample is not None:
                    self.samples.append(sample)
                    writer.writerow(sample)
                    fh.flush()
                self._stop.wait(self.interval)

    def _sample(self) -> dict[str, Any] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={GPU_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        selected: list[str] | None = None
        for line in result.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if values and values[0] == str(self.gpu_index):
                selected = values
                break
        if selected is None or len(selected) != 7:
            return None

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        with self._phase_lock:
            phase = self._phase
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(time.perf_counter() - self._started_at, 3),
            "phase": phase,
            "gpu_index": self.gpu_index,
            "memory_used_mib": _number(selected[1]),
            "memory_total_mib": _number(selected[2]),
            "gpu_utilization_pct": _number(selected[3]),
            "memory_utilization_pct": _number(selected[4]),
            "power_w": _number(selected[5]),
            "temperature_c": _number(selected[6]),
            "system_memory_available_mib": round(memory.available / 2**20, 1),
            "swap_used_mib": round(swap.used / 2**20, 1),
        }


def _wait_for_server(process: subprocess.Popen[str], base_url: str, timeout: int) -> str:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            response.raise_for_status()
            models = response.json()["data"]
            if models:
                return str(models[0]["id"])
        except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become healthy in {timeout}s: {last_error}")


def _fixed_prompt_tokens(tokenizer: Any, count: int) -> list[int]:
    text = (
        "You are evaluating an AppWorld automation agent. Read the available API "
        "documentation, preserve state across turns, recover from API errors, and "
        "complete the requested task accurately.\n"
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("Tokenizer produced an empty benchmark prompt")
    repeats = (count + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:count]


def _completion(
    *,
    base_url: str,
    model_id: str,
    prompt_tokens: list[int],
    output_tokens: int,
    seed: int,
) -> dict[str, float | int]:
    payload = {
        "model": model_id,
        "prompt": prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": 0.1,
        "ignore_eos": True,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0
    completion_tokens: int | None = None
    with requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        stream=True,
        timeout=(10, 900),
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line.startswith(b"data: "):
                continue
            body = raw_line[6:]
            if body == b"[DONE]":
                break
            event = json.loads(body)
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                chunks += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("Completion stream contained no generated token")
    return {
        "latency_seconds": finished - started,
        "ttft_seconds": first_token_at - started,
        "completion_tokens": completion_tokens if completion_tokens is not None else chunks,
    }


def _run_batch(
    *,
    base_url: str,
    model_id: str,
    prompt_tokens: list[int],
    output_tokens: int,
    concurrency: int,
    seed_base: int,
) -> tuple[float, list[dict[str, float | int]]]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _completion,
                base_url=base_url,
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                seed=seed_base + index,
            )
            for index in range(concurrency)
        ]
        results = [future.result() for future in futures]
    return time.perf_counter() - started, results


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _phase_samples(monitor: ResourceMonitor, prefix: str) -> list[dict[str, Any]]:
    return [sample for sample in monitor.samples if str(sample["phase"]).startswith(prefix)]


def _mean(samples: list[dict[str, Any]], field: str) -> float:
    values = [float(sample[field]) for sample in samples]
    return statistics.fmean(values) if values else float("nan")


def _maximum(samples: list[dict[str, Any]], field: str) -> float:
    values = [float(sample[field]) for sample in samples]
    return max(values) if values else float("nan")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", default="qwen35_4b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=5755)
    parser.add_argument("--vllm-executable", default=str(Path(sys.executable).with_name("vllm")))
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=4 * 2**30)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measured-batches", type=int, default=3)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model.expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    monitor = ResourceMonitor(
        gpu_index=args.gpu,
        output_path=output_dir / "resource_samples.csv",
        interval=args.monitor_interval,
    )
    monitor.set_phase("baseline")
    monitor.start()
    time.sleep(3)

    command = [
        args.vllm_executable,
        "serve",
        model.as_posix(),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype=bfloat16",
        "--tensor-parallel-size=1",
        "--pipeline-parallel-size=1",
        "--max-model-len",
        str(args.max_model_len),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--enable-prefix-caching",
        "--generation-config=vllm",
        "--max-logprobs=1",
        "--no-enable-log-requests",
    ]
    if args.language_model_only:
        command.append("--language-model-only")
    if args.enforce_eager:
        command.append("--enforce-eager")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(name, None)

    server_log_path = output_dir / "vllm_server.log"
    process: subprocess.Popen[str] | None = None
    requests_log: list[dict[str, Any]] = []
    try:
        monitor.set_phase("startup")
        startup_started = time.perf_counter()
        with server_log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            base_url = f"http://127.0.0.1:{args.port}"
            model_id = _wait_for_server(process, base_url, args.startup_timeout)
            startup_seconds = time.perf_counter() - startup_started

            monitor.set_phase("loaded_idle")
            time.sleep(5)
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            prompt_tokens = _fixed_prompt_tokens(tokenizer, args.prompt_tokens)

            for batch_index in range(args.warmup_batches):
                monitor.set_phase(f"warmup_{batch_index + 1}")
                _run_batch(
                    base_url=base_url,
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    output_tokens=args.output_tokens,
                    concurrency=args.concurrency,
                    seed_base=1000 + batch_index * args.concurrency,
                )

            measured_elapsed = 0.0
            for batch_index in range(args.measured_batches):
                monitor.set_phase(f"measured_{batch_index + 1}")
                batch_elapsed, batch_results = _run_batch(
                    base_url=base_url,
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    output_tokens=args.output_tokens,
                    concurrency=args.concurrency,
                    seed_base=2000 + batch_index * args.concurrency,
                )
                measured_elapsed += batch_elapsed
                for request_index, result in enumerate(batch_results):
                    requests_log.append(
                        {
                            "batch": batch_index + 1,
                            "request": request_index,
                            **result,
                        }
                    )

            monitor.set_phase("cooldown")
            time.sleep(3)

        loaded_samples = _phase_samples(monitor, "loaded_idle")
        measured_samples = _phase_samples(monitor, "measured_")
        baseline_samples = _phase_samples(monitor, "baseline")
        ttfts = [float(item["ttft_seconds"]) for item in requests_log]
        latencies = [float(item["latency_seconds"]) for item in requests_log]
        generated_tokens = sum(int(item["completion_tokens"]) for item in requests_log)
        summary = {
            "model": args.model_label,
            "model_path": model.as_posix(),
            "gpu_index": args.gpu,
            "max_model_len": args.max_model_len,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_num_seqs": args.max_num_seqs,
            "prompt_tokens": args.prompt_tokens,
            "requested_output_tokens": args.output_tokens,
            "concurrency": args.concurrency,
            "warmup_batches": args.warmup_batches,
            "measured_batches": args.measured_batches,
            "language_model_only": args.language_model_only,
            "enforce_eager": args.enforce_eager,
            "startup_seconds": round(startup_seconds, 3),
            "baseline_memory_mib": round(_mean(baseline_samples, "memory_used_mib"), 1),
            "loaded_idle_memory_mib": round(_mean(loaded_samples, "memory_used_mib"), 1),
            "peak_inference_memory_mib": round(
                _maximum(measured_samples, "memory_used_mib"), 1
            ),
            "mean_inference_gpu_utilization_pct": round(
                _mean(measured_samples, "gpu_utilization_pct"), 1
            ),
            "peak_inference_gpu_utilization_pct": round(
                _maximum(measured_samples, "gpu_utilization_pct"), 1
            ),
            "peak_power_w": round(_maximum(measured_samples, "power_w"), 1),
            "generated_tokens": generated_tokens,
            "measured_elapsed_seconds": round(measured_elapsed, 3),
            "throughput_tokens_per_second": round(generated_tokens / measured_elapsed, 2),
            "median_ttft_seconds": round(statistics.median(ttfts), 4),
            "p95_ttft_seconds": round(_percentile(ttfts, 0.95), 4),
            "median_request_latency_seconds": round(statistics.median(latencies), 4),
        }
        (output_dir / "benchmark_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "request_metrics.json").write_text(
            json.dumps(requests_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (output_dir / "memory_benchmark.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        monitor.set_phase("shutdown")
        if process is not None:
            _stop_process(process)
        monitor.stop()


if __name__ == "__main__":
    main()
