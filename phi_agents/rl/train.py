#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import datetime
import gc
import itertools
import json
import math
import os
import random
import sys
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra.utils
import numpy as np
import psutil
import ray
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchtune.training.checkpointing._utils import safe_torch_load
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

import phi_agents.utils.file_utils as fu
from phi_agents.rl.cloud_checkpointer import (
    CloudCheckpointer,
    checkpoint_name,
    get_last_checkpoint,
)
from phi_agents.rl.config import get_config
from phi_agents.rl.iteration_manifest import (
    IterationManifestStore,
    manifest_abort_fields_from_exception,
)
from phi_agents.rl.parallel_scenario_sampler import ParallelScenarioSampler
from phi_agents.rl.rl_utils import (
    Baseline,
    GradientRMS,
    LossType,
    RLAlgorithm,
    compute_advantage_estimates,
    inference_and_learning_share_gpus,
    is_policy_gradient_loss,
    wandb_init,
)
from phi_agents.rl.rollout_diagnostics import (
    compute_rollout_diagnostics,
    save_rollout_diagnostics,
    save_sanitized_trajectories,
)
from phi_agents.rl.type_defs import TrainingRollout
from phi_agents.rl.utils.download import (
    distributed_download_models,
)
from phi_agents.rl.utils.fsdp2_utils import (
    setup_cpu_offload_policy,
    setup_fsdp2_model,
    setup_mixed_precision_policy,
)
from phi_agents.rl.utils.ray_utils import (
    VLLM_RESOURCE,
    connect_ray_cluster,
    run_on_nodes_with_resource,
)
from phi_agents.rl.vllm_rollout_worker import VLLMRolloutWorker
from phi_agents.utils.cce import (
    EfficientLMOutput,
    enable_memory_efficient_forward,
)
from phi_agents.utils.logger import NullLogger, get_phi_logger
from phi_agents.utils.profiling import (
    NVMLPeakMemProfiler,
    Speedometer,
    profile,
    profiler_load_state_dict,
    profiler_state_dict,
    profiling_memory_summary,
    profiling_summary,
    set_profiling_num_elements,
)
from phi_agents.utils.utils import barrier_guard, timeit, torch_dist_barrier

if TYPE_CHECKING:
    import logging

logger = get_phi_logger()
null_logger = NullLogger()


type OptimizedModule = Any  # torch._dynamo.eval_frame.OptimizedModule
type ModelType = PeftModel | OptimizedModule


_RANK_TRAINING_STATE_SCHEMA_VERSION = 1
_RANK_TRAINING_STATE_DIRECTORY = "training_state"


def _numpy_global_rng_state() -> dict[str, Any]:
    """Return the legacy NumPy RNG state using weights-only-safe values."""
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _set_numpy_global_rng_state(state: dict[str, Any]) -> None:
    keys = state["keys"]
    if not isinstance(keys, torch.Tensor):
        raise ValueError("NumPy RNG keys must be stored as a torch.Tensor")
    np.random.set_state(
        (
            str(state["bit_generator"]),
            keys.cpu().numpy().astype(np.uint32, copy=False),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


def _capture_rng_state(training_rng: np.random.Generator) -> dict[str, Any]:
    """Capture all RNGs that can affect learner updates after a checkpoint."""
    return {
        "python": random.getstate(),
        "numpy_global": _numpy_global_rng_state(),
        # ``default_rng`` uses PCG64 here, whose state is made only of safe
        # primitive values. This generator chooses the minibatch/epoch order.
        "training_sampler": training_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any], training_rng: np.random.Generator) -> None:
    """Restore RNGs captured by :func:`_capture_rng_state`."""
    random.setstate(state["python"])
    _set_numpy_global_rng_state(state["numpy_global"])
    training_rng.bit_generator.state = state["training_sampler"]
    torch.set_rng_state(state["torch_cpu"])

    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state, but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count changed across resume: "
                f"checkpoint={len(cuda_states)} current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _capture_rank_training_state(
    *,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    training_rng: np.random.Generator,
    rank: int,
    world_size: int,
    gradient_rms: GradientRMS,
    invalid_steps_skipped: int,
    high_kl_events: int,
    n_outlier_grads: int,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Build the rank-local state required for an exact iteration-boundary resume."""
    return {
        "schema_version": _RANK_TRAINING_STATE_SCHEMA_VERSION,
        "rank": int(rank),
        "world_size": int(world_size),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "rng": _capture_rng_state(training_rng),
        "gradient_rms": {
            "mean": float(gradient_rms.mean),
            "var": float(gradient_rms.var),
            "count": int(gradient_rms.count),
        },
        "counters": {
            "invalid_steps_skipped": int(invalid_steps_skipped),
            "high_kl_events": int(high_kl_events),
            "n_outlier_grads": int(n_outlier_grads),
        },
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
    }


def _restore_rank_training_state(
    state: dict[str, Any],
    *,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    training_rng: np.random.Generator,
    rank: int,
    world_size: int,
    gradient_rms: GradientRMS,
    scaler: Any | None = None,
) -> dict[str, int]:
    """Restore optimizer, scheduler, RNG and rank-local trainer statistics."""
    if state.get("schema_version") != _RANK_TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported rank training-state schema: " f"{state.get('schema_version')!r}"
        )
    if int(state.get("rank", -1)) != rank:
        raise ValueError(f"Checkpoint rank {state.get('rank')!r} does not match rank {rank}")
    if int(state.get("world_size", -1)) != world_size:
        raise ValueError(
            "Checkpoint world size does not match the current run: "
            f"{state.get('world_size')!r} != {world_size}"
        )

    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])

    gradient_rms_state = state["gradient_rms"]
    gradient_rms.mean = float(gradient_rms_state["mean"])
    gradient_rms.var = float(gradient_rms_state["var"])
    gradient_rms.count = int(gradient_rms_state["count"])

    scaler_state = state.get("grad_scaler")
    if scaler_state is not None:
        if scaler is None:
            raise RuntimeError("Checkpoint contains GradScaler state, but no scaler is active")
        scaler.load_state_dict(scaler_state)

    # Restore RNG last: model construction, wrapping, and state loading are
    # allowed to consume randomness without changing the continued sequence.
    _restore_rng_state(state["rng"], training_rng)
    return {key: int(value) for key, value in state["counters"].items()}


def get_temp_local_directory(local_rank: int) -> Path:
    paths: list[Path | None]
    if local_rank == 0:
        path = Path(tempfile.TemporaryDirectory().name)
        path.mkdir()
        paths = [path]
    else:
        paths = [None]
    dist.broadcast_object_list(paths, src=0)
    assert isinstance(paths[0], Path)
    return paths[0]


def kl_estimate(log_importance_weights: np.ndarray) -> float:
    # input: log(p(x) / q(x)) for samples x ~ q
    return -np.mean(log_importance_weights)  # type: ignore


def improved_kl_estimate(log_importance_weights: np.ndarray) -> float:
    # input: log(p(x) / q(x)) for samples x ~ q
    # k3 from http://joschu.net/blog/kl-approx.html
    logr = log_importance_weights
    return np.mean((np.exp(logr) - 1) - logr)  # type: ignore


def compute_gpu_mem_utilization(
    cfg: DictConfig, llm_cfg: DictConfig, local_rank: int, rank: int
) -> tuple[bool, float | None, float | None]:
    """Compute whether exclusive inference & learning, and max inference/learn mem utilization."""
    share_gpus: bool = inference_and_learning_share_gpus(
        cfg.gpu_allocation.inference_gpus, cfg.gpu_allocation.learning_gpus
    )

    if not share_gpus:
        return False, None, None  # this is to speedup startup and avoid initializing CUDA early

    total_mem_gb = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)

    max_mem_fraction = 0.95  # maximum fraction of overall system memory allowed
    required_mem_gb = cfg.inference_requires_memory_gb + cfg.learning_requires_memory_gb
    required_fraction = required_mem_gb / total_mem_gb

    logger.info(
        f"{share_gpus=} {cfg.inference_requires_memory_gb=:.1f} {cfg.learning_requires_memory_gb=:.1f} "
        f"{total_mem_gb=:.2f} {max_mem_fraction=:.2f} {required_fraction=:.3f}"
    )

    # keep rollout worker and learner running together by default (if we can)
    _exclusive_inference_and_learning: bool = False

    _inference_max_mem_utilization: float | None = llm_cfg.max_gpu_mem_utilization
    if _inference_max_mem_utilization is not None:
        assert _inference_max_mem_utilization * total_mem_gb >= cfg.inference_requires_memory_gb, (
            f"Invalid configuration {_inference_max_mem_utilization=:.3f} {cfg.inference_requires_memory_gb=:.3f}, "
            f"consider relaxing inference memory requirements."
        )

    _learning_max_mem_utilization: float | None = None  # do not limit by default

    if share_gpus:
        if required_fraction > max_mem_fraction:
            assert (
                not cfg.async_rollouts
            ), "Not enough GPU memory to collect rollouts and learn at the same time"

            _exclusive_inference_and_learning = True
            logger.info(
                f"Not enough memory to keep learner and rollout worker in GPU memory "
                f"at the same time: {_exclusive_inference_and_learning=}. Current "
                f"configuration requires full cleanup between inference and learning stages."
            )
        else:
            # We should have enough GPU memory to do inference and learning in parallel, but
            # some memory management is required.
            # We split memory proportionally to the requested usage:
            _learning_max_mem_utilization = (
                cfg.learning_requires_memory_gb / required_mem_gb
            ) * max_mem_fraction
            inference_max_mem_utilization = (
                cfg.inference_requires_memory_gb / required_mem_gb
            ) * max_mem_fraction
            if _inference_max_mem_utilization is None:
                _inference_max_mem_utilization = inference_max_mem_utilization
            else:
                _inference_max_mem_utilization = min(
                    _inference_max_mem_utilization, inference_max_mem_utilization
                )

    logger.info(f"{_inference_max_mem_utilization=} {_learning_max_mem_utilization=}")

    return (
        _exclusive_inference_and_learning,
        _inference_max_mem_utilization,
        _learning_max_mem_utilization,
    )


@dataclass
class RolloutID:
    """Unique identifying information for a rollout."""

    global_step: int
    rank: int
    scenario_idx: int
    rollout_idx: int

    def __str__(self) -> str:
        return f"{self.global_step}_{self.rank}_{self.scenario_idx}_{self.rollout_idx}"


@dataclass
class PPODebugInfo:
    clipped: bool
    clipped_fraction: float
    clipped_observations: int
    total_observations: int
    epsilon: float
    policy_loss_1: float
    policy_loss_2: float


@dataclass
class RolloutLossDebugInfo:
    loss: float
    n_tokens: int
    n_output_tokens: int
    truncated: bool
    max_log_prob_diff: float
    min_log_prob_diff: float
    argmax_log_prob_diff: int
    argmin_log_prob_diff: int
    log_importance_weight: float
    importance_weight: float
    per_token_kl_sum: float
    advantage: float | np.ndarray
    ppo: PPODebugInfo | None


RolloutsAdvantagesIDs = tuple[list[TrainingRollout], np.ndarray, list[RolloutID]]


def get_rollout_ids(
    rollouts: list[list[TrainingRollout]], global_step: int, rank: int
) -> list[list[RolloutID]]:
    """Create identifying information for each rollout.

    Args:
        rollouts: The rollouts organized by scenario.
        global_step: The global step at which the rollouts were generated.
        rank: The rank on which the rollouts were generated.
    """
    ids: list[list[RolloutID]] = []
    for scenario_idx, per_scenario_rollouts in enumerate(rollouts):
        per_scenario_ids: list[RolloutID] = []
        for rollout_idx, _ in enumerate(per_scenario_rollouts):
            per_scenario_ids.append(RolloutID(global_step, rank, scenario_idx, rollout_idx))
        ids.append(per_scenario_ids)
    return ids


@dataclass
class RolloutStats:
    n_rollouts: int
    total_return: float
    n_total_tokens: int
    n_output_tokens: int


def _finite_scalar_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {
            "count": len(values),
            "finite_count": 0,
            "nonfinite_count": len(values),
            "mean": None,
            "min": None,
            "max": None,
            "last": None,
        }
    return {
        "count": len(values),
        "finite_count": len(finite_values),
        "nonfinite_count": len(values) - len(finite_values),
        "mean": float(np.mean(finite_values)),
        "min": min(finite_values),
        "max": max(finite_values),
        "last": finite_values[-1],
    }


def sampled_token_entropy_stats(rollouts: Sequence[TrainingRollout]) -> dict[str, Any]:
    """Estimate sampling entropy from sampled-token surprisal without another model pass.

    The result is the Monte Carlo mean of ``-log p(sampled_token)`` over output
    tokens. It estimates categorical entropy when rollouts are sampled from the
    reported distribution, but it is not an exact full-vocabulary entropy.
    """
    total_surprisal = 0.0
    finite_output_tokens = 0
    nonfinite_output_tokens = 0
    for rollout in rollouts:
        info = rollout.policy_token_info
        for is_output, log_probability in zip(info.is_output, info.log_probs, strict=True):
            if not is_output:
                continue
            if math.isfinite(float(log_probability)):
                total_surprisal -= float(log_probability)
                finite_output_tokens += 1
            else:
                nonfinite_output_tokens += 1

    return {
        "definition": "mean sampled-token surprisal -log(p), in nats",
        "finite_output_tokens": finite_output_tokens,
        "nonfinite_output_tokens": nonfinite_output_tokens,
        "mean_nats": (total_surprisal / finite_output_tokens if finite_output_tokens > 0 else None),
    }


@dataclass
class IterationMetricAccumulator:
    iteration: int
    global_step_start: int
    started_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    attempted_gradient_steps: int = 0
    optimizer_steps: int = 0
    invalid_loss_steps: int = 0
    high_kl_events: int = 0
    per_token_kl_sum: float = 0.0
    per_token_kl_observations: int = 0
    ppo_clipped_observations: int = 0
    ppo_total_observations: int = 0
    grad_norms: list[float] = field(default_factory=list)
    parameter_update_l2_norms: list[float] = field(default_factory=list)
    optimizer_learning_rates: list[float] = field(default_factory=list)
    advantage_filter: dict[str, Any] = field(default_factory=dict)
    sampling_entropy: dict[str, Any] = field(default_factory=dict)
    learning_rate_before_scheduler: float | None = None
    learning_rate_after_scheduler: float | None = None

    def record_gradient_observation(
        self,
        *,
        per_token_kl_sum: float,
        per_token_kl_observations: int,
        ppo_clipped_observations: int,
        ppo_total_observations: int,
    ) -> None:
        self.attempted_gradient_steps += 1
        self.per_token_kl_sum += float(per_token_kl_sum)
        self.per_token_kl_observations += int(per_token_kl_observations)
        self.ppo_clipped_observations += int(ppo_clipped_observations)
        self.ppo_total_observations += int(ppo_total_observations)

    def record_optimizer_result(
        self,
        *,
        grad_norm: float,
        optimizer_stepped: bool,
        parameter_update_l2_norm: float,
        learning_rate: float,
    ) -> None:
        self.grad_norms.append(float(grad_norm))
        if optimizer_stepped:
            self.optimizer_steps += 1
            self.parameter_update_l2_norms.append(float(parameter_update_l2_norm))
            self.optimizer_learning_rates.append(float(learning_rate))

    def report(
        self,
        *,
        status: str,
        global_step_end: int,
        cumulative_high_kl_events: int,
        failure_event: str | None = None,
        failure_type: str | None = None,
    ) -> dict[str, Any]:
        per_token_kl = (
            self.per_token_kl_sum / self.per_token_kl_observations
            if self.per_token_kl_observations > 0
            else None
        )
        clip_fraction = (
            self.ppo_clipped_observations / self.ppo_total_observations
            if self.ppo_total_observations > 0
            else None
        )
        return {
            "schema_version": "loop-iteration-training-metrics-v1",
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
            "status": status,
            "iteration": self.iteration,
            "started_at": self.started_at,
            "global_step_start": self.global_step_start,
            "global_step_end": global_step_end,
            "attempted_gradient_steps": self.attempted_gradient_steps,
            "actual_optimizer_steps": self.optimizer_steps,
            "per_token_kl": {
                "definition": "mean old_log_prob - new_log_prob over optimization token exposures",
                "mean": _finite_or_none(per_token_kl),
                "sum": _finite_or_none(self.per_token_kl_sum),
                "token_observations": self.per_token_kl_observations,
            },
            "ppo_clip_fraction": {
                "definition": "clipped PPO objective observations / PPO objective observations",
                "fraction": clip_fraction,
                "clipped_observations": self.ppo_clipped_observations,
                "total_observations": self.ppo_total_observations,
            },
            "grad_norm_before_clipping": {
                "definition": "global L2 norm returned before max-grad-norm clipping",
                **_finite_scalar_summary(self.grad_norms),
            },
            "parameter_update_l2_norm": {
                "definition": "global L2 norm of trainable parameter delta across actual optimizer steps",
                **_finite_scalar_summary(self.parameter_update_l2_norms),
            },
            "learning_rate": {
                "optimizer_steps": _finite_scalar_summary(self.optimizer_learning_rates),
                "before_iteration_scheduler_step": _finite_or_none(
                    self.learning_rate_before_scheduler
                ),
                "after_iteration_scheduler_step": _finite_or_none(
                    self.learning_rate_after_scheduler
                ),
            },
            "abs_adv_threshold_filter": self.advantage_filter,
            "sampling_entropy": self.sampling_entropy,
            "invalid_loss_steps": self.invalid_loss_steps,
            "high_kl_events": {
                "iteration": self.high_kl_events,
                "cumulative": cumulative_high_kl_events,
            },
            "failure": (
                {"event": failure_event, "exception_type": failure_type}
                if failure_event is not None
                else None
            ),
        }


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _write_metric_payload(path: Path, report: dict[str, Any]) -> None:
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    scheme, raw_path = fu.get_scheme_and_path(path)
    if scheme == "file":
        destination = Path(raw_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_name).replace(destination)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
    else:
        with tempfile.NamedTemporaryFile("w", delete=False) as temporary:
            temporary.write(data)
            temporary_name = temporary.name
        try:
            fu.copy(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)


def write_iteration_metric_report(cloud_path: Path, report: dict[str, Any]) -> Path:
    """Atomically persist metrics without downgrading a completed canonical report."""
    iteration = int(report["iteration"])
    path = cloud_path / "training_metrics" / f"iteration-{iteration:06d}.json"
    if report.get("status") == "failed" and fu.exists(path):
        with fu.uri_open(path, "r") as file_handle:
            existing = json.load(file_handle)
        if isinstance(existing, dict) and existing.get("status") == "completed":
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            audit_path = (
                cloud_path
                / "training_metrics"
                / "interruption_audits"
                / f"iteration-{iteration:06d}-{timestamp}.json"
            )
            audit_report = {
                **report,
                "schema_version": "loop-iteration-interruption-audit-v1",
                "source_metrics_schema_version": report.get("schema_version"),
                "record_kind": "post_completion_interruption_audit",
                "canonical_status_preserved": "completed",
            }
            _write_metric_payload(audit_path, audit_report)
            return audit_path

    _write_metric_payload(path, report)
    return path


@torch.no_grad()
def _snapshot_trainable_parameters(model: ModelType) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (parameter, parameter.detach().clone())
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def _distributed_all_reduce(tensor: torch.Tensor, op: dist.ReduceOp) -> torch.Tensor:
    """Reduce a tensor on a device supported by the active process-group backend."""
    if not dist.is_initialized():
        return tensor
    if dist.get_backend() == "nccl" and tensor.device.type != "cuda":
        reduced = tensor.to(torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(reduced, op=op)
        return reduced.to(tensor.device)
    dist.all_reduce(tensor, op=op)
    return tensor


@torch.no_grad()
def _parameter_update_l2_norm(
    snapshots: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    local_squared_norm: torch.Tensor | None = None
    for parameter, before in snapshots:
        delta = parameter.detach() - before
        if hasattr(delta, "to_local"):
            delta = delta.to_local()
        delta_float = delta.float()
        squared_norm = torch.sum(delta_float * delta_float)
        local_squared_norm = (
            squared_norm
            if local_squared_norm is None
            else local_squared_norm + squared_norm.to(local_squared_norm.device)
        )
    if local_squared_norm is None:
        return 0.0
    local_squared_norm = _distributed_all_reduce(local_squared_norm, dist.ReduceOp.SUM)
    return float(torch.sqrt(local_squared_norm).item())


def _log_training_failure(exc: BaseException) -> None:
    """Emit one idempotent Dragon Sentinel alert without serializing exception content."""
    if getattr(exc, "_dragon_sentinel_training_failure_logged", False):
        return
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        logger.critical(
            "CUDA out of memory interrupted training.",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={"event": "cuda_oom"},
        )
    elif isinstance(exc, KeyboardInterrupt):
        logger.error(
            "Training was interrupted before completion.",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={"event": "training_failed"},
        )
    else:
        logger.error(
            "Training failed with an unhandled exception.",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={"event": "training_failed"},
        )
    try:
        exc._dragon_sentinel_training_failure_logged = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


class RLOOTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        full_cfg: DictConfig,
        local_rank: int,
        rank: int,
        world_size: int,
    ):
        self._accelerator = accelerator

        self._experiment_name = full_cfg.experiment_name
        self._full_cfg = full_cfg
        self._cfg = cfg = full_cfg.rl
        self._llm_cfg = full_cfg.llm
        self._wandb_cfg = full_cfg.wandb
        self._local_rank = local_rank

        self._device = self._accelerator.device

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg.dtype]
        self._dtype = torch_dtype

        self._rank = rank
        self._world_size = world_size

        self._rank0_logger: logging.Logger | NullLogger = logger if self._rank == 0 else null_logger

        if not self._llm_cfg.compile_torch_model:
            # better numerical stability
            os.environ["TORCHDYNAMO_DISABLE"] = "1"

        self._rng = np.random.default_rng(seed=cfg.seed)
        self._local_path = get_temp_local_directory(local_rank)  # local working directory

        self._cfg.cloud_path = Path(self._cfg.cloud_path)  # Convert to Path
        scheme, _ = fu.get_scheme_and_path(self._cfg.cloud_path)
        if scheme == "file":
            self._cfg.cloud_path.mkdir(parents=True, exist_ok=True)

        self._cloud_checkpointer = CloudCheckpointer(
            self._cfg.cloud_path, self._local_path, local_rank, max_ckpts=self._cfg.max_ckpts
        )
        iteration_manifest_cfg = self._cfg.get("iteration_manifest", {})
        self._iteration_manifest_store = IterationManifestStore(
            self._cfg.cloud_path,
            enabled=bool(iteration_manifest_cfg.get("enabled", True)),
        )
        self._speedometer = Speedometer(avg_period_seconds=900.0)  # 15 min
        assert (
            cfg.params.scenarios_per_iteration % world_size == 0
        ), f"{cfg.params.scenarios_per_iteration=} {world_size=}"
        assert self._cfg.params.minibatch_size % world_size == 0

        if self._cfg.recompute_rollout_probs and self._cfg.async_rollouts:
            raise ValueError("recompute_rollout_probs is incompatible with async_rollouts.")

        self._validate_algorithm_config()

        # Logging
        if self._rank == 0:
            logger.info(
                f"{self._cfg.params.minibatch_size=}, {cfg.params.rollouts_per_scenario=} {world_size=}"
            )

            # log the config to cloud if it doesn't already exist
            config_cloud_path = self._cfg.cloud_path / "conf.yaml"
            if not fu.exists(config_cloud_path):
                with tempfile.NamedTemporaryFile("w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                    f.flush()
                    fu.copy(f.name, config_cloud_path)

        if (
            self._cfg.inference_requires_memory_gb is not None
            or self._cfg.learning_requires_memory_gb is not None
        ):
            assert (
                self._cfg.inference_requires_memory_gb is not None
                and self._cfg.learning_requires_memory_gb is not None
            ), "Need to specify both to compute memory requirements"
            (
                self._exclusive_inference_and_learning,
                _inference_max_mem_utilization,
                _learning_max_mem_utilization,
            ) = compute_gpu_mem_utilization(
                cfg=self._cfg, llm_cfg=self._llm_cfg, local_rank=self._local_rank, rank=self._rank
            )
        else:
            # if there are no vllm_server resources then use this
            self._exclusive_inference_and_learning = inference_and_learning_share_gpus(
                self._cfg.gpu_allocation.inference_gpus, self._cfg.gpu_allocation.learning_gpus
            )
            _inference_max_mem_utilization = self._llm_cfg.max_gpu_mem_utilization
            _learning_max_mem_utilization = None

        # Limit the max mem utilization for learning
        if _learning_max_mem_utilization is not None:
            logger.info(
                f"rank{rank}: limiting the learner memory usage to {_learning_max_mem_utilization:.3f}"
            )
            torch.cuda.set_per_process_memory_fraction(_learning_max_mem_utilization)

        self._file_upload_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"file_upload_rank_{rank}"
        )

        if self._llm_cfg.adapter_path is not None:
            raise NotImplementedError(f"{self._llm_cfg.adapter_path=}")

        # download base model and construct tokenizer
        with timeit("download_models", logger):
            self._base_model_local_dir = Path(
                distributed_download_models(self._llm_cfg.base_model_path, local_rank)
            )
        self._model_config = json.loads((self._base_model_local_dir / "config.json").read_text())
        self._model: ModelType | None = None
        self._tokenizer: PreTrainedTokenizer | None = None

        self._optimizer: Optimizer | None = None
        self._lr_scheduler: LRScheduler | None = None
        self._grad_rms = GradientRMS()

        self._iterations_completed = 0
        self._global_step = 0
        self._rollouts_generated = self._output_tokens_generated = 0
        self._active_iteration_manifest: tuple[int, int] | None = None
        self._iteration_metrics: IterationMetricAccumulator | None = None

        self._with_wandb = self._rank == 0 and self._wandb_cfg.enable
        wandb_run_name: str | None = None
        if self._with_wandb:
            wandb_run_name = wandb_init(full_cfg)

        # in some envs scenario generation can be costly and is parallelized
        n_cpu_cores = psutil.cpu_count(logical=True) or 16
        max_parallel_samplers = cfg.scenario_sampler.get("max_parallel", 1)
        sampler_num_threads = int(min(max_parallel_samplers, max(1, n_cpu_cores // world_size)))
        self._scenario_sampler = ParallelScenarioSampler(
            create_sampler_func=lambda: hydra.utils.instantiate(cfg.scenario_sampler),
            num_threads=int(sampler_num_threads),
        )

        self._rollout_worker = VLLMRolloutWorker(
            scenario_sampler=self._scenario_sampler,
            rollouts_per_scenario=self._cfg.params.rollouts_per_scenario,
            runner_cfg=cfg.scenario_runner,
            rank=rank,
            local_rank=local_rank,
            barrier=self._accelerator.wait_for_everyone,
            inference_gpus=self._cfg.gpu_allocation.inference_gpus,
            exclusive_inference_and_learning=self._exclusive_inference_and_learning,
            max_gpu_mem_utilization=_inference_max_mem_utilization,
            num_runners=cfg.num_scenario_runners,
            llm_cfg=self._llm_cfg,
        )

        import phi_agents.rl.callbacks as cb  # avoiding circular imports...

        callbacks: list[cb.Callback] = []
        if self._rank == 0:
            eval_callback = cb.EvalCallback(
                self, self._wandb_cfg.project, self._wandb_cfg.group, wandb_run_name
            )
            callbacks = [eval_callback]
        self._callbacks = cb.CallbackList(callbacks)

        self._invalid_steps_skipped: int = 0
        self._high_kl_events: int = 0
        self._n_outlier_grads: int = 0

    def _validate_algorithm_config(self) -> None:
        algorithm = RLAlgorithm(self._cfg.params.get("algorithm", RLAlgorithm.LOOP))

        if self._cfg.params.rollouts_per_scenario < 2:
            raise ValueError(
                "At least two rollouts per scenario are required for grouped advantage estimates."
            )

        if algorithm == RLAlgorithm.GRPO:
            loss_type = LossType(self._cfg.params.loss_type)
            if loss_type != LossType.PG_PER_TOKEN:
                raise ValueError(f"GRPO requires loss_type={LossType.PG_PER_TOKEN}.")
            if not self._cfg.params.do_ppo_clipping:
                raise ValueError("GRPO requires do_ppo_clipping=True.")
            if not math.isclose(float(self._cfg.params.ppo_epsilon), 0.1):
                raise ValueError("GRPO requires ppo_epsilon=0.1.")
            if not math.isclose(float(self._cfg.params.rloo_kl_lambda), 0.0):
                raise ValueError("GRPO does not use rloo_kl_lambda; set it to 0.0.")

    def _get_tokenizer(self) -> PreTrainedTokenizer:
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_local_dir)

        assert self._tokenizer is not None, f"{self._tokenizer=} !!!"
        return self._tokenizer

    def _save_lora_checkpoint(self, checkpoint_dir: Path) -> None:
        assert self._model is not None
        assert self._optimizer is not None
        assert self._lr_scheduler is not None
        if self._cfg.fsdp:  # Run on all processes
            self._accelerator.wait_for_everyone()
            unwrapped = self._accelerator.unwrap_model(self._model)

            with torch.no_grad():  # this needs to be called on all ranks!
                self._rank0_logger.info(f"Getting full LORA tensors...")
                lora_state_dict = {
                    k: v.to(self._device).full_tensor()
                    for k, v in unwrapped.state_dict().items()
                    if "lora_" in k
                }
                lora_dir = fu.lora_path(checkpoint_dir)
                assert lora_dir is not None, f"{lora_dir=} {checkpoint_dir=}"
                lora_dir.mkdir(parents=True, exist_ok=True)

                self._rank0_logger.info(f"Saving LORA adapter...")
                unwrapped.save_pretrained(
                    lora_dir,
                    is_main_process=True,
                    save_function=self._accelerator.save,
                    state_dict=lora_state_dict,
                    safe_serialization=True,
                )
                self._accelerator.wait_for_everyone()
                self._rank0_logger.info(f"Saving LORA adapter...Done!")
        else:
            raise AssertionError(f"{self._cfg.fsdp=} is not supported")

        # Optimizer state is rank-local under FSDP. Save one compact file per
        # rank instead of Accelerate's full-model checkpoint. The barrier keeps
        # the cloud uploader from committing a partially written checkpoint.
        self._save_rank_training_state(checkpoint_dir)
        self._accelerator.wait_for_everyone()

        if self._rank == 0:
            self._save_trainer_state(checkpoint_dir / "trainer_state.pt")

        self._accelerator.wait_for_everyone()

    def _rank_training_state_path(self, checkpoint_dir: Path) -> Path:
        return checkpoint_dir / _RANK_TRAINING_STATE_DIRECTORY / f"rank-{self._rank:05d}.pt"

    def _save_rank_training_state(self, checkpoint_dir: Path) -> None:
        assert self._optimizer is not None
        assert self._lr_scheduler is not None
        state_path = self._rank_training_state_path(checkpoint_dir)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = _capture_rank_training_state(
            optimizer=self._optimizer,
            lr_scheduler=self._lr_scheduler,
            training_rng=self._rng,
            rank=self._rank,
            world_size=self._world_size,
            gradient_rms=self._grad_rms,
            invalid_steps_skipped=self._invalid_steps_skipped,
            high_kl_events=self._high_kl_events,
            n_outlier_grads=self._n_outlier_grads,
            scaler=self._accelerator.scaler,
        )
        temporary_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
        try:
            torch.save(state, temporary_path)
            temporary_path.replace(state_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._rank0_logger.info(f"Saved exact rank training state to {state_path}")

    def _load_rank_training_state(
        self,
        checkpoint_dir: Path,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
    ) -> bool:
        state_path = self._rank_training_state_path(checkpoint_dir)
        if not state_path.is_file():
            self._rank0_logger.warning(
                "Checkpoint has no exact rank training state; continuing with a fresh "
                "optimizer and reconstructed scheduler. This legacy resume cannot recover "
                f"Adam moments or RNG streams: {checkpoint_dir}"
            )
            return False

        state = safe_torch_load(state_path, mmap=False)
        counters = _restore_rank_training_state(
            state,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            training_rng=self._rng,
            rank=self._rank,
            world_size=self._world_size,
            gradient_rms=self._grad_rms,
            scaler=self._accelerator.scaler,
        )
        self._invalid_steps_skipped = counters["invalid_steps_skipped"]
        self._high_kl_events = counters["high_kl_events"]
        self._n_outlier_grads = counters["n_outlier_grads"]
        self._rank0_logger.info(
            "Restored exact optimizer, scheduler, RNG, minibatch sampler, and trainer "
            f"statistics from {state_path}"
        )
        return True

    def _save_trainer_state(
        self,
        trainer_state_path: Path,
    ) -> None:
        trainer_state = {
            "checkpoint_schema_version": 2,
            "iterations_completed": self._iterations_completed,
            "output_tokens_generated": self._output_tokens_generated,
            "rollouts_generated": self._rollouts_generated,
            "profiler_state": profiler_state_dict(),
            "global_step": self._global_step,
        }
        torch.save(trainer_state, trainer_state_path)

    def _load_trainer_state(self, checkpoint_dir: Path) -> None:
        trainer_state = safe_torch_load(checkpoint_dir / f"trainer_state.pt", mmap=False)
        state_vars = (
            "iterations_completed",
            "output_tokens_generated",
            "rollouts_generated",
            "global_step",
        )
        for var_name in state_vars:
            if var_name not in trainer_state:
                raise ValueError(f"Could not load {var_name} from {trainer_state=}")

            setattr(self, f"_{var_name}", trainer_state[var_name])
            self._rank0_logger.info(
                f"Loaded {var_name}={trainer_state[var_name]} from checkpoint {str(checkpoint_dir)}..."
            )

        profiler_load_state_dict(trainer_state["profiler_state"], logger)

    def _expected_rollouts_per_iteration(self) -> int:
        return int(
            self._cfg.params.scenarios_per_iteration * self._cfg.params.rollouts_per_scenario
        )

    def _target_iteration(self) -> int:
        return int(self._iterations_completed + 1)

    def _begin_iteration_manifest(self, target_iteration: int, expected_rollouts: int) -> None:
        if self._rank != 0:
            return
        manifest = self._iteration_manifest_store.begin(
            iteration=target_iteration,
            expected_rollouts=expected_rollouts,
        )
        logger.info(
            "iteration manifest pending: "
            f"iteration={manifest.iteration}, expected_rollouts={manifest.expected_rollouts}, "
            f"finished_rollouts={manifest.finished_rollouts}, status={manifest.status}, "
            f"attempt={manifest.attempt}"
        )

    def _commit_iteration_manifest(
        self, target_iteration: int, expected_rollouts: int, finished_rollouts: int
    ) -> None:
        if self._rank == 0:
            manifest = self._iteration_manifest_store.commit(
                iteration=target_iteration,
                expected_rollouts=expected_rollouts,
                finished_rollouts=finished_rollouts,
            )
            logger.info(
                "iteration manifest committed: "
                f"iteration={manifest.iteration}, expected_rollouts={manifest.expected_rollouts}, "
                f"finished_rollouts={manifest.finished_rollouts}, status={manifest.status}, "
                f"attempt={manifest.attempt}"
            )

        self._accelerator.wait_for_everyone()
        self._iteration_manifest_store.assert_committed(
            iteration=target_iteration,
            expected_rollouts=expected_rollouts,
        )

    def _global_finished_rollout_count(self, n_rollouts: int) -> int:
        finished_rollouts: list[int | None] = [n_rollouts if self._rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(finished_rollouts, src=0)
        assert finished_rollouts[0] is not None
        return int(finished_rollouts[0])

    def _abort_iteration_manifest(
        self, target_iteration: int, expected_rollouts: int, exc: BaseException
    ) -> None:
        if self._rank != 0:
            return
        fields = manifest_abort_fields_from_exception(exc)
        manifest = self._iteration_manifest_store.abort(
            iteration=target_iteration,
            expected_rollouts=expected_rollouts,
            finished_rollouts=min(int(fields["finished_rollouts"]), expected_rollouts),
            reason=str(fields["reason"]),
            server_pid=fields["server_pid"],
            server_port=fields["server_port"],
            task_id=fields["task_id"],
        )
        logger.error(
            "iteration manifest aborted: "
            f"iteration={manifest.iteration}, expected_rollouts={manifest.expected_rollouts}, "
            f"finished_rollouts={manifest.finished_rollouts}, status={manifest.status}, "
            f"reason={manifest.reason}, server_pid={manifest.server_pid}, "
            f"server_port={manifest.server_port}, task_id={manifest.task_id}"
        )

    def _abort_active_iteration_manifest(self, exc: BaseException) -> None:
        if self._active_iteration_manifest is None:
            return
        target_iteration, expected_rollouts = self._active_iteration_manifest
        self._abort_iteration_manifest(target_iteration, expected_rollouts, exc)

    def _persist_iteration_metrics(
        self,
        *,
        status: str,
        failure_event: str | None = None,
        exc: BaseException | None = None,
    ) -> Path | None:
        if self._rank != 0 or self._iteration_metrics is None:
            return None
        if (
            self._iteration_metrics.learning_rate_after_scheduler is None
            and self._lr_scheduler is not None
        ):
            self._iteration_metrics.learning_rate_after_scheduler = float(
                self._lr_scheduler.get_last_lr()[0]
            )
        report = self._iteration_metrics.report(
            status=status,
            global_step_end=self._global_step,
            cumulative_high_kl_events=self._high_kl_events,
            failure_event=failure_event,
            failure_type=type(exc).__name__ if exc is not None else None,
        )
        path = write_iteration_metric_report(self._cfg.cloud_path, report)
        logger.info(
            "Persisted iteration training metrics: iteration=%s status=%s",
            self._iteration_metrics.iteration,
            status,
        )
        return path

    def _persist_failed_iteration_metrics(self, exc: BaseException, *, failure_event: str) -> None:
        try:
            self._persist_iteration_metrics(
                status="failed",
                failure_event=failure_event,
                exc=exc,
            )
        except Exception:
            logger.error(
                "Failed to persist interrupted iteration metrics.",
                exc_info=True,
                extra={"event": "training_metrics_persist_failed"},
            )

    def _persist_rollout_diagnostics(
        self,
        grouped_rollouts: list[list[TrainingRollout]],
        *,
        iteration: int,
    ) -> None:
        diagnostics_cfg = self._cfg.get("rollout_diagnostics", {})
        if not bool(diagnostics_cfg.get("enabled", False)):
            return

        gathered: list[list[list[TrainingRollout]] | None] | None = None
        if self._world_size > 1:
            gathered = [None] * self._world_size if self._rank == 0 else None
            dist.gather_object(grouped_rollouts, gathered, dst=0)
        if self._rank != 0:
            return

        if gathered is None:
            global_groups = grouped_rollouts
        else:
            global_groups = [
                group
                for worker_groups in gathered
                if worker_groups is not None
                for group in worker_groups
            ]

        from phi_agents.evals.appworld_rollout_data import AppWorldTrainingRollout

        if not all(
            isinstance(rollout, AppWorldTrainingRollout)
            for group in global_groups
            for rollout in group
        ):
            raise TypeError(
                "rl.rollout_diagnostics is only supported for AppWorld training rollouts"
            )
        appworld_groups = cast(list[list[AppWorldTrainingRollout]], global_groups)

        scheme, raw_cloud_path = fu.get_scheme_and_path(self._cfg.cloud_path)
        if scheme != "file":
            raise ValueError("rl.rollout_diagnostics currently requires a local file cloud_path")
        run_path = Path(raw_cloud_path)
        diagnostics = compute_rollout_diagnostics(
            appworld_groups,
            baseline=self._cfg.params.baseline,
            adv_normalization=self._cfg.params.adv_normalization,
            abs_adv_threshold=self._cfg.params.abs_adv_threshold,
            pos_adv_only=self._cfg.params.pos_adv_only,
        )
        diagnostics["training_iteration"] = iteration
        diagnostics["scenario_task_ids"] = [
            group[0].appworld_rollout_data.task.task_id for group in appworld_groups
        ]
        save_rollout_diagnostics(
            diagnostics,
            output_path=(run_path / "rollout_diagnostics" / f"iteration-{iteration:06d}.json"),
        )
        trajectory_count = 0
        if bool(diagnostics_cfg.get("save_trajectories", True)):
            trajectory_count = len(
                save_sanitized_trajectories(
                    appworld_groups,
                    output_root=run_path / "rollouts",
                    iteration=iteration,
                )
            )
        logger.info(
            "Persisted LOOP rollout diagnostics: iteration=%s groups=%s trajectories=%s",
            iteration,
            len(appworld_groups),
            trajectory_count,
            extra={"event": "loop_rollout_diagnostics_saved"},
        )

    def _compile_model(self, model: ModelType) -> None:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        for m in reversed(list(model.modules())):
            if isinstance(m, Qwen2DecoderLayer | Qwen3DecoderLayer):
                m.compile(backend=backend)

    def _setup_model_fsdp(self, checkpoint_dir: Path | None) -> ModelType:
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM

        # FSDP wrapping happens inside prepare()
        model = AutoModelForCausalLM.from_pretrained(
            self._base_model_local_dir, torch_dtype=self._dtype, use_cache=False
        )

        if self._llm_cfg.compile_torch_model:
            with timeit("compile_model", self._rank0_logger):
                # straightforward application of torch.compile manages to make the model slower AND use more memory
                # model = torch.compile(model, mode="default", dynamic=True)

                # this is closer to what torchtune does:
                self._compile_model(model)

        enable_memory_efficient_forward(model)

        if checkpoint_dir is None:
            from peft import LoraConfig, TaskType, get_peft_model

            self._rank0_logger.info("No checkpoint to load, initialize LORA from scratch...")

            lora_config = LoraConfig(
                r=self._llm_cfg.lora_rank,
                lora_alpha=self._llm_cfg.lora_alpha,
                lora_dropout=self._llm_cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=list(self._llm_cfg.lora_target_modules),
            )
            model = get_peft_model(model, lora_config)
        else:
            logger.info(f"Loading LoRA adapter from {checkpoint_dir}...")
            model = PeftModel.from_pretrained(
                model, checkpoint_dir / "lora", torch_dtype=self._dtype, is_trainable=True
            )

        model.print_trainable_parameters()
        assert isinstance(model, PeftModel)

        self._accelerator.wait_for_everyone()

        return model

    def _setup_model_optimizer_lr(
        self, checkpoint_dir: Path | None = None
    ) -> tuple[ModelType, Optimizer, LRScheduler]:
        with NVMLPeakMemProfiler("setup_model", logger=self._rank0_logger):
            if self._cfg.fsdp:
                model = self._setup_model_fsdp(checkpoint_dir)
            else:
                raise AssertionError(f"{self._cfg.fsdp=} not supported in this version!")

        if self._accelerator.is_fsdp2:
            model = setup_fsdp2_model(model, self._accelerator, self._rank0_logger)

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = hydra.utils.instantiate(self._cfg.optimization.optimizer, trainable_parameters)

        # We currently don't load optimizer and scheduler state, only LORA parameters are re-loaded
        # when the experiment is restarted, which seems to work fine in practice.
        # Full `accelerator_state` checkpoints are huge (investigate why?) while LORA-only checkpoints are tiny.
        # if checkpoint_dir is not None:
        #     train_state_dir = checkpoint_dir / "accelerator_state"
        #     if train_state_dir.exists():
        #         logger.info(f"Restoring optimizer / scheduler state from {train_state_dir=}")
        #         self._accelerator.load_state(str(train_state_dir))

        lr_scheduler = hydra.utils.instantiate(
            self._cfg.optimization.lr_scheduler,
            optimizer,
            num_training_steps=self._cfg.params.total_iterations,
            last_epoch=-1,
        )
        exact_state_available = (
            checkpoint_dir is not None and self._rank_training_state_path(checkpoint_dir).is_file()
        )
        if self._iterations_completed > 0 and not exact_state_available:
            # Backward compatibility for checkpoints written before exact
            # optimizer/scheduler state was introduced.
            lr_scheduler.step(self._iterations_completed)

        # FSDP / DDP wrapping
        with timeit("accelerator.prepare", self._rank0_logger):
            self._rank0_logger.info("Accelerator prepare...")

            if self._accelerator.is_fsdp2:
                # Accelerate gets upset if you wrapped the model with checkpointing/fsdp2
                # yourself if you use .prepare. We can work around this by calling
                # the prepare_* functions directly.
                lr_scheduler = self._accelerator.prepare_scheduler(lr_scheduler)
                model = self._accelerator.prepare_model(model)
                optimizer = self._accelerator.prepare_optimizer(optimizer)
            else:
                model, optimizer, lr_scheduler = self._accelerator.prepare(
                    model, optimizer, lr_scheduler
                )

        if checkpoint_dir is not None:
            self._load_rank_training_state(checkpoint_dir, optimizer, lr_scheduler)

        self._accelerator.wait_for_everyone()
        return model, optimizer, lr_scheduler

    def _compute_adv_estimates(self, rollouts: list[list[TrainingRollout]]) -> np.ndarray:
        """Compute rollout advantages according to the configured RL algorithm."""
        return compute_advantage_estimates(
            rollouts,
            algorithm=RLAlgorithm(self._cfg.params.get("algorithm", RLAlgorithm.LOOP)),
            baseline=Baseline(self._cfg.params.baseline),
            adv_normalization=bool(self._cfg.params.adv_normalization),
        )

    def _filter_rollouts(
        self, rollouts: list[TrainingRollout], adv_estimates: np.ndarray, ids: list[RolloutID]
    ) -> tuple[list[RolloutsAdvantagesIDs], float, float, dict[str, Any]]:
        # sort by decreasing absolute value
        # NOTE: Assume that adv threshold >= 0 and therefore neg adv will be filtered out later
        #       when pos_adv_only is true
        assert self._cfg.params.abs_adv_threshold >= 0
        abs_adv = adv_estimates if self._cfg.params.pos_adv_only else np.abs(adv_estimates)
        indices = np.argsort(-abs_adv)
        sorted_rollouts = [rollouts[i] for i in indices]
        sorted_ids = [ids[i] for i in indices]
        sorted_adv = adv_estimates[indices]
        sorted_abs_adv = abs_adv[indices]

        adv_threshold = self._cfg.params.abs_adv_threshold
        output_tokens = [sum(rollout.policy_token_info.is_output) for rollout in rollouts]
        below_threshold = sorted_abs_adv < adv_threshold
        below_threshold_rollouts = [
            sorted_rollouts[index] for index in np.where(below_threshold)[0]
        ]

        # this can be O(logn) but linear time here should be fine
        keep_indices = np.where(sorted_abs_adv >= adv_threshold)[0]
        n_keep = len(keep_indices)

        minibatch_size = self._cfg.params.minibatch_size
        assert minibatch_size <= len(rollouts)

        # round up to the nearest world_size but make sure not to exceed the total dataset size
        # each worker gets the same number of rollouts
        world_size = self._world_size
        n_keep_worlds = max(1, int(math.ceil(n_keep / world_size)))
        n_keep_worlds = min(n_keep_worlds, len(rollouts) // world_size)
        n_keep = n_keep_worlds * world_size

        assert n_keep <= len(sorted_rollouts)

        # we want to keep the same amount of work on all workers to maximize GPU utilization and
        # step synchronously
        adv_filtered_fraction = (len(rollouts) - n_keep) / len(rollouts)
        empirical_adv_filter_threshold = float(
            sorted_abs_adv[n_keep] if n_keep < len(rollouts) else 0.0
        )
        retained_output_tokens = sum(
            sum(rollout.policy_token_info.is_output) for rollout in sorted_rollouts[:n_keep]
        )
        filter_stats = {
            "definition": "output-token counts use complete rollout token metadata before sequence truncation",
            "configured_threshold": float(adv_threshold),
            "empirical_threshold_after_world_size_rounding": empirical_adv_filter_threshold,
            "actually_filtered_rollout_fraction": adv_filtered_fraction,
            "candidate_rollouts": len(sorted_rollouts),
            "candidate_output_tokens": sum(output_tokens),
            "below_threshold_rollouts": len(below_threshold_rollouts),
            "below_threshold_output_tokens": sum(
                sum(rollout.policy_token_info.is_output) for rollout in below_threshold_rollouts
            ),
            "retained_rollouts_after_world_size_rounding": n_keep,
            "retained_output_tokens_after_world_size_rounding": retained_output_tokens,
            "actually_filtered_rollouts": len(sorted_rollouts) - n_keep,
            "actually_filtered_output_tokens": sum(output_tokens) - retained_output_tokens,
        }

        logger.info(f"{n_keep=} {adv_filtered_fraction=:.2f} {empirical_adv_filter_threshold=:.5f}")

        # shuffle data before scattering back to the workers
        indices = self._rng.permutation(n_keep)
        rollouts = [sorted_rollouts[i] for i in indices]
        adv_estimates = sorted_adv[indices]
        ids = [sorted_ids[i] for i in indices]
        assert len(rollouts) == n_keep
        assert len(adv_estimates) == n_keep
        assert len(ids) == n_keep

        # split the rollouts into equal size subsets for each worker
        rollout_subsets: list[RolloutsAdvantagesIDs] = []
        assert n_keep % self._world_size == 0
        n_worker = n_keep // self._world_size
        for i in range(0, n_keep, n_worker):
            rollout_subsets.append(
                (
                    rollouts[i : i + n_worker],
                    adv_estimates[i : i + n_worker],
                    ids[i : i + n_worker],
                )
            )

        return (
            rollout_subsets,
            adv_filtered_fraction,
            empirical_adv_filter_threshold,
            filter_stats,
        )

    def _reduce_stats(self, rollouts: list[TrainingRollout]) -> RolloutStats:
        n_rollouts = torch.tensor(len(rollouts), device=self._device, dtype=torch.int)
        total_return = torch.tensor(
            sum([r.ret for r in rollouts]), device=self._device, dtype=torch.float32
        )
        n_total_tokens = torch.tensor(
            sum([len(r.policy_token_info.tokens) for r in rollouts]),
            device=self._device,
            dtype=torch.int,
        )
        n_output_tokens = torch.tensor(
            sum([sum(r.policy_token_info.is_output) for r in rollouts]),
            device=self._device,
            dtype=torch.int,
        )
        dist.reduce(n_rollouts, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_return, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(n_total_tokens, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(n_output_tokens, dst=0, op=dist.ReduceOp.SUM)
        return RolloutStats(
            int(n_rollouts.item()),
            total_return.item(),
            int(n_total_tokens.item()),
            int(n_output_tokens.item()),
        )

    def _log_probs(
        self, model: ModelType, tokens: torch.Tensor, is_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute log probabilities of output tokens under the current model.

        Returns float tensor of same length as input tokens tensor (first element always NaN).
        """
        (n_tokens,) = tokens.shape
        # add a batch dimension before passing into the model
        tokens = tokens.unsqueeze(0).contiguous()

        if self._llm_cfg.temperature == 1.0:
            temperature = None
        else:
            temperature = torch.tensor(self._llm_cfg.temperature, device=tokens.device)

        outputs: EfficientLMOutput = model(
            tokens, is_output=is_output.unsqueeze(0), temperature=temperature, use_cache=False
        )
        assert outputs.sampled_logprobs is not None
        assert outputs.hidden_states is not None

        log_probs = -outputs.sampled_logprobs.squeeze(0)
        last_hidden_state = None
        if len(outputs.hidden_states) > 0:
            last_hidden_state = outputs.hidden_states[-1].squeeze(0)

        assert log_probs.shape == (n_tokens - 1,), f"{log_probs.shape=} {n_tokens=}"
        assert log_probs[~is_output[1:]].sum() == 0
        log_probs = torch.cat([torch.tensor([torch.nan], device=self._device), log_probs])
        log_probs[~is_output] = torch.nan
        return last_hidden_state, log_probs

    def _get_tensors(
        self, rollout: TrainingRollout, id: RolloutID
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return tokens, is_output, and log_probs tensors, possibly truncating to maximum seq length.

        Also returns bool indicating whether truncation occured.
        """
        info = rollout.policy_token_info
        device = self._device
        n_tokens = len(info.tokens)
        max_seq_len = self._cfg.learning_max_seq_len or self._get_tokenizer().model_max_length
        if n_tokens > max_seq_len:
            logger.warning(
                f"Number of tokens ({n_tokens}) exceeds max_seq_len ({max_seq_len}). Truncating. "
                f"global_step={self._global_step}, rank={self._rank}, id={id}"
            )
            truncated = True
        else:
            truncated = False

        tokens = torch.tensor(info.tokens[:max_seq_len], dtype=torch.int, device=device)
        is_output = torch.tensor(info.is_output[:max_seq_len], dtype=torch.bool, device=device)
        log_probs = torch.tensor(info.log_probs[:max_seq_len], device=device)
        return tokens, is_output, log_probs, truncated

    def _ppo_loss(
        self, adv: torch.Tensor, importance_weight: torch.Tensor
    ) -> tuple[torch.Tensor, PPODebugInfo]:
        # NOTE: importance_weight may be either a scalar or a vector
        objective = importance_weight * adv

        # this version supports eps > 1.0, e.g. setting eps=1000 is practically equivalent to no clipping
        clip_min: float = 1 / (1 + self._cfg.params.ppo_epsilon)
        clip_max: float = 1 + self._cfg.params.ppo_epsilon

        # surrogate loss with PPO clipping
        # https://spinningup.openai.com/en/latest/algorithms/ppo.html
        def g(A: torch.Tensor) -> torch.Tensor:
            return torch.where(A >= 0, clip_max * A, clip_min * A)

        clip_threshold = g(adv)

        # binary 0/1 for full trajectories, fractional value for token-based clipping
        clipped_mask = (objective > clip_threshold).detach()
        ppo_clipped_observations = int(clipped_mask.sum().cpu())
        ppo_total_observations = clipped_mask.numel()
        ppo_clipped_fraction = ppo_clipped_observations / ppo_total_observations
        ppo_clipped_debug = ppo_clipped_observations > 0
        objective = torch.minimum(objective, clip_threshold)

        # just to check against this version
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
        policy_loss_1 = adv * importance_weight
        policy_loss_2 = adv * torch.clamp(importance_weight, clip_min, clip_max)
        policy_loss = torch.min(policy_loss_1, policy_loss_2)

        # Temporary to get more info on nan crashes
        if torch.isnan(objective).any():
            logger.warning(f"OBJECTIVE IS NAN: {adv=} {importance_weight=}")
        elif not torch.allclose(policy_loss, objective):
            max_diff = torch.abs(policy_loss - objective).max().cpu()
            raise ValueError(f"{policy_loss=} {objective=} mismatch: {max_diff=}")

        # NOTE: if importance_weight is a scalar then loss is a scalar
        # if importance_weight is a vector then loss is a vector
        loss = -objective

        debug_info = PPODebugInfo(
            clipped=ppo_clipped_debug,
            clipped_fraction=ppo_clipped_fraction,
            clipped_observations=ppo_clipped_observations,
            total_observations=ppo_total_observations,
            epsilon=self._cfg.params.ppo_epsilon,
            policy_loss_1=float(policy_loss_1.detach().mean().cpu()),
            policy_loss_2=float(policy_loss_2.detach().mean().cpu()),
        )
        return loss, debug_info

    def _loss(
        self,
        model: ModelType,
        rollout: TrainingRollout,
        monte_carlo_advantage: float,
        id: RolloutID,
    ) -> tuple[torch.Tensor, RolloutLossDebugInfo]:
        tokens, is_output, old_log_probs, truncated = self._get_tensors(rollout, id)
        n_tokens = len(tokens)
        n_output_tokens = int(is_output.sum())
        logger.info(f"rank{self._rank}: _loss() for {id=} with {n_tokens=} {n_output_tokens=}")

        last_hidden_state, new_log_probs = self._log_probs(model, tokens, is_output)
        assert new_log_probs.shape == (n_tokens,)
        tokens = tokens.cpu()

        advantage_estimates = torch.tensor(monte_carlo_advantage, device=new_log_probs.device)

        # compute importance weight
        new_log_probs_output_only = new_log_probs[is_output]
        old_log_probs_output_only = old_log_probs[is_output]

        # scalar
        new_log_prob = new_log_probs_output_only.sum()
        old_log_prob = old_log_probs_output_only.sum()

        trajectory_importance_weight = torch.exp(new_log_prob - old_log_prob)
        trajectory_log_importance_weight = float(new_log_prob) - float(old_log_prob)

        ppo_debug_info = None
        loss, ppo_debug_info = self._surrogate_loss(
            advantage_estimates,
            new_log_probs_output_only,
            old_log_probs_output_only,
            trajectory_importance_weight,
        )

        # debug info
        diff_log_probs = (
            new_log_probs_output_only.detach().cpu().numpy()
            - old_log_probs_output_only.detach().cpu().numpy()
        )
        argmax_log_prob_diff = int(diff_log_probs.argmax())
        argmin_log_prob_diff = int(diff_log_probs.argmin())
        debug_info = RolloutLossDebugInfo(
            loss=float(loss.detach().cpu()),
            n_tokens=n_tokens,
            n_output_tokens=n_output_tokens,
            truncated=truncated,
            max_log_prob_diff=float(diff_log_probs[argmax_log_prob_diff]),
            min_log_prob_diff=float(diff_log_probs[argmin_log_prob_diff]),
            argmax_log_prob_diff=argmax_log_prob_diff,
            argmin_log_prob_diff=argmin_log_prob_diff,
            log_importance_weight=trajectory_log_importance_weight,
            importance_weight=float(trajectory_importance_weight.detach().cpu()),
            per_token_kl_sum=float(-diff_log_probs.sum()),
            advantage=advantage_estimates.detach().cpu().numpy(),
            ppo=ppo_debug_info,
        )

        return loss, debug_info

    def _surrogate_loss(
        self,
        advantage_estimates: torch.Tensor,
        new_log_probs_output_only: torch.Tensor,
        old_log_probs_output_only: torch.Tensor,
        trajectory_importance_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, PPODebugInfo | None]:
        """Compute the surrogate loss used for gradient estimation for a rollout.

        The gradient of this loss with respect to the adapter parameters will be this
        rollout's contribution to the stochastic gradient estimate.

        Also returns the log importance weight
        """
        # this should have shape [n_output_tokens]
        per_token_importance_weight = torch.exp(
            new_log_probs_output_only - old_log_probs_output_only
        )

        loss_type = LossType(self._cfg.params.loss_type)
        assert is_policy_gradient_loss(loss_type)

        # Not an ideal place for this check
        if self._cfg.params.rloo_kl_lambda < 0:
            raise ValueError("rloo_kl_lambda must be non-negative")

        advantage_estimate_with_kl: torch.Tensor = (
            advantage_estimates + self._cfg.params.rloo_kl_lambda
        )
        if self._cfg.params.do_ppo_clipping:
            if loss_type is LossType.PG_PER_TOKEN:
                loss, ppo_debug_info = self._ppo_loss(
                    advantage_estimate_with_kl, per_token_importance_weight
                )
                loss = loss.mean()
            elif loss_type is LossType.PG_PER_TRAJECTORY:
                loss, ppo_debug_info = self._ppo_loss(
                    advantage_estimate_with_kl, trajectory_importance_weight
                )
            else:
                raise ValueError(f"Unsupported {loss_type=}")
        else:  # REINFORCE (no importance weights)
            if loss_type is LossType.PG_PER_TOKEN:
                loss = -(advantage_estimate_with_kl * new_log_probs_output_only).mean()
            elif loss_type is LossType.PG_PER_TRAJECTORY:  # Vanilla RLOO
                # loss = -advantage_estimate_with_kl * new_log_prob
                # Mean works better than sum in practice
                loss = -advantage_estimate_with_kl * new_log_probs_output_only.mean()
            else:
                raise ValueError(f"Unsupported {loss_type=}")
            ppo_debug_info = None

        return loss, ppo_debug_info

    def _check_init_log_probs_difference(
        self, infos: list[RolloutLossDebugInfo], ids: list[RolloutID]
    ) -> None:
        """Check that at gradient step 0, the models used for rollout generation and for gradient computation agree on log probabilities."""
        if self._global_step == 0:
            max_allowed_abs_diff = self._cfg.get("abs_log_prob_diff_threshold", float("inf"))
            failures = []
            for id, info in zip(ids, infos, strict=False):
                if max(info.max_log_prob_diff, -info.min_log_prob_diff) > max_allowed_abs_diff:
                    failures.append((id, info))
            if failures:
                raise ValueError(
                    f"Maximum per-token abs log probability difference at global step 0 ({max_allowed_abs_diff}) was exceeded on one or more rollouts: {failures}"
                )

    def _get_grad_norm(self, grads: Iterable[torch.Tensor], norm_type: float) -> torch.Tensor:
        """
        Return the gradient norm of parameters ``param`` s, where the gradients are viewed as a single vector.

        The returned norm is in FP32 even if parameters/gradients are in a low precision. This is because the downstream
        use of this return value is a reduction across ranks.
        """
        # Compute the gradient norm in FP32, where we treat the gradients as a
        # single vector
        grad_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                    for grad in grads
                ]
            ),
            norm_type,
            dtype=torch.float32,
        )
        return grad_norm.to(device=self._device)

    @torch.no_grad()  # type: ignore
    def clip_grad_norm_fsdp_(
        self,
        parameters: torch.Tensor | Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """
        Clip the gradient norm of an iterable of parameters.

        Gradient norm clipping requires computing the gradient norm over the entire model.
        `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
        We need to manually reduce the gradient norm across PP stages.
        See https://github.com/pytorch/torchtitan/issues/596 for details.

        Args:
            parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).

        """
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = self._get_grad_norm(grads, norm_type)

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        from torch.distributed.tensor._api import DTensor

        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.to(self._device).full_tensor()

        if math.isinf(norm_type):
            total_norm = _distributed_all_reduce(total_norm, dist.ReduceOp.MAX)
        else:
            total_norm **= norm_type
            total_norm = _distributed_all_reduce(total_norm, dist.ReduceOp.SUM)
            total_norm **= 1.0 / norm_type

        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        grads_device = next(iter(grads)).device
        clip_coef_clamped_device = clip_coef_clamped.to(grads_device)
        for g in grads:
            g.mul_(clip_coef_clamped_device)

        return total_norm

    def _maybe_optimizer_step(
        self,
        model: ModelType,
        optimizer: Optimizer,
    ) -> tuple[float, bool, float]:
        """Here in addition to clipping we (optionally) entirely reject gradients
        that exceed the max norm, or their norm is at least a 5 sigma outlier.
        Returns pre-clipping grad norm, whether the optimizer stepped, and the
        post-step trainable-parameter update L2 norm.
        """
        if self._cfg.fsdp:
            grad_norm_tensor = self.clip_grad_norm_fsdp_(
                model.parameters(), max_norm=self._cfg.params.max_grad_norm
            )
        else:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self._cfg.params.max_grad_norm
            )

        grad_norm = float(grad_norm_tensor.item())

        is_outlier_grad = not self._grad_rms.should_update(
            grad_norm, max_norm=self._cfg.params.max_grad_norm
        )
        self._n_outlier_grads += is_outlier_grad

        if self._cfg.params.skip_outlier_grads and is_outlier_grad:
            logger.warning(
                f"rank{self._rank}: Skipped gradient update due to large {grad_norm=}, {self._n_outlier_grads=}"
            )
            optimizer_stepped = False
            parameter_update_l2_norm = 0.0
        else:
            parameter_snapshots = _snapshot_trainable_parameters(model)
            optimizer.step()
            parameter_update_l2_norm = _parameter_update_l2_norm(parameter_snapshots)
            optimizer_stepped = True
            self._grad_rms.update(grad_norm)

        return grad_norm, optimizer_stepped, parameter_update_l2_norm

    def _gradient_step(
        self,
        model: ModelType,
        optimizer: Optimizer,
        rollouts: list[TrainingRollout],
        monte_carlo_advantages: np.ndarray,
        rollout_ids: list[RolloutID],
        loss_div_factor: int,
    ) -> bool:
        """Returns True when the training iteration needs to be interrupted."""
        # Sub sample memory profiling so we don't slow down training
        sample_memory_profile = torch.rand(1, device=self._device) < 0.05
        dist.broadcast(sample_memory_profile, src=0)

        with profile("compute_gradients"):
            loss_debug_infos: list[RolloutLossDebugInfo] = []
            for i, (rollout, id) in enumerate(zip(rollouts, rollout_ids, strict=False)):
                with NVMLPeakMemProfiler("_loss", logger=self._rank0_logger):
                    loss, loss_debug_info = self._loss(
                        model, rollout, monte_carlo_advantages[i], id
                    )

                if torch.isnan(loss).item():
                    logger.warning(f"rank{self._rank}: {loss=}. Skipping the iteration!")

                with NVMLPeakMemProfiler("model_backward", logger=self._rank0_logger):
                    (loss / loss_div_factor).backward()

                loss_debug_infos.append(loss_debug_info)

        avg_loss = np.array([info.loss for info in loss_debug_infos]).mean()
        log_importance_weights = np.array([info.log_importance_weight for info in loss_debug_infos])

        if dist.is_initialized():
            avg_losses_tensor = torch.tensor(avg_loss, device=self._device)
            dist.reduce(avg_losses_tensor, dst=0, op=dist.ReduceOp.AVG)
            avg_loss = avg_losses_tensor.item()

            log_iw_tensor = torch.from_numpy(log_importance_weights).to(self._device)
            gather_list = [torch.empty_like(log_iw_tensor) for _ in range(self._world_size)]
            dist.all_gather(gather_list, log_iw_tensor)
            all_log_importance_weights = torch.cat(gather_list, dim=0).cpu().numpy()
        else:
            all_log_importance_weights = log_importance_weights

        local_metric_totals = torch.tensor(
            [
                sum(info.per_token_kl_sum for info in loss_debug_infos),
                sum(info.n_output_tokens for info in loss_debug_infos),
                sum(
                    info.ppo.clipped_observations
                    for info in loss_debug_infos
                    if info.ppo is not None
                ),
                sum(
                    info.ppo.total_observations for info in loss_debug_infos if info.ppo is not None
                ),
            ],
            dtype=torch.float64,
            device=self._device,
        )
        if dist.is_initialized():
            dist.all_reduce(local_metric_totals, op=dist.ReduceOp.SUM)
        if self._iteration_metrics is not None:
            self._iteration_metrics.record_gradient_observation(
                per_token_kl_sum=float(local_metric_totals[0].item()),
                per_token_kl_observations=int(local_metric_totals[1].item()),
                ppo_clipped_observations=int(local_metric_totals[2].item()),
                ppo_total_observations=int(local_metric_totals[3].item()),
            )

        invalid_loss = math.isnan(avg_loss)

        all_kl_estimate = kl_estimate(all_log_importance_weights)
        high_kl = abs(all_kl_estimate) > self._cfg.params.high_kl_threshold
        if self._rank == 0:
            logger.info(f"{all_kl_estimate=:.3f} {high_kl=}")

        stop_iteration = False
        grad_norm = 0.0

        if invalid_loss:
            self._invalid_steps_skipped += 1
            if self._iteration_metrics is not None:
                self._iteration_metrics.invalid_loss_steps += 1
            logger.error(
                f"rank{self._rank}: Skipping an optimizer step! {self._invalid_steps_skipped}"
            )
            stop_iteration = True
            if self._invalid_steps_skipped > 20:
                raise RuntimeError(
                    f"Interrupt training after skipping {self._invalid_steps_skipped} iterations"
                )
        elif high_kl:
            self._high_kl_events += 1
            if self._iteration_metrics is not None:
                self._iteration_metrics.high_kl_events += 1
            logger.warning(
                f"rank{self._rank}: Stopping the training iteration early because of {high_kl=}, {all_kl_estimate=}"
            )
            stop_iteration = True

        else:
            if self._cfg.fsdp:
                with (
                    profile("optimizer_step"),
                    NVMLPeakMemProfiler("optimizer_step", logger=self._rank0_logger),
                ):
                    (
                        grad_norm,
                        optimizer_stepped,
                        parameter_update_l2_norm,
                    ) = self._maybe_optimizer_step(model, optimizer)
                if self._iteration_metrics is not None:
                    self._iteration_metrics.record_optimizer_result(
                        grad_norm=grad_norm,
                        optimizer_stepped=optimizer_stepped,
                        parameter_update_l2_norm=parameter_update_l2_norm,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                    )
            else:
                raise AssertionError(f"{self._cfg.fsdp=}")

        optimizer.zero_grad()

        self._check_init_log_probs_difference(loss_debug_infos, rollout_ids)

        # log stuff
        if self._with_wandb:
            if self._cfg.params.do_ppo_clipping and all(
                info.ppo is not None for info in loss_debug_infos
            ):
                clipped_fractions = np.array(
                    [cast(PPODebugInfo, info.ppo).clipped_fraction for info in loss_debug_infos]
                )
            else:
                clipped_fractions = np.zeros(len(loss_debug_infos))
            wandb.log(
                dict(
                    global_step=self._global_step,
                    kl_estimate=all_kl_estimate,
                    ppo_clipped_fraction=np.mean(clipped_fractions),
                    high_kl=float(high_kl),
                    n_high_kl=float(self._high_kl_events),
                    n_invalid_loss=float(self._invalid_steps_skipped),
                    grad_norm=float(grad_norm),
                    grad_norm_mean=self._grad_rms.mean,
                    grad_norm_std=self._grad_rms._std_dev(),
                    n_outlier_grads=float(self._n_outlier_grads),
                    avg_loss=float(avg_loss),
                )
            )

        return stop_iteration

    def log_rollout_metrics(
        self,
        rollouts: Sequence[TrainingRollout],
        prefix: str,
        log_highest_return: bool,
        log_html: bool = False,
    ) -> None:
        """Log rollout metrics to wandb.

        This function has domain-specific dependencies, which isn't ideal given that this class is
        otherwise generic to any domain. That said, we'll tolerate it in favor of moving fast.
        """
        if type(rollouts[0]) is TrainingRollout:
            return

        # AppWorld logging
        from phi_agents.evals.appworld_rollout_data import (
            AppWorldTrainingRollout,
            log_appworld_rollouts_html,
            log_appworld_rollouts_to_wandb,
        )

        assert isinstance(rollouts[0], AppWorldTrainingRollout)
        appworld_rollouts = [cast(AppWorldTrainingRollout, r) for r in rollouts]
        log_appworld_rollouts_to_wandb(
            self._global_step,
            appworld_rollouts,
            prefix=prefix,
            log_highest_return=log_highest_return,
            iterations_completed=self._iterations_completed,
        )
        if log_html:
            log_appworld_rollouts_html(
                appworld_rollouts,
                self._global_step,
                self._iterations_completed,
                self._get_tokenizer(),
            )

    def _check_sampling_parameters_supported(self) -> None:
        for sampling_method in ("min_p", "top_p", "top_k", "frequency_penalty"):
            if self._llm_cfg.vllm_class[sampling_method] is not None:
                raise ValueError(
                    f"{__file__} does not support {sampling_method=} (cfg: {self._llm_cfg.vllm_class}) "
                    f"due to probability distribution mismatch between PyTorch and vLLM"
                )

    def _recompute_rollout_probs(
        self, model: ModelType, rollouts: list[TrainingRollout], ids: list[RolloutID]
    ) -> dict[str, float]:
        """Recompute log probabilities using PyTorch. Overwrites rollouts.log_probs."""
        logger.info("Recomputing rollout log probabilities...")

        logger.info("switching into eval mode")
        model.eval()
        logger.info("done")

        self._check_sampling_parameters_supported()

        avg_logprob_diffs = []
        max_logprob_diffs = []
        avg_prob_diffs = []
        max_prob_diffs = []
        min_imp_weights = []
        max_imp_weights = []
        p95_prob_diffs = []
        p99_prob_diffs = []

        # note: using torch.inference_mode instead of torch.no_grad here caused error
        with torch.no_grad():
            # with torch.inference_mode():
            for i_, (rollout, id) in enumerate(zip(rollouts, ids, strict=False)):
                with NVMLPeakMemProfiler("recompute_logprobs", logger=self._rank0_logger):
                    # note: this clips the lengths
                    tokens, is_output, old_log_probs, _ = self._get_tensors(rollout, id)
                    n_tokens = len(tokens)

                    _, new_log_probs_all = self._log_probs(model, tokens, is_output)
                    assert (
                        n_tokens == len(is_output) == len(old_log_probs) == len(new_log_probs_all)
                    )

                    old_log_probs = old_log_probs[is_output]
                    new_log_probs = new_log_probs_all[is_output]

                    old_probs = torch.exp(old_log_probs)
                    new_probs = torch.exp(new_log_probs)

                    log_prob_diff = (new_log_probs - old_log_probs).abs()
                    probs_diff = (new_probs - old_probs).abs()

                    avg_logprob_diff = log_prob_diff.mean().item()
                    max_logprob_diff = log_prob_diff.max().item()
                    avg_prob_diff = probs_diff.mean().item()
                    max_prob_diff = probs_diff.max().item()
                    p95_prob_diff = torch.quantile(probs_diff, 0.95).item()
                    p99_prob_diff = torch.quantile(probs_diff, 0.99).item()
                    importance_weight = new_probs / old_probs.clamp(min=1e-12)
                    min_imp_weight = importance_weight.min().item()
                    max_imp_weight = importance_weight.max().item()

                    avg_logprob_diffs.append(avg_logprob_diff)
                    max_logprob_diffs.append(max_logprob_diff)
                    avg_prob_diffs.append(avg_prob_diff)
                    max_prob_diffs.append(max_prob_diff)
                    p95_prob_diffs.append(p95_prob_diff)
                    p99_prob_diffs.append(p99_prob_diff)
                    min_imp_weights.append(min_imp_weight)
                    max_imp_weights.append(max_imp_weight)

                    # optional check that original log probs and new log probs are close
                    max_allowed_abs_diff = self._cfg.get(
                        "abs_log_prob_diff_threshold", float("inf")
                    )
                    if max_logprob_diff > max_allowed_abs_diff:
                        raise ValueError(
                            f"Per-token abs log probability difference ({max_logprob_diff}) exceeded allowed limit ({max_allowed_abs_diff}) on rollout {id}"
                        )
                    elif self._rank == 0:
                        logger.info(
                            f"Conversation {i_} len={len(tokens[:-1])}: {avg_prob_diff=:.4f}, "
                            f"{max_prob_diff=:.4f}, {p95_prob_diff=:.4f}, {p99_prob_diff=:.4f}, "
                            f"{min_imp_weight=:.4f}, {max_imp_weight=:.4f}, "
                            f"{avg_logprob_diff=:.4f} {max_logprob_diff=:.4f}"
                        )

                    # overwrite the log_probs
                    rollout.policy_token_info.log_probs = new_log_probs_all.tolist()

                    # because the log_probs maybe resized, we also resize the other fields in policy_token_info
                    rollout.policy_token_info.tokens = rollout.policy_token_info.tokens[:n_tokens]
                    rollout.policy_token_info.is_output = rollout.policy_token_info.is_output[
                        :n_tokens
                    ]

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        stats = {
            "recompute_stats/avg_logprob_diff": np.mean(avg_logprob_diffs).item(),
            "recompute_stats/max_logprob_diff": max(max_logprob_diffs),
            "recompute_stats/avg_prob_diff": np.mean(avg_prob_diffs).item(),
            "recompute_stats/max_prob_diff": max(max_prob_diffs),
            "recompute_stats/max_p95_prob_diff": max(p95_prob_diffs),
            "recompute_stats/max_p99_prob_diff": max(p99_prob_diffs),
            "recompute_stats/min_imp_weight": min(min_imp_weights),
            "recompute_stats/max_imp_weight": max(max_imp_weights),
        }

        if self._rank == 0:
            for key, value in stats.items():
                logger.info(f"{key}: {value:.4f}")

        logger.info("switching into train mode")
        model.train()
        logger.info("done")

        return stats

    def _gradient_steps(
        self,
        model: ModelType,
        optimizer: Optimizer,
        rollouts: list[TrainingRollout],
        monte_carlo_advantages: np.ndarray,
        rollout_ids: list[RolloutID],
    ) -> None:
        logger.info("Starting gradient steps.")
        # do gradient steps
        if self._rank == 0:
            pbar = tqdm(
                desc="Epochs completed for current iteration",
                total=self._cfg.params.epochs_per_iteration,
            )

        if self._cfg.params.strictly_on_policy:
            assert self._cfg.params.epochs_per_iteration == 1
            local_minibatch_size = len(rollouts)
            loss_div_factor = len(rollouts)
        else:
            # scale loss by a constant factor regardless of how many rollouts are left after filtering
            local_minibatch_size = self._cfg.params.minibatch_size // self._world_size
            loss_div_factor = self._cfg.params.minibatch_size

        # Logging, this is repetitive but that is ok for now
        if self._rank == 0:
            logger.info(f"{local_minibatch_size=}")

        stop_iteration = False
        for i_step in range(self._cfg.params.epochs_per_iteration):
            if stop_iteration:
                logger.warning(
                    f"Stopping iteration {self._iterations_completed} early after epoch {i_step=}"
                )
                break

            # shuffle the experience every step
            if i_step > 0:  # already shuffled before the 1st step
                indices = self._rng.permutation(len(rollouts))
                rollouts = [rollouts[i] for i in indices]
                monte_carlo_advantages = monte_carlo_advantages[indices]
                rollout_ids = [rollout_ids[i] for i in indices]

            for mb_idx, i in enumerate(range(0, len(rollouts), local_minibatch_size)):
                # attempt to reduce memory usage
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

                rollouts_minibatch = rollouts[i : i + local_minibatch_size]

                if mb_idx < (len(rollouts) // local_minibatch_size):
                    assert len(rollouts_minibatch) == local_minibatch_size
                else:
                    # allow the last minibatch to be smaller
                    assert len(rollouts_minibatch) < local_minibatch_size
                    assert len(rollouts_minibatch) == len(rollouts) % local_minibatch_size

                num_tokens = len(
                    [
                        token
                        for rollout in rollouts_minibatch
                        for token in rollout.policy_token_info.tokens
                    ]
                )

                with set_profiling_num_elements(num_tokens):
                    stop_iteration |= self._gradient_step(
                        model,
                        optimizer,
                        rollouts_minibatch,
                        monte_carlo_advantages[i : i + local_minibatch_size],
                        rollout_ids[i : i + local_minibatch_size],
                        loss_div_factor,
                    )
                    if stop_iteration:
                        logger.warning(f"Stop epoch {i_step} early after {i + 1} minibatches")
                        break

                    self._global_step += 1

            # Ensure only one grad step was taken
            if self._cfg.params.strictly_on_policy:
                assert i == 0

            if self._rank == 0:
                pbar.update(1)

            self._cloud_checkpointer.on_step_end()  # TODO needed?

    def _stress_test(self, profiler_key: str) -> None:
        assert self._model is not None, f"{self._model=}"
        assert self._optimizer is not None, f"{self._optimizer=}"

        with NVMLPeakMemProfiler(f"{profiler_key}_forward", logger=self._rank0_logger):
            max_seq_len = self._cfg.learning_max_seq_len
            tokens = torch.full((max_seq_len,), 198, dtype=torch.int, device=self._device)
            is_output = torch.zeros((max_seq_len,), dtype=torch.bool, device=self._device)
            is_output[int(max_seq_len // 10) :] = True
            self._rank0_logger.info(
                f"Stress tests with {max_seq_len=} num_outputs={is_output.sum().item()}"
            )
            last_hidden_state, new_log_probs = self._log_probs(self._model, tokens, is_output)

        with NVMLPeakMemProfiler(f"{profiler_key}_loss", logger=self._rank0_logger):
            adv = torch.randn_like(new_log_probs)
            old_log_probs = new_log_probs.detach() + 0.01 * torch.randn_like(new_log_probs)
            ratio = (new_log_probs - old_log_probs).exp()
            test_loss = -(torch.clamp(ratio, 0.8, 1.2) * adv).mean()

        with NVMLPeakMemProfiler(f"{profiler_key}_backward", logger=self._rank0_logger):
            test_loss.backward()

        self._optimizer.zero_grad()

    def _run_stress_test(self) -> None:
        self._rank0_logger.warning(f"Stress test is enabled with {self._cfg.stress_test_iters=} !")

        for stress_test_idx in range(self._cfg.stress_test_iters):
            profiler_key = "initial_stress" if stress_test_idx == 0 else "stress"

            with barrier_guard(before=True, after=True), timeit("stress_test", self._rank0_logger):
                assert self._model is not None, f"{self._model=}"
                self._model.train()

                gc.collect()
                torch.cuda.empty_cache()
                with NVMLPeakMemProfiler(f"{profiler_key}_before", logger=self._rank0_logger):
                    pass
                self._stress_test(profiler_key)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                with NVMLPeakMemProfiler(
                    f"{profiler_key}_after_cleanup", logger=self._rank0_logger
                ):
                    pass

                self._rank0_logger.info(
                    f"CUDA mem allocated {torch.cuda.memory_allocated() / 1024**3:.1f} GB"
                )
                self._rank0_logger.info(
                    f"CUDA mem reserved {torch.cuda.memory_reserved() / 1024**3:.1f} GB"
                )

    def _run(self) -> None:
        with barrier_guard(before=False, after=True):
            self._rank0_logger.info("Starting inference servers...")
            self._rollout_worker.start_servers()

            # check if there is an existing checkpoint, and if so, resume from it
            if (last_checkpoint := get_last_checkpoint(self._cfg.cloud_path)) is not None:
                last_checkpoint_local_path = self._local_path / last_checkpoint.name
                # All local ranks share the same temporary directory. Copy once,
                # then make the completed checkpoint visible to every rank.
                if self._rank == 0:
                    fu.copy(last_checkpoint, last_checkpoint_local_path)
                self._accelerator.wait_for_everyone()
                self._load_trainer_state(last_checkpoint_local_path)
            else:
                last_checkpoint_local_path = None

        if not self._exclusive_inference_and_learning:
            self._rank0_logger.info(
                "Learning and inference fit into memory together, initialize the model while inference servers are starting..."
            )
            with (
                barrier_guard(before=True, after=True),
                timeit(f"rank{self._rank}: setup_model", self._rank0_logger),
                NVMLPeakMemProfiler("_setup_model_optimizer_lr", logger=self._rank0_logger),
            ):
                self._model, self._optimizer, self._lr_scheduler = self._setup_model_optimizer_lr(
                    last_checkpoint_local_path
                )

            self._run_stress_test()

        with barrier_guard(before=False, after=True):
            self._rollout_worker.await_servers()

        with barrier_guard(before=False, after=True):
            if (
                self._local_rank == 0
                and last_checkpoint is not None
                and last_checkpoint_local_path is not None
            ):
                logger.info("Copy the checkpoint to inference workers...")
                run_on_nodes_with_resource(
                    remote_fn=ray.remote(fu.copy),
                    resource_name=VLLM_RESOURCE,
                    run_on_current_node=False,
                    src_uri=last_checkpoint,
                    dst_uri=last_checkpoint_local_path,
                )
                logger.info("Copy the checkpoint to inference workers...Done!")

        # request scenario generation before the start of the 1st iteration
        scenarios_per_gpu_per_iteration = (
            self._cfg.params.scenarios_per_iteration // self._world_size
        )
        self._scenario_sampler.ensure_scenarios_requested(scenarios_per_gpu_per_iteration)

        # init rollout worker and start rollout generation for the 1st iteration
        with barrier_guard(before=True, after=True):
            self._rollout_worker.request_rollout_generation(
                scenarios_per_gpu_per_iteration, fu.lora_path(last_checkpoint_local_path)
            )

        self._rank0_logger.info(
            f"Running for {self._cfg.params.total_iterations=}, {self._cfg.async_rollouts=}"
        )

        for _ in range(self._iterations_completed, self._cfg.params.total_iterations):
            self._callbacks.before_iteration(self._iterations_completed, last_checkpoint_local_path)
            target_iteration = self._target_iteration()
            self._iteration_metrics = IterationMetricAccumulator(
                iteration=target_iteration,
                global_step_start=self._global_step,
            )
            expected_rollouts = self._expected_rollouts_per_iteration()
            self._begin_iteration_manifest(target_iteration, expected_rollouts)
            self._active_iteration_manifest = (target_iteration, expected_rollouts)

            # use most recent adapter checkpoint (or base model if there is no adapter checkpoint) to do rollouts
            with profile("get_rollouts"), timeit(f"rank{self._rank}: get_rollouts", logger):
                # wait until the requested number of rollouts is generated
                scenarios, rollouts = self._rollout_worker.get_rollouts(
                    scenarios_per_gpu_per_iteration,
                    self._cfg.rollouts_fraction,
                    self._cfg.rollouts_per_scenario_fraction,
                    self._world_size,
                    self._device,
                    show_progress=self._rank == 0,
                )

                self._accelerator.wait_for_everyone()

            # load current model and broadcast to all workers
            if self._model is None or self._optimizer is None or self._lr_scheduler is None:
                with profile("setup_model", burn=0), timeit("setup_model", self._rank0_logger):
                    self._model, self._optimizer, self._lr_scheduler = (
                        self._setup_model_optimizer_lr(last_checkpoint_local_path)
                    )

                if self._iterations_completed == 0 or self._cfg.stress_test_on_resume:
                    self._run_stress_test()

            local_rollouts = list(itertools.chain(*rollouts))  # flatten the list of lists
            all_rollout_stats = self._reduce_stats(local_rollouts)
            finished_rollouts = self._global_finished_rollout_count(all_rollout_stats.n_rollouts)
            if finished_rollouts != expected_rollouts:
                exc = RuntimeError(
                    f"Incomplete rollout batch for iteration {target_iteration}: "
                    f"{finished_rollouts=} {expected_rollouts=}"
                )
                exc.finished_rollouts = finished_rollouts  # type: ignore[attr-defined]
                exc.expected_rollouts = expected_rollouts  # type: ignore[attr-defined]
                raise exc
            self._commit_iteration_manifest(target_iteration, expected_rollouts, finished_rollouts)
            self._persist_rollout_diagnostics(rollouts, iteration=target_iteration)
            with profile("recycle_scenario_runners"), timeit("recycle_scenario_runners", logger):
                self._rollout_worker.recycle_scenario_runners()

            if (
                self._cfg.async_rollouts
                and self._iterations_completed + 1 < self._cfg.params.total_iterations
            ):
                self._callbacks.before_new_rollouts()

                # request rollout generation for the next iteration right away
                self._rollout_worker.request_rollout_generation(
                    scenarios_per_gpu_per_iteration, fu.lora_path(last_checkpoint_local_path)
                )

            # keep track of identifying information for each rollout
            local_rollout_ids = list(
                itertools.chain(*get_rollout_ids(rollouts, self._global_step, self._rank))
            )

            # Advantage filtering
            with (
                set_profiling_num_elements(all_rollout_stats.n_output_tokens),
                profile("adv_filtering"),
            ):
                local_adv = self._compute_adv_estimates(rollouts)

                local_rollout_data: RolloutsAdvantagesIDs = (
                    local_rollouts,
                    local_adv,
                    local_rollout_ids,
                )
                # gather all rollouts on worker 0
                rollout_gather_list = [None] * self._world_size if self._rank == 0 else None
                dist.gather_object(local_rollout_data, rollout_gather_list, dst=0)

                if self._rank == 0:
                    all_rollouts = []
                    all_adv_estimates = []
                    all_ids = []
                    worker_rollout_data = cast(list[RolloutsAdvantagesIDs], rollout_gather_list)
                    for ith_worker_rollouts, ith_worker_adv, ith_worker_ids in worker_rollout_data:
                        all_rollouts.extend(ith_worker_rollouts)
                        all_adv_estimates.append(ith_worker_adv)
                        all_ids.extend(ith_worker_ids)
                    all_adv_estimates = np.concatenate(all_adv_estimates)

                    (
                        all_rollouts_and_adv_and_ids,
                        adv_filtered_fraction,
                        empirical_adv_filter_threshold,
                        advantage_filter_stats,
                    ) = self._filter_rollouts(all_rollouts, all_adv_estimates, all_ids)
                    self._iteration_metrics.advantage_filter = advantage_filter_stats
                    self._iteration_metrics.sampling_entropy = sampled_token_entropy_stats(
                        all_rollouts
                    )
                else:
                    all_rollouts_and_adv_and_ids = None

                local_rollouts_and_adv_and_ids = [None]
                dist.scatter_object_list(
                    local_rollouts_and_adv_and_ids, all_rollouts_and_adv_and_ids, src=0
                )
                local_rollouts, local_adv, local_rollout_ids = local_rollouts_and_adv_and_ids[0]  # type: ignore

            if self._with_wandb:
                assert all_rollouts_and_adv_and_ids is not None
                filtered_rollouts: list[TrainingRollout] = list(
                    itertools.chain(*[ra[0] for ra in all_rollouts_and_adv_and_ids])
                )
                self.log_rollout_metrics(
                    all_rollouts, prefix="all_rollouts", log_highest_return=True
                )
                self.log_rollout_metrics(
                    filtered_rollouts,
                    prefix="filtered_rollouts",
                    log_highest_return=False,
                    log_html=True,
                )

            # compute old probabilities using the current model.
            recompute_stats = dict()
            if self._cfg.recompute_rollout_probs:
                with (
                    set_profiling_num_elements(
                        all_rollout_stats.n_output_tokens * self._cfg.params.epochs_per_iteration
                    ),
                    profile("recompute_probs"),
                ):
                    recompute_stats = self._recompute_rollout_probs(
                        self._model, local_rollouts, local_rollout_ids
                    )

            # do gradient steps using rollouts
            with (
                set_profiling_num_elements(
                    all_rollout_stats.n_output_tokens * self._cfg.params.epochs_per_iteration
                ),
                profile("gradient_steps"),
            ):
                self._gradient_steps(
                    self._model,
                    self._optimizer,
                    local_rollouts,
                    local_adv,
                    local_rollout_ids,
                )

            # Step the scheduler once per RL iteration instead of once per SGD step.
            # We need to know the total number of steps to define the schedule, and the total
            # number of SGD steps is unknown due to adv. filtering, while the total number of RL
            # iterations is currently predetermined.
            self._iteration_metrics.learning_rate_before_scheduler = float(
                self._lr_scheduler.get_last_lr()[0]
            )
            self._lr_scheduler.step()
            self._iteration_metrics.learning_rate_after_scheduler = float(
                self._lr_scheduler.get_last_lr()[0]
            )

            self._iterations_completed += 1
            n_rollouts = all_rollout_stats.n_rollouts
            self._rollouts_generated += n_rollouts
            self._output_tokens_generated += all_rollout_stats.n_output_tokens
            self._speedometer.track(
                rollouts=self._rollouts_generated, output_tokens=self._output_tokens_generated
            )

            if self._with_wandb:
                wandb.log(
                    dict(
                        global_step=self._global_step,
                        iterations_completed=self._iterations_completed,
                        total_output_tokens=self._output_tokens_generated,
                        total_rollouts=self._rollouts_generated,
                        avg_output_tokens=all_rollout_stats.n_output_tokens / n_rollouts,
                        avg_return=all_rollout_stats.total_return / n_rollouts,
                        adv_filtered_fraction=adv_filtered_fraction,
                        empirical_adv_filter_threshold=empirical_adv_filter_threshold,
                        last_lr=self._lr_scheduler.get_last_lr()[0],
                        **profiling_summary(),
                        **profiling_memory_summary(),
                        **self._speedometer.summary("system_throughput/"),
                        **recompute_stats,
                    ),
                )

            # save a checkpoint
            last_checkpoint_local_path = self._local_path / checkpoint_name(
                self._iterations_completed
            )
            self._save_lora_checkpoint(last_checkpoint_local_path)
            self._cloud_checkpointer.on_save()
            self._active_iteration_manifest = None
            self._persist_iteration_metrics(status="completed")

            if self._exclusive_inference_and_learning:
                # inference and learning don't fit in memory together; free up CUDA memory:
                with barrier_guard():
                    self._model.to("meta")

                    del self._optimizer
                    del self._lr_scheduler
                    del self._model
                    self._accelerator.free_memory()

                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                torch_cuda_mem_bytes = torch.cuda.memory_reserved(self._device)
                logger.info(f"torch_cuda_mem_bytes: {torch_cuda_mem_bytes}")
                if torch_cuda_mem_bytes >= 20e9:
                    raise AssertionError(f"torch still using {torch_cuda_mem_bytes} bytes")
                self._model = self._optimizer = None

            if not self._cfg.async_rollouts:
                self._callbacks.before_new_rollouts()

                # start new rollout generation once we finished learning
                self._rollout_worker.request_rollout_generation(
                    scenarios_per_gpu_per_iteration, fu.lora_path(last_checkpoint_local_path)
                )

            if self._rank == 0:
                logger.info(
                    f"Finished training iterations: {self._iterations_completed}, global step: {self._global_step}"
                )
                self._speedometer.log_summary(logger)

            self._callbacks.after_iteration(self._iterations_completed)
            self._iteration_metrics = None

    def run(self) -> None:
        try:
            self._run()
        except KeyboardInterrupt as exc:
            self._abort_active_iteration_manifest(exc)
            self._persist_failed_iteration_metrics(exc, failure_event="training_failed")
            _log_training_failure(exc)
            raise
        except torch.cuda.OutOfMemoryError as exc:
            self._abort_active_iteration_manifest(exc)
            self._persist_failed_iteration_metrics(exc, failure_event="cuda_oom")
            _log_training_failure(exc)
            raise
        except Exception as exc:
            self._abort_active_iteration_manifest(exc)
            self._persist_failed_iteration_metrics(exc, failure_event="training_failed")
            _log_training_failure(exc)
            raise
        finally:
            logger.info("Finished!")
            self._scenario_sampler.stop()
            self._rollout_worker.stop()
            self._cloud_checkpointer.stop()
            self._file_upload_executor.shutdown()
            self._callbacks.shutdown()
            ray.shutdown()
            logger.info("Stopped all components!")


def main() -> int:
    logger.info("Starting training with %s CLI override(s)", len(sys.argv) - 1)
    _cfg = get_config(mode="train", overrides=sys.argv[1:])

    import pynvml

    pynvml.nvmlInit()

    from accelerate.utils import InitProcessGroupKwargs

    init_proc = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[init_proc])

    setup_mixed_precision_policy(accelerator)
    setup_cpu_offload_policy(accelerator)

    if accelerator.is_main_process:
        logger.info(OmegaConf.to_yaml(_cfg))

    local_rank = accelerator.local_process_index
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if accelerator.device.type == "cuda":
        torch.cuda.set_device(accelerator.device)

    connect_ray_cluster(rank=rank, barrier=torch_dist_barrier)

    status = 0
    try:
        trainer = RLOOTrainer(accelerator, _cfg, local_rank, rank, world_size)
        trainer.run()
    except KeyboardInterrupt as exc:
        _log_training_failure(exc)
        raise
    except Exception as exc:
        _log_training_failure(exc)
        status = 1
    finally:
        accelerator.end_training()
        pynvml.nvmlShutdown()

    return status


if __name__ == "__main__":
    sys.exit(main())
