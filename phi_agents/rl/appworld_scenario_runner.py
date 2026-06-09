#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import cattrs
import numpy as np
import psutil
import yaml
from omegaconf import DictConfig, OmegaConf

from phi_agents.appworld.interface import AppWorldInterface, load_task_ids
from phi_agents.evals.appworld_evals import run_vllm_inference_single_server_single_task
from phi_agents.evals.appworld_rollout_data import AppWorldRolloutData, AppWorldTrainingRollout
from phi_agents.inference.config import AppWorldConfig
from phi_agents.rl.type_defs import (
    PolicyMessage,
    PolicyTokenInfo,
    Scenario,
    ScenarioRunner,
    TrainingRollout,
)
from phi_agents.utils.logger import get_phi_logger

if TYPE_CHECKING:
    from phi_agents.rl.llm import TrainableLLM


logger = get_phi_logger()


@dataclass
class AppWorldServerHealthConfig:
    enabled: bool = False
    recycle_rss_gb: float = 25.0
    draining_rss_gb: float = 35.0
    soft_dead_rss_gb: float = 45.0
    hard_dead_rss_gb: float = 65.0
    host_memory_hard_dead_percent: float = 85.0
    max_rollout_retries: int = 2
    max_soft_dead_per_iteration: int = 8
    check_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not (
            0
            < self.recycle_rss_gb
            < self.draining_rss_gb
            < self.soft_dead_rss_gb
            < self.hard_dead_rss_gb
        ):
            raise ValueError(
                "Expected 0 < recycle_rss_gb < draining_rss_gb < "
                "soft_dead_rss_gb < hard_dead_rss_gb, got "
                f"{self.recycle_rss_gb=}, {self.draining_rss_gb=}, "
                f"{self.soft_dead_rss_gb=}, {self.hard_dead_rss_gb=}"
            )
        if not (0 < self.host_memory_hard_dead_percent <= 100):
            raise ValueError(
                f"{self.host_memory_hard_dead_percent=} must be in (0, 100]"
            )
        if self.max_rollout_retries < 0:
            raise ValueError(f"{self.max_rollout_retries=} must be non-negative")
        if self.max_soft_dead_per_iteration < 0:
            raise ValueError(
                f"{self.max_soft_dead_per_iteration=} must be non-negative"
            )
        if self.check_interval_seconds <= 0:
            raise ValueError(f"{self.check_interval_seconds=} must be positive")


@dataclass
class AppWorldServerHealthSnapshot:
    status: str
    rss_gb: float
    host_memory_used_percent: float
    server_pid: int | None
    server_port: int
    task_id: str | None


class AppWorldServerHealthError(RuntimeError):
    is_soft_dead = False
    is_hard_dead = False

    def __init__(self, snapshot: AppWorldServerHealthSnapshot, *, reason: str):
        self.reason = reason
        self.server_pid = snapshot.server_pid
        self.server_port = snapshot.server_port
        self.task_id = snapshot.task_id
        self.rss_gb = snapshot.rss_gb
        self.host_memory_used_percent = snapshot.host_memory_used_percent
        super().__init__(
            f"{self.reason}: rss_gb={snapshot.rss_gb:.2f}, "
            f"host_memory_used_percent={snapshot.host_memory_used_percent:.1f}, "
            f"server_pid={snapshot.server_pid}, server_port={snapshot.server_port}, "
            f"task_id={snapshot.task_id}"
        )


class AppWorldServerSoftDeadError(AppWorldServerHealthError):
    is_soft_dead = True

    def __init__(self, snapshot: AppWorldServerHealthSnapshot):
        super().__init__(snapshot, reason="appworld_server_rss_45gb_soft_dead")


class AppWorldServerDeadError(AppWorldServerHealthError):
    is_hard_dead = True

    def __init__(self, snapshot: AppWorldServerHealthSnapshot):
        reason = (
            "host_memory_used_85pct"
            if snapshot.host_memory_used_percent >= 85.0
            else "appworld_server_rss_65gb"
        )
        super().__init__(snapshot, reason=reason)


def _process_tree_rss_gb(pid: int | None) -> float:
    if pid is None or pid <= 0:
        return 0.0
    try:
        proc = psutil.Process(pid)
    except psutil.Error:
        return 0.0

    rss = 0
    processes = [proc]
    try:
        processes.extend(proc.children(recursive=True))
    except psutil.Error:
        pass

    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.Error:
            continue
    return rss / 1e9


def _host_memory_used_percent() -> float:
    return float(psutil.virtual_memory().percent)


@dataclass
class AppWorldScenario(Scenario):
    task_id: str
    dataset_name: str

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Scenario:
        return cast(AppWorldScenario, cattrs.structure(yaml.safe_load(yaml_str), cls))


class AppWorldScenarioSampler(Iterator[AppWorldScenario]):
    """Sampler for AppWorld scenarios (tasks)."""

    def __init__(
        self,
        dataset_name: str,
        seed: int | None = None,
        task_id: str | None = None,
    ):
        if task_id:
            self.all_task_ids = [task_id]
        else:
            self.all_task_ids = load_task_ids(dataset_name)
        self.rng = np.random.default_rng(seed)
        self.task_ids: list[str] = []
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[AppWorldScenario]:
        return self

    def __next__(self) -> AppWorldScenario:
        if len(self.task_ids) == 0:
            self.task_ids = copy.copy(self.all_task_ids)
            self.rng.shuffle(self.task_ids)
        task_id = str(self.task_ids.pop())
        return AppWorldScenario(task_id=task_id, dataset_name=self.dataset_name)


class MixtureAppWorldScenarioSampler(Iterator[AppWorldScenario]):
    """Mixture of two samplers for AppWorld scenarios (tasks)."""

    def __init__(self, samplers: Sequence[Iterator[AppWorldScenario]], probs: Sequence[float]):
        self.samplers = samplers
        self.probs = probs
        if not np.isclose(sum(probs), 1):
            raise ValueError(f"Probabilities do not sum to 1: {probs}")
        if len(samplers) != len(probs):
            raise ValueError(
                f"Number of samplers does not match number of probabilites {len(samplers)} != {len(probs)}"
            )
        self.rng = np.random.default_rng()

    def __iter__(self) -> Iterator[AppWorldScenario]:
        return self

    def __next__(self) -> AppWorldScenario:
        sampler = cast(Iterator[AppWorldScenario], self.rng.choice(self.samplers, p=self.probs))
        return next(sampler)


class AppWorldScenarioRunner(ScenarioRunner):
    """Runs AppWorld using a VLLM server for LLM generation."""

    def __init__(
        self,
        *,
        appworld_config: AppWorldConfig | dict[str, Any],
        server_health: AppWorldServerHealthConfig | dict[str, Any] | None = None,
    ):
        self.appworld_config = (
            cast(AppWorldConfig, cattrs.structure(appworld_config, AppWorldConfig))
            if isinstance(appworld_config, dict)
            else appworld_config
        )
        if server_health is None:
            self.server_health = AppWorldServerHealthConfig()
        elif isinstance(server_health, AppWorldServerHealthConfig):
            self.server_health = server_health
        else:
            server_health_data = (
                OmegaConf.to_container(server_health, resolve=True)
                if isinstance(server_health, DictConfig)
                else server_health
            )
            self.server_health = cast(
                AppWorldServerHealthConfig,
                cattrs.structure(server_health_data, AppWorldServerHealthConfig),
            )

        self.world = AppWorldInterface(stdout_to_devnull=True)
        self.lock = threading.Lock()
        self._restart_after_current_task = False

    def _health_snapshot(self, task_id: str | None = None) -> AppWorldServerHealthSnapshot:
        rss_gb = _process_tree_rss_gb(self.world.server_pid)
        host_memory_used_percent = _host_memory_used_percent()
        if host_memory_used_percent >= self.server_health.host_memory_hard_dead_percent:
            status = "hard_dead"
        elif rss_gb >= self.server_health.hard_dead_rss_gb:
            status = "hard_dead"
        elif rss_gb >= self.server_health.soft_dead_rss_gb:
            status = "soft_dead"
        elif rss_gb >= self.server_health.draining_rss_gb:
            status = "draining"
        elif rss_gb >= self.server_health.recycle_rss_gb:
            status = "recycle_after_task"
        else:
            status = "healthy"
        return AppWorldServerHealthSnapshot(
            status=status,
            rss_gb=rss_gb,
            host_memory_used_percent=host_memory_used_percent,
            server_pid=self.world.server_pid,
            server_port=self.world.port,
            task_id=task_id,
        )

    def _raise_if_hard_dead(self, snapshot: AppWorldServerHealthSnapshot) -> None:
        if snapshot.status == "hard_dead":
            logger.error(
                "AppWorld server hit hard-dead health policy: "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                f"port={snapshot.server_port}, task_id={snapshot.task_id}, "
                f"host_memory_used_percent={snapshot.host_memory_used_percent:.1f}"
            )
            raise AppWorldServerDeadError(snapshot)

    def _raise_if_soft_dead(self, snapshot: AppWorldServerHealthSnapshot) -> None:
        if snapshot.status == "soft_dead":
            logger.error(
                "AppWorld server hit soft-dead RSS policy: "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                f"port={snapshot.server_port}, task_id={snapshot.task_id}"
            )
            self.world.force_close_server()
            raise AppWorldServerSoftDeadError(snapshot)

    def _restart_server(self, reason: str) -> None:
        logger.warning(f"Restarting AppWorld server: {reason}")
        if self.world.clean:
            self.world.ensure_server()
        else:
            self.world.restart()
        self._restart_after_current_task = False

    def _check_health_before_task(self, task_id: str) -> None:
        if not self.server_health.enabled:
            return
        if self._restart_after_current_task:
            self._restart_server("server was marked for recycle after previous task")

        snapshot = self._health_snapshot(task_id)
        self._raise_if_hard_dead(snapshot)
        if snapshot.status == "soft_dead":
            self.world.force_close_server()
            self._restart_server(
                "server was soft-dead before accepting a new task; no rollout discarded"
            )
            return
        if snapshot.status in {"recycle_after_task", "draining"}:
            self._restart_server(
                "server was above recycle/draining threshold before accepting a new task"
            )
            return

    def _mark_restart_after_task(self, snapshot: AppWorldServerHealthSnapshot) -> None:
        if snapshot.status == "draining":
            logger.warning(
                "AppWorld server entering draining state after current task; "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, port={snapshot.server_port}"
            )
            self._restart_after_current_task = True
        elif snapshot.status == "recycle_after_task":
            logger.warning(
                "AppWorld server will recycle after current task: "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, port={snapshot.server_port}"
            )
            self._restart_after_current_task = True

    def _check_health_after_task(self, task_id: str) -> None:
        if not self.server_health.enabled:
            return
        snapshot = self._health_snapshot(task_id)
        self._raise_if_hard_dead(snapshot)
        self._raise_if_soft_dead(snapshot)
        self._mark_restart_after_task(snapshot)

    def _start_dead_monitor(
        self, task_id: str, stop_event: threading.Event
    ) -> tuple[threading.Thread | None, list[AppWorldServerHealthSnapshot], list[AppWorldServerHealthSnapshot]]:
        soft_dead_snapshots: list[AppWorldServerHealthSnapshot] = []
        hard_dead_snapshots: list[AppWorldServerHealthSnapshot] = []
        if not self.server_health.enabled:
            return None, soft_dead_snapshots, hard_dead_snapshots

        def _monitor() -> None:
            while not stop_event.wait(self.server_health.check_interval_seconds):
                snapshot = self._health_snapshot(task_id)
                if snapshot.status == "hard_dead":
                    hard_dead_snapshots.append(snapshot)
                    logger.error(
                        "AppWorld server hit hard-dead threshold during task; force closing. "
                        f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                        f"port={snapshot.server_port}, task_id={task_id}, "
                        f"host_memory_used_percent={snapshot.host_memory_used_percent:.1f}"
                    )
                    self.world.force_close_server()
                    return
                if snapshot.status == "soft_dead":
                    soft_dead_snapshots.append(snapshot)
                    logger.error(
                        "AppWorld server hit soft-dead threshold during task; "
                        "force closing and discarding rollout. "
                        f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                        f"port={snapshot.server_port}, task_id={task_id}"
                    )
                    self.world.force_close_server()
                    return
                if snapshot.status in {"recycle_after_task", "draining"}:
                    logger.warning(
                        "AppWorld server RSS monitor warning: "
                        f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                        f"port={snapshot.server_port}, task_id={task_id}, "
                        f"status={snapshot.status}"
                    )

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        return thread, soft_dead_snapshots, hard_dead_snapshots

    def run(self, scenario: Scenario, llm: TrainableLLM) -> TrainingRollout:
        assert isinstance(scenario, AppWorldScenario)
        task_id = scenario.task_id

        self.world.ensure_server()
        assert self.world.server is not None
        self._check_health_before_task(task_id)

        experiment_name = self.appworld_config.experiment_name or f"runner_{self.world.port:06}"
        if self.appworld_config.experiment_name is not None:
            assert (
                self.appworld_config.agent["mode"] == "eval"
            ), "Because multiple runners could be writing to the same directory, we want to make sure this is only used during eval"

        with self.lock:
            logger.info(f"Acquired lock for AppWorld at {self.world.remote_environment_url}")
            logger.info(f"Generating episode; experiment_name={experiment_name}, task_id={task_id}")
            start = time.perf_counter()
            monitor_stop_event = threading.Event()
            monitor_thread, soft_dead_snapshots, hard_dead_snapshots = (
                self._start_dead_monitor(task_id, monitor_stop_event)
            )
            try:
                episode = run_vllm_inference_single_server_single_task(
                    world=self.world,
                    task_id=task_id,
                    experiment_name=experiment_name,
                    appworld_config=self.appworld_config,
                    llm=llm,
                    with_evaluation=True,
                )
            except Exception as exc:
                if hard_dead_snapshots:
                    raise AppWorldServerDeadError(hard_dead_snapshots[-1]) from exc
                if soft_dead_snapshots:
                    raise AppWorldServerSoftDeadError(soft_dead_snapshots[-1]) from exc
                raise
            finally:
                monitor_stop_event.set()
                if monitor_thread is not None:
                    monitor_thread.join(timeout=1.0)

        if hard_dead_snapshots:
            raise AppWorldServerDeadError(hard_dead_snapshots[-1])
        if soft_dead_snapshots:
            raise AppWorldServerSoftDeadError(soft_dead_snapshots[-1])

        assert episode.eval_result is not None
        eval_result = episode.eval_result

        if self.appworld_config.env.sparse_reward:
            ret = float(eval_result.success)
        else:
            no_code_found_penalty = self.appworld_config.env.no_code_found_penalty
            execution_failed_penalty = self.appworld_config.env.execution_failed_penalty
            ret = (
                len(eval_result.passes) / eval_result.num_tests  # Percentage of unit tests passed
                - execution_failed_penalty * episode.n_execution_failed  # Execution code failures
                - no_code_found_penalty * episode.n_no_code_found  # Missing code blocks
            )
            ret = float(np.clip(ret, a_min=0.0, a_max=None))  # Return must be >= 0

        messages = episode.chat_history

        elapsed = time.perf_counter() - start
        n_policy_messages = sum(isinstance(msg, PolicyMessage) for msg in messages)
        logger.info(
            f"Episode completed (elapsed={elapsed:.2f} seconds, ret={ret:.3f}, {n_policy_messages=})"
        )
        self._check_health_after_task(task_id)

        if sum(msg.ipython for msg in messages if isinstance(msg, PolicyMessage)) > 0:
            logger.warning(f"got <|python_tag|> in rollout: {messages}")

        appworld_rollout_data = AppWorldRolloutData.from_episode(episode, scenario.dataset_name)

        return AppWorldTrainingRollout(
            messages,
            ret,
            elapsed,
            episode.cancelled,
            PolicyTokenInfo() if episode.cancelled else llm.get_policy_token_info(messages),
            appworld_rollout_data=appworld_rollout_data,
        )

    def cleanup(self) -> None:
        if not self.world.clean:
            self.world.close_server()

    def recycle_server(self) -> None:
        with self.lock:
            self._restart_server("end of committed iteration")
