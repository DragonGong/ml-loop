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
    warning_rss_gb: float = 30.0
    draining_rss_gb: float = 45.0
    dead_rss_gb: float = 55.0
    check_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not (
            0
            < self.warning_rss_gb
            < self.draining_rss_gb
            < self.dead_rss_gb
        ):
            raise ValueError(
                "Expected 0 < warning_rss_gb < draining_rss_gb < dead_rss_gb, got "
                f"{self.warning_rss_gb=}, {self.draining_rss_gb=}, {self.dead_rss_gb=}"
            )
        if self.check_interval_seconds <= 0:
            raise ValueError(f"{self.check_interval_seconds=} must be positive")


@dataclass
class AppWorldServerHealthSnapshot:
    status: str
    rss_gb: float
    server_pid: int | None
    server_port: int
    task_id: str | None


class AppWorldServerDeadError(RuntimeError):
    def __init__(self, snapshot: AppWorldServerHealthSnapshot):
        self.reason = "appworld_server_rss_55gb"
        self.server_pid = snapshot.server_pid
        self.server_port = snapshot.server_port
        self.task_id = snapshot.task_id
        self.rss_gb = snapshot.rss_gb
        super().__init__(
            f"{self.reason}: rss_gb={snapshot.rss_gb:.2f}, "
            f"server_pid={snapshot.server_pid}, server_port={snapshot.server_port}, "
            f"task_id={snapshot.task_id}"
        )


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
        self._draining_after_current_task = False

    def _health_snapshot(self, task_id: str | None = None) -> AppWorldServerHealthSnapshot:
        rss_gb = _process_tree_rss_gb(self.world.server_pid)
        if rss_gb >= self.server_health.dead_rss_gb:
            status = "dead"
        elif rss_gb >= self.server_health.draining_rss_gb:
            status = "draining"
        else:
            status = "healthy"
        return AppWorldServerHealthSnapshot(
            status=status,
            rss_gb=rss_gb,
            server_pid=self.world.server_pid,
            server_port=self.world.port,
            task_id=task_id,
        )

    def _raise_if_dead(self, snapshot: AppWorldServerHealthSnapshot) -> None:
        if snapshot.status == "dead":
            logger.error(
                "AppWorld server is dead by RSS policy: "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                f"port={snapshot.server_port}, task_id={snapshot.task_id}"
            )
            raise AppWorldServerDeadError(snapshot)

    def _check_health_before_task(self, task_id: str) -> None:
        if not self.server_health.enabled:
            return
        if self._draining_after_current_task:
            logger.warning("Restarting draining AppWorld server before accepting a new task.")
            self.world.restart()
            self._draining_after_current_task = False

        snapshot = self._health_snapshot(task_id)
        self._raise_if_dead(snapshot)
        if snapshot.status == "draining":
            logger.warning(
                "AppWorld server already draining before task; restarting before new task. "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, port={snapshot.server_port}"
            )
            self.world.restart()
            return
        if snapshot.rss_gb >= self.server_health.warning_rss_gb:
            logger.warning(
                "AppWorld server RSS warning before task: "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, port={snapshot.server_port}"
            )

    def _check_health_after_task(self, task_id: str) -> None:
        if not self.server_health.enabled:
            return
        snapshot = self._health_snapshot(task_id)
        self._raise_if_dead(snapshot)
        if snapshot.status == "draining":
            logger.warning(
                "AppWorld server entering draining state after current task; "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, port={snapshot.server_port}"
            )
            self._draining_after_current_task = True
        elif snapshot.rss_gb >= self.server_health.warning_rss_gb:
            logger.warning(
                "AppWorld server RSS warning after task: "
                f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, port={snapshot.server_port}"
            )

    def _start_dead_monitor(
        self, task_id: str, stop_event: threading.Event
    ) -> tuple[threading.Thread | None, list[AppWorldServerHealthSnapshot]]:
        dead_snapshots: list[AppWorldServerHealthSnapshot] = []
        if not self.server_health.enabled:
            return None, dead_snapshots

        def _monitor() -> None:
            while not stop_event.wait(self.server_health.check_interval_seconds):
                snapshot = self._health_snapshot(task_id)
                if snapshot.status == "dead":
                    dead_snapshots.append(snapshot)
                    logger.error(
                        "AppWorld server hit dead RSS threshold during task; force closing. "
                        f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                        f"port={snapshot.server_port}, task_id={task_id}"
                    )
                    self.world.force_close_server()
                    return
                if snapshot.rss_gb >= self.server_health.warning_rss_gb:
                    logger.warning(
                        "AppWorld server RSS monitor warning: "
                        f"rss_gb={snapshot.rss_gb:.2f}, pid={snapshot.server_pid}, "
                        f"port={snapshot.server_port}, task_id={task_id}"
                    )

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        return thread, dead_snapshots

    def run(self, scenario: Scenario, llm: TrainableLLM) -> TrainingRollout:
        assert isinstance(scenario, AppWorldScenario)
        task_id = scenario.task_id

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
            monitor_thread, dead_snapshots = self._start_dead_monitor(task_id, monitor_stop_event)
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
                if dead_snapshots:
                    raise AppWorldServerDeadError(dead_snapshots[-1]) from exc
                raise
            finally:
                monitor_stop_event.set()
                if monitor_thread is not None:
                    monitor_thread.join(timeout=1.0)

        if dead_snapshots:
            raise AppWorldServerDeadError(dead_snapshots[-1])

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
