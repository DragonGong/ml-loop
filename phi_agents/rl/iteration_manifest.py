#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import phi_agents.utils.file_utils as fu


IterationManifestStatus = Literal["pending", "committed", "aborted"]


@dataclass
class IterationManifest:
    iteration: int
    expected_rollouts: int
    finished_rollouts: int
    status: IterationManifestStatus
    attempt: int
    reason: str | None = None
    server_pid: int | None = None
    server_port: int | None = None
    task_id: str | None = None


class IterationManifestStore:
    """Durable two-phase gate for RL iteration rollout batches."""

    def __init__(self, cloud_path: Path, *, enabled: bool = True) -> None:
        self._cloud_path = cloud_path
        self.enabled = enabled
        self._manifest_dir = cloud_path / "iteration_manifests"
        self._history_dir = self._manifest_dir / "history"

    def begin(self, *, iteration: int, expected_rollouts: int) -> IterationManifest:
        manifest = IterationManifest(
            iteration=iteration,
            expected_rollouts=expected_rollouts,
            finished_rollouts=0,
            status="pending",
            attempt=self._next_attempt(iteration),
        )
        if not self.enabled:
            return manifest

        existing = self._read(iteration)
        if existing is not None:
            checkpoint_path = self._cloud_path / f"checkpoint-{iteration}"
            if existing.status != "committed" or not fu.exists(checkpoint_path / "done.txt"):
                self._archive_existing(iteration, existing)

        self._write(manifest)
        return manifest

    def commit(
        self, *, iteration: int, expected_rollouts: int, finished_rollouts: int
    ) -> IterationManifest:
        if finished_rollouts != expected_rollouts:
            raise ValueError(f"{finished_rollouts=} does not match {expected_rollouts=}")

        manifest = self._with_current_attempt(
            iteration=iteration,
            expected_rollouts=expected_rollouts,
            finished_rollouts=finished_rollouts,
            status="committed",
        )
        if self.enabled:
            self._write(manifest)
        return manifest

    def abort(
        self,
        *,
        iteration: int,
        expected_rollouts: int,
        finished_rollouts: int,
        reason: str,
        server_pid: int | None = None,
        server_port: int | None = None,
        task_id: str | None = None,
    ) -> IterationManifest:
        manifest = self._with_current_attempt(
            iteration=iteration,
            expected_rollouts=expected_rollouts,
            finished_rollouts=finished_rollouts,
            status="aborted",
            reason=reason,
            server_pid=server_pid,
            server_port=server_port,
            task_id=task_id,
        )
        if self.enabled:
            self._write(manifest)
        return manifest

    def assert_committed(self, *, iteration: int, expected_rollouts: int) -> None:
        if not self.enabled:
            return

        manifest = self._read(iteration)
        if manifest is None:
            raise RuntimeError(f"Missing iteration manifest for {iteration=}")
        if manifest.status != "committed":
            raise RuntimeError(f"Iteration {iteration} is not committed: {manifest.status}")
        if manifest.expected_rollouts != expected_rollouts:
            raise RuntimeError(
                f"Iteration {iteration} expected_rollouts changed: "
                f"{manifest.expected_rollouts} != {expected_rollouts}"
            )
        if manifest.finished_rollouts != expected_rollouts:
            raise RuntimeError(
                f"Iteration {iteration} has incomplete rollouts: "
                f"{manifest.finished_rollouts} != {expected_rollouts}"
            )

    def _with_current_attempt(
        self,
        *,
        iteration: int,
        expected_rollouts: int,
        finished_rollouts: int,
        status: IterationManifestStatus,
        reason: str | None = None,
        server_pid: int | None = None,
        server_port: int | None = None,
        task_id: str | None = None,
    ) -> IterationManifest:
        existing = self._read(iteration) if self.enabled else None
        attempt = existing.attempt if existing is not None else 1
        return IterationManifest(
            iteration=iteration,
            expected_rollouts=expected_rollouts,
            finished_rollouts=finished_rollouts,
            status=status,
            attempt=attempt,
            reason=reason,
            server_pid=server_pid,
            server_port=server_port,
            task_id=task_id,
        )

    def _next_attempt(self, iteration: int) -> int:
        existing = self._read(iteration) if self.enabled else None
        return 1 if existing is None else existing.attempt + 1

    def _manifest_path(self, iteration: int) -> Path:
        return self._manifest_dir / f"iteration-{iteration:06d}.json"

    def _read(self, iteration: int) -> IterationManifest | None:
        path = self._manifest_path(iteration)
        if not fu.exists(path):
            return None
        with fu.uri_open(path, "r") as f:
            data = json.load(f)
        return IterationManifest(**data)

    def _archive_existing(self, iteration: int, manifest: IterationManifest) -> None:
        src = self._manifest_path(iteration)
        if not fu.exists(src):
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dst = (
            self._history_dir
            / f"iteration-{iteration:06d}-attempt-{manifest.attempt:03d}"
            / f"{manifest.status}-{timestamp}.json"
        )
        fu.safe_mkdir(dst.parent, parents=True, exist_ok=True)
        fu.copy(src, dst)
        fu.delete(src)

    def _write(self, manifest: IterationManifest) -> None:
        path = self._manifest_path(manifest.iteration)
        data = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
        scheme, raw_path = fu.get_scheme_and_path(path)
        if scheme == "file":
            dst = Path(raw_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent)
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                Path(tmp_name).replace(dst)
            finally:
                tmp_path = Path(tmp_name)
                if tmp_path.exists():
                    tmp_path.unlink()
        else:
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(data)
                tmp_name = tmp.name
            try:
                fu.copy(tmp_name, path)
            finally:
                Path(tmp_name).unlink(missing_ok=True)


def manifest_abort_fields_from_exception(exc: BaseException) -> dict[str, Any]:
    """Extract structured abort metadata from rollout/server exceptions."""

    return {
        "reason": str(getattr(exc, "reason", type(exc).__name__)),
        "server_pid": getattr(exc, "server_pid", None),
        "server_port": getattr(exc, "server_port", None),
        "task_id": getattr(exc, "task_id", None),
        "finished_rollouts": int(getattr(exc, "finished_rollouts", 0) or 0),
    }
