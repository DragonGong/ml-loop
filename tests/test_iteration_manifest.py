import json
from pathlib import Path

import pytest

from phi_agents.rl.iteration_manifest import (
    IterationManifestStore,
    manifest_abort_fields_from_exception,
)


def _manifest_path(root: Path, iteration: int) -> Path:
    return root / "iteration_manifests" / f"iteration-{iteration:06d}.json"


def test_iteration_manifest_pending_commit_and_assert(tmp_path: Path) -> None:
    store = IterationManifestStore(tmp_path)

    pending = store.begin(iteration=123, expected_rollouts=144)
    assert pending.status == "pending"
    assert pending.finished_rollouts == 0

    committed = store.commit(
        iteration=123,
        expected_rollouts=144,
        finished_rollouts=144,
    )
    assert committed.status == "committed"
    assert committed.finished_rollouts == 144
    store.assert_committed(iteration=123, expected_rollouts=144)

    data = json.loads(_manifest_path(tmp_path, 123).read_text())
    assert data["status"] == "committed"
    assert data["expected_rollouts"] == 144
    assert data["finished_rollouts"] == 144


def test_iteration_manifest_rejects_incomplete_commit(tmp_path: Path) -> None:
    store = IterationManifestStore(tmp_path)
    store.begin(iteration=1, expected_rollouts=144)

    with pytest.raises(ValueError):
        store.commit(iteration=1, expected_rollouts=144, finished_rollouts=143)


def test_iteration_manifest_archives_pending_on_retry(tmp_path: Path) -> None:
    store = IterationManifestStore(tmp_path)
    store.begin(iteration=18, expected_rollouts=144)

    retry = store.begin(iteration=18, expected_rollouts=144)

    assert retry.status == "pending"
    assert retry.attempt == 2
    history_files = list((tmp_path / "iteration_manifests" / "history").rglob("*.json"))
    assert len(history_files) == 1
    assert json.loads(history_files[0].read_text())["status"] == "pending"


def test_iteration_manifest_archives_orphan_committed_without_checkpoint(tmp_path: Path) -> None:
    store = IterationManifestStore(tmp_path)
    store.begin(iteration=18, expected_rollouts=144)
    store.commit(iteration=18, expected_rollouts=144, finished_rollouts=144)

    retry = store.begin(iteration=18, expected_rollouts=144)

    assert retry.status == "pending"
    assert retry.attempt == 2
    history_files = list((tmp_path / "iteration_manifests" / "history").rglob("*.json"))
    assert len(history_files) == 1
    assert json.loads(history_files[0].read_text())["status"] == "committed"


def test_iteration_manifest_abort_records_server_metadata(tmp_path: Path) -> None:
    store = IterationManifestStore(tmp_path)
    store.begin(iteration=18, expected_rollouts=144)

    aborted = store.abort(
        iteration=18,
        expected_rollouts=144,
        finished_rollouts=19,
        reason="appworld_server_rss_55gb",
        server_pid=2489447,
        server_port=37161,
        task_id="d0b1f43_3",
    )

    assert aborted.status == "aborted"
    assert aborted.finished_rollouts == 19
    assert aborted.reason == "appworld_server_rss_55gb"
    assert aborted.server_pid == 2489447
    assert aborted.server_port == 37161
    assert aborted.task_id == "d0b1f43_3"


def test_manifest_abort_fields_from_exception() -> None:
    exc = RuntimeError("boom")
    exc.reason = "appworld_server_rss_55gb"  # type: ignore[attr-defined]
    exc.server_pid = 2489447  # type: ignore[attr-defined]
    exc.server_port = 37161  # type: ignore[attr-defined]
    exc.task_id = "d0b1f43_3"  # type: ignore[attr-defined]
    exc.finished_rollouts = 19  # type: ignore[attr-defined]

    assert manifest_abort_fields_from_exception(exc) == {
        "reason": "appworld_server_rss_55gb",
        "server_pid": 2489447,
        "server_port": 37161,
        "task_id": "d0b1f43_3",
        "finished_rollouts": 19,
    }
