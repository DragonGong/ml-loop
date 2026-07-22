import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts.loop7b.migrate_rollout_trajectories_v2 import migrate_trajectories


def _source_trajectory(root: Path) -> Path:
    path = root / "iteration-000001/scenario-0000/rollout-00/trajectory.json"
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": "appworld-sanitized-trajectory-v1",
        "iteration": 1,
        "scenario_idx": 0,
        "rollout_idx": 0,
        "task_id": "task_1",
        "dataset_name": "train_difficulty_1_2",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "```python\n"
                    "fs_password = 'secret-value'\n"
                    "result = login(password=fs_password, access_token=token)\n"
                    "```"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Access token: eyJheaderpart123.payloadpart123.signaturepart123\n"
                    "My name is: Alice Smith. email=alice@example.com "
                    "phone=+86 138-1234-5678\nsecret-value"
                ),
            },
        ],
        "metadata": {"strict_success": False},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_migration_is_non_destructive_private_atomic_and_manifested(tmp_path: Path) -> None:
    source_root = tmp_path / "v1"
    source_path = _source_trajectory(source_root)
    source_before = source_path.read_bytes()
    source_hash = hashlib.sha256(source_before).hexdigest()
    output_root = tmp_path / "v2"

    manifest_path = migrate_trajectories(
        source_root=source_root,
        output_root=output_root,
        expected_count=1,
    )

    assert source_path.read_bytes() == source_before
    output_path = output_root / source_path.relative_to(source_root)
    output_bytes = output_path.read_bytes()
    assert len(output_bytes.splitlines()) == 1
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    output = json.loads(output_bytes)
    serialized = json.dumps(output)
    assert output["schema_version"] == "appworld-sanitized-trajectory-v2"
    assert output["sanitization"]["pre_existing_corruption_repaired"] is False
    for sensitive in (
        "secret-value",
        "eyJheaderpart123.payloadpart123.signaturepart123",
        "Alice Smith",
        "alice@example.com",
        "+86 138-1234-5678",
    ):
        assert sensitive not in serialized

    manifest_bytes = manifest_path.read_bytes()
    assert len(manifest_bytes.splitlines()) == 1
    manifest = json.loads(manifest_bytes)
    assert manifest["trajectory_count"] == 1
    assert manifest["changed_trajectory_count"] == 1
    assert manifest["files"][0]["source_sha256"] == source_hash
    assert manifest["files"][0]["output_sha256"] == hashlib.sha256(output_bytes).hexdigest()
    assert not (output_root / "MIGRATION_INCOMPLETE").exists()


def test_migration_refuses_existing_destination_and_count_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "v1"
    _source_trajectory(source_root)
    existing_output = tmp_path / "existing"
    existing_output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to reuse"):
        migrate_trajectories(source_root=source_root, output_root=existing_output)
    with pytest.raises(ValueError, match="expected 2 trajectories, found 1"):
        migrate_trajectories(
            source_root=source_root,
            output_root=tmp_path / "count-mismatch",
            expected_count=2,
        )


def test_failed_migration_leaves_private_incomplete_marker(tmp_path: Path) -> None:
    source_root = tmp_path / "bad-v1"
    source_path = _source_trajectory(source_root)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "unexpected-schema"
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    output_root = tmp_path / "bad-v2"

    with pytest.raises(ValueError, match="expected appworld-sanitized-trajectory-v1"):
        migrate_trajectories(source_root=source_root, output_root=output_root)

    marker = output_root / "MIGRATION_INCOMPLETE"
    assert marker.is_file()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert not (output_root / "migration_manifest.json").exists()
