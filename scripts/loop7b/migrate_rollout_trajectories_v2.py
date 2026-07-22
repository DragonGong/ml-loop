"""Create a private, immutable v2 copy of sanitized LOOP rollout trajectories.

The migration never edits its source tree and deliberately does not read AppWorld raw outputs,
which contain plaintext credentials. Existing v1 content corruption is irreversible; this tool
only guarantees that the copied JSON is atomic, single-line, mode 600, and sanitized again with
the current policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from phi_agents.rl.rollout_diagnostics import (
    SANITIZED_TRAJECTORY_SCHEMA_VERSION,
    _redact_sensitive_text,
    _trajectory_sensitive_literals,
    _write_json,
)

SOURCE_SCHEMA_VERSION = "appworld-sanitized-trajectory-v1"
MANIFEST_SCHEMA_VERSION = "appworld-trajectory-sanitization-migration-v1"
IRREVERSIBLE_WARNING = (
    "The v1 sanitizer may already have removed delimiters; this migration does not reconstruct "
    "or claim to repair pre-existing content corruption."
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _validate_messages(payload: dict[str, Any], source_path: Path) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source_path}: messages must be a non-empty list")
    validated: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{source_path}: message {index} must contain only role and content")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"{source_path}: message {index} values must be strings")
        validated.append({"role": role, "content": content})
    return validated


def _sanitize_payload(payload: dict[str, Any], source_path: Path) -> tuple[dict[str, Any], int]:
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"{source_path}: expected {SOURCE_SCHEMA_VERSION}, got "
            f"{payload.get('schema_version')!r}"
        )
    messages = _validate_messages(payload, source_path)
    contents = [message["content"] for message in messages]
    sensitive_literals = _trajectory_sensitive_literals(contents)
    sanitized_messages = [
        {
            "role": message["role"],
            "content": _redact_sensitive_text(
                message["content"], sensitive_literals=sensitive_literals
            ),
        }
        for message in messages
    ]
    for index, message in enumerate(sanitized_messages):
        content = message["content"]
        if _redact_sensitive_text(content) != content:
            raise ValueError(f"{source_path}: message {index} failed the idempotent privacy check")

    migrated = deepcopy(payload)
    migrated["schema_version"] = SANITIZED_TRAJECTORY_SCHEMA_VERSION
    migrated["messages"] = sanitized_messages
    migrated["sanitization"] = {
        "policy": "appworld-trajectory-redaction-v2",
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "pre_existing_corruption_repaired": False,
        "warning": IRREVERSIBLE_WARNING,
    }
    changed_count = sum(
        before["content"] != after["content"]
        for before, after in zip(messages, sanitized_messages, strict=True)
    )
    return migrated, changed_count


def _validate_roots(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    if source_root.is_symlink():
        raise ValueError("source root must not be a symlink")
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {output_root}")
    if (
        source_root == output_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise ValueError("source and output roots must be separate, non-nested trees")
    return source_root, output_root


def migrate_trajectories(
    *, source_root: Path, output_root: Path, expected_count: int | None = None
) -> Path:
    """Migrate all v1 trajectory.json files to a new v2 tree and return its manifest."""
    source_root, output_root = _validate_roots(source_root, output_root)
    source_paths = sorted(source_root.rglob("trajectory.json"))
    if not source_paths:
        raise ValueError(f"no trajectory.json files found under {source_root}")
    if expected_count is not None and len(source_paths) != expected_count:
        raise ValueError(f"expected {expected_count} trajectories, found {len(source_paths)}")
    if any(path.is_symlink() for path in source_paths):
        raise ValueError("trajectory symlinks are not allowed")

    _private_directory(output_root)
    incomplete_path = output_root / "MIGRATION_INCOMPLETE"
    incomplete_path.touch(mode=0o600, exist_ok=False)
    incomplete_path.chmod(0o600)
    rows: list[dict[str, Any]] = []
    try:
        for source_path in source_paths:
            source_bytes = source_path.read_bytes()
            payload = json.loads(source_bytes)
            if not isinstance(payload, dict):
                raise ValueError(f"{source_path}: root JSON value must be an object")
            migrated, changed_count = _sanitize_payload(payload, source_path)
            relative_path = source_path.relative_to(source_root)
            output_path = output_root / relative_path
            _private_directory(output_path.parent)
            _write_json(output_path, migrated, overwrite=False)

            output_bytes = output_path.read_bytes()
            if len(output_bytes.splitlines()) != 1 or not isinstance(
                json.loads(output_bytes), dict
            ):
                raise ValueError(f"{output_path}: output is not one complete JSON line")
            rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "source_sha256": _sha256_bytes(source_bytes),
                    "output_sha256": _sha256_bytes(output_bytes),
                    "changed_message_count": changed_count,
                }
            )

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "output_schema_version": SANITIZED_TRAJECTORY_SCHEMA_VERSION,
            "trajectory_count": len(rows),
            "changed_trajectory_count": sum(row["changed_message_count"] > 0 for row in rows),
            "warning": IRREVERSIBLE_WARNING,
            "files": rows,
        }
        manifest_path = output_root / "migration_manifest.json"
        _write_json(manifest_path, manifest, overwrite=False)
        incomplete_path.unlink()
        return manifest_path
    except BaseException:
        # A failed destination remains quarantined and is never mistaken for a complete migration.
        incomplete_path.touch(mode=0o600, exist_ok=True)
        incomplete_path.chmod(0o600)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = migrate_trajectories(
        source_root=args.source_root,
        output_root=args.output_root,
        expected_count=args.expected_count,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
