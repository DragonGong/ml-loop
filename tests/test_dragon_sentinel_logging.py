import json
import logging
from datetime import datetime
from pathlib import Path

from phi_agents.utils.logger import (
    ALLOWED_LEVELS,
    DRAGON_SENTINEL_LOG_PATH,
    SafeWatchedFileHandler,
    create_dragon_sentinel_handler,
    redact_log_text,
)

REQUIRED_FIELDS = {
    "timestamp",
    "level",
    "project",
    "service",
    "environment",
    "event",
    "message",
}


def _test_logger(log_path: Path) -> logging.Logger:
    logger = logging.Logger("dragon-sentinel-test", level=logging.DEBUG)
    logger.propagate = False
    logger.addHandler(create_dragon_sentinel_handler(log_path))
    return logger


def test_jsonl_records_are_single_line_parseable_and_complete(tmp_path: Path) -> None:
    log_path = tmp_path / "appworld.jsonl"
    logger = _test_logger(log_path)

    logger.debug("debug event", extra={"event": "debug_event"})
    logger.info("info event")
    logger.warning("warning event")
    logger.error("error event")
    logger.critical("critical event")

    try:
        raise RuntimeError("first exception line\nsecond exception line")
    except RuntimeError:
        logger.exception("operation failed", extra={"event": "operation_failed"})

    for handler in logger.handlers:
        handler.flush()
        handler.close()

    raw_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 6
    assert all(line.startswith("{") and line.endswith("}") for line in raw_lines)

    records = [json.loads(line) for line in raw_lines]
    for record in records:
        assert REQUIRED_FIELDS <= record.keys()
        assert all(isinstance(record[field], str) and record[field] for field in REQUIRED_FIELDS)
        assert record["level"] in ALLOWED_LEVELS
        assert record["project"] == "appworld"
        assert record["service"] == "appworld"
        assert record["timestamp"].endswith("Z")
        parsed_timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        assert parsed_timestamp.utcoffset().total_seconds() == 0
    assert {record["level"] for record in records} == ALLOWED_LEVELS

    exception_line = raw_lines[-1]
    assert "\\n" in exception_line
    assert records[-1]["exception"].count("\n") >= 1


def test_secrets_and_personal_information_are_redacted(tmp_path: Path) -> None:
    log_path = tmp_path / "appworld.jsonl"
    logger = _test_logger(log_path)
    sensitive_values = (
        "top-secret-password",
        "sk-token-value",
        "cookie-value",
        "webhook-secret",
        "person@example.com",
        "+86 138-1234-5678",
        "203.0.113.42",
        "Alice",
        "+1 (202) 555-0133",
        "session=abc; sid=def; csrftoken=ghi",
        "session=uvw; sid=xyz; HttpOnly",
        "sk-live-abcdefghijklmnop",
        "eyJheaderpart123.payloadpart123.signaturepart123",
    )

    try:
        raise ValueError(
            "Authorization: Bearer sk-token-value; email=person@example.com; phone=+86 138-1234-5678"
        )
    except ValueError:
        logger.exception(
            "password=top-secret-password token=sk-token-value "
            "webhook=https://example.com/hooks/webhook-secret "
            "first_name=Alice source_ip=203.0.113.42 contact=alternate@example.org "
            "caller +1 (202) 555-0133\n"
            "Cookie: cookie-value; session=abc; sid=def; csrftoken=ghi\n"
            "Set-Cookie=session=uvw; sid=xyz; HttpOnly\n"
            "bare sk-live-abcdefghijklmnop and JWT "
            "eyJheaderpart123.payloadpart123.signaturepart123"
        )

    for handler in logger.handlers:
        handler.flush()
        handler.close()

    raw_log = log_path.read_text(encoding="utf-8")
    json.loads(raw_log)
    for sensitive_value in sensitive_values:
        assert sensitive_value not in raw_log
    assert "[REDACTED]" in raw_log
    assert "[REDACTED_EMAIL]" in raw_log
    assert "[REDACTED_PHONE]" in raw_log
    assert "[REDACTED_IP]" in raw_log


def test_decimal_training_metrics_are_not_misclassified_as_phone_numbers() -> None:
    safe = redact_log_text("runtime_seconds=976.0763 validation_loss=0.320944607257843")

    assert safe == "runtime_seconds=976.0763 validation_loss=0.320944607257843"
    assert redact_log_text("call +86 138-1234-5678") == "call [REDACTED_PHONE]"


def test_default_path_and_rotation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert DRAGON_SENTINEL_LOG_PATH == Path("/var/log/dragon-sentinel/appworld/appworld.jsonl")

    rotation = (project_root / ".dragonsentinel/logrotate.conf").read_text(encoding="utf-8")
    assert str(DRAGON_SENTINEL_LOG_PATH) in rotation
    assert "rotate 14" in rotation
    assert "maxsize 100M" in rotation
    assert "compress" in rotation
    assert "copytruncate" not in rotation
    assert "create 0640" in rotation


def test_file_handler_failure_never_echoes_raw_record(tmp_path: Path, monkeypatch, capsys) -> None:
    handler = create_dragon_sentinel_handler(tmp_path / "appworld.jsonl")
    assert isinstance(handler, SafeWatchedFileHandler)

    def fail_reopen() -> None:
        raise OSError("simulated disk or rotation failure")

    monkeypatch.setattr(handler, "reopenIfNeeded", fail_reopen)
    record = logging.LogRecord(
        name="dragon-sentinel-test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="password=%s bare sk-live-handler-secret",
        args=("raw-handler-password",),
        exc_info=None,
    )

    handler.emit(record)
    handler.close()

    error_output = capsys.readouterr().err
    assert "raw-handler-password" not in error_output
    assert "sk-live-handler-secret" not in error_output
    assert "safely omitted" in error_output


def test_watched_handler_continues_after_rename_rotation(tmp_path: Path) -> None:
    log_path = tmp_path / "appworld.jsonl"
    rotated_path = tmp_path / "appworld.jsonl.1"
    logger = _test_logger(log_path)

    logger.info("before rotation", extra={"event": "before_rotation"})
    for handler in logger.handlers:
        handler.flush()

    log_path.rename(rotated_path)
    log_path.touch(mode=0o640)

    logger.info("after rotation", extra={"event": "after_rotation"})
    for handler in logger.handlers:
        handler.flush()
        handler.close()

    old_records = [json.loads(line) for line in rotated_path.read_text().splitlines()]
    new_records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["event"] for record in old_records] == ["before_rotation"]
    assert [record["event"] for record in new_records] == ["after_rotation"]
