#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""Project logging with a Dragon Sentinel compatible JSONL sink."""

import json
import logging
import os
import re
import sys
from copy import copy
from datetime import datetime, timezone
from logging.handlers import WatchedFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any

DRAGON_SENTINEL_LOG_PATH = Path("/var/log/dragon-sentinel/appworld/appworld.jsonl")
DRAGON_SENTINEL_PROJECT = "appworld"
DRAGON_SENTINEL_SERVICE = "appworld"
DRAGON_SENTINEL_ENVIRONMENT = os.environ.get("DRAGON_SENTINEL_ENVIRONMENT", "local")

ALLOWED_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
REDACTED = "[REDACTED]"

_SECRET_KEY = (
    r"(?:[a-z0-9]+[_-])*(?:password|passwd|pwd|secret|api[_-]?key|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|token|cookie|set[_-]?cookie|authorization|"
    r"auth|webhook(?:[_-]?url)?)(?:[_-][a-z0-9]+)*"
)
_PII_KEY = (
    r"(?:first[_-]?name|last[_-]?name|full[_-]?name|display[_-]?name|email|"
    r"phone(?:[_-]?number)?|mobile(?:[_-]?number)?|street[_-]?address|postal[_-]?address|"
    r"birth(?:day|date)|ssn|social[_-]?security(?:[_-]?number)?)"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b(?P<key>{_SECRET_KEY}|{_PII_KEY})\b(?P<key_quote>['\"]?)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?:(?P<quote>['\"])(?P<quoted>.*?)(?P=quote)|(?P<plain>[^\s,;}\]]+))"
)
_AUTHORIZATION_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_COOKIE_HEADER_RE = re.compile(r"(?im)\b(?P<header>set-cookie|cookie)\b['\"]?\s*[:=]\s*[^\r\n]*")
_WEBHOOK_URL_RE = re.compile(r"(?i)https?://[^\s'\"]*(?:webhook|hooks)[^\s'\"]*")
_BARE_CREDENTIAL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-[A-Za-z0-9][A-Za-z0-9._-]{7,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AIza[A-Za-z0-9_-]{20,}|"
    r"AKIA[A-Z0-9]{16}"
    r")(?![A-Za-z0-9])"
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
_IP_ADDRESS_RE = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
)
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?\d(?:[\s()-]*\d){6,14}|\d{2,4}(?:\.\d{2,4}){2,3})(?![\w.])"
)
_HOME_PATH_RE = re.compile(r"(?i)(?P<prefix>/(?:home|users)/)[^/\s]+")


def _redact_assignment(match: re.Match[str]) -> str:
    prefix = f"{match.group('key')}{match.group('key_quote')}{match.group('separator')}"
    quote = match.group("quote") or ""
    return f"{prefix}{quote}{REDACTED}{quote}"


def redact_log_text(value: Any) -> str:
    """Return a log-safe string with common secrets and personal identifiers removed."""
    text = str(value)
    text = _COOKIE_HEADER_RE.sub(lambda match: f"{match.group('header')}: {REDACTED}", text)
    text = _AUTHORIZATION_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", text)
    text = _WEBHOOK_URL_RE.sub(REDACTED, text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(_redact_assignment, text)
    text = _BARE_CREDENTIAL_RE.sub(REDACTED, text)
    text = _JWT_RE.sub(REDACTED, text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _IP_ADDRESS_RE.sub("[REDACTED_IP]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return _HOME_PATH_RE.sub(r"\g<prefix>[REDACTED_USER]", text)


def _safe_exception_text(
    exc_info: tuple[type[BaseException], BaseException, TracebackType] | tuple[None, None, None],
) -> str:
    """Keep exception type and stack locations without serializing a sensitive exception message."""
    exception_type, _exception, traceback = exc_info
    lines = [exception_type.__name__ if exception_type is not None else "Exception"]
    while traceback is not None:
        frame = traceback.tb_frame
        lines.append(
            f"  at {Path(frame.f_code.co_filename).name}:{traceback.tb_lineno} "
            f"in {frame.f_code.co_name}"
        )
        traceback = traceback.tb_next
    return "\n".join(lines)


def _metadata_value(value: str, fallback: str) -> str:
    """Keep static contract fields single-valued and free from arbitrary content."""
    return value if re.fullmatch(r"[A-Za-z0-9_.-]+", value) else fallback


class DragonSentinelJsonFormatter(logging.Formatter):
    """Format one complete Dragon Sentinel Log Contract v1 object per record."""

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname if record.levelname in ALLOWED_LEVELS else "INFO"
        item = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": level,
            "project": DRAGON_SENTINEL_PROJECT,
            "service": DRAGON_SENTINEL_SERVICE,
            "environment": _metadata_value(DRAGON_SENTINEL_ENVIRONMENT, "local"),
            "event": redact_log_text(getattr(record, "event", "application_log")),
            "message": redact_log_text(record.getMessage()),
        }
        if record.exc_info:
            item["exception"] = _safe_exception_text(record.exc_info)
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


class RedactingTextFormatter(logging.Formatter):
    """Apply the same secret and PII policy to the existing console output."""

    def formatException(self, exc_info: Any) -> str:
        return _safe_exception_text(exc_info)

    def format(self, record: logging.LogRecord) -> str:
        safe_record = copy(record)
        safe_record.msg = redact_log_text(record.getMessage())
        safe_record.args = ()
        safe_record.threadName = redact_log_text(record.threadName)
        safe_record.exc_text = None
        return super().format(safe_record)


class SafeWatchedFileHandler(WatchedFileHandler):
    """Watched file handler whose failure path never serializes the original record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        del record
        try:
            sys.stderr.write(
                "Dragon Sentinel JSONL logging failed; the log record was safely omitted.\n"
            )
        except OSError:
            pass


def create_dragon_sentinel_handler(log_path: Path | str) -> logging.Handler:
    """Create the JSONL handler; external logrotate handles bounded retention."""
    path = Path(log_path)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    handler = SafeWatchedFileHandler(path, encoding="utf-8")
    handler.setFormatter(DragonSentinelJsonFormatter())
    try:
        path.chmod(0o640)
    except PermissionError:
        # A mounted file may deliberately be owned and permissioned by the host.
        pass
    return handler


def get_phi_logger() -> logging.Logger:
    return logging.getLogger(__name__)


def setup_phi_logger(log_path: Path | str | None = None) -> None:
    """Configure the existing console logger and the Dragon Sentinel JSONL sink."""
    logger = get_phi_logger()
    for old_handler in logger.handlers:
        old_handler.close()
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        RedactingTextFormatter(
            "PHI: %(asctime)s - %(process)s - %(threadName)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(console_handler)

    try:
        logger.addHandler(create_dragon_sentinel_handler(log_path or DRAGON_SENTINEL_LOG_PATH))
    except OSError:
        # Logging must never prevent AppWorld from starting. The console remains available,
        # while deployments should fix ownership of the mounted host directory.
        logger.warning(
            "Dragon Sentinel JSONL logging is unavailable; check the log directory permissions.",
            extra={"event": "json_log_unavailable"},
        )

    logger.setLevel(logging.DEBUG)
    logger.propagate = False


setup_phi_logger()


class NullLogger(logging.Logger):
    def __init__(self) -> None:
        pass

    def __getattr__(self, name: Any) -> Any:
        def no_op(*args: Any, **kwargs: Any) -> Any:
            pass

        return no_op
