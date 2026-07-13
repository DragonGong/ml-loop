from __future__ import annotations

import argparse
import signal
import subprocess
from typing import Any

from phi_agents.utils.logger import get_phi_logger

logger = get_phi_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT and emit Dragon Sentinel training_failed on non-zero exit."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def run(command: list[str]) -> int:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A child training command is required after --")

    child = subprocess.Popen(command)
    previous_handlers: dict[signal.Signals, Any] = {}

    def forward(signum: int, frame: Any) -> None:
        del frame
        if child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        return_code = child.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if return_code != 0:
        logger.critical(
            "AppWorld SFT subprocess exited unsuccessfully (exit_code=%s).",
            return_code,
            extra={"event": "training_failed"},
        )
    return return_code


def main() -> None:
    raise SystemExit(run(parse_args().command))


if __name__ == "__main__":
    main()
