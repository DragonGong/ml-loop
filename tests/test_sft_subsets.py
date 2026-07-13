import logging

from phi_agents.sft.launch import run
from phi_agents.sft.subsets import balanced_nested_order
from phi_agents.utils.logger import get_phi_logger


def _row(index: int) -> dict:
    return {
        "metadata": {
            "trajectory_id": f"trajectory-{index}",
            "scenario_id": f"scenario-{index % 4}",
            "task_id": f"task-{index % 12}",
            "success_type": "clean_success" if index % 3 == 0 else "recovered_success",
        }
    }


def test_balanced_nested_order_is_complete_deterministic_and_seeded() -> None:
    rows = [_row(index) for index in range(40)]
    tokens = {row["metadata"]["trajectory_id"]: 10 + index for index, row in enumerate(rows)}
    first = balanced_nested_order(rows, tokens, seed=20260713)
    repeat = balanced_nested_order(rows, tokens, seed=20260713)
    alternate = balanced_nested_order(rows, tokens, seed=20260714)

    def ids(values: list[dict]) -> list[str]:
        return [row["metadata"]["trajectory_id"] for row in values]

    assert ids(first) == ids(repeat)
    assert ids(first) != ids(alternate)
    assert set(ids(first)) == set(ids(rows))
    assert set(ids(first[:10])) < set(ids(first[:20])) < set(ids(first))


def test_failed_sft_subprocess_emits_dragon_sentinel_alert_event() -> None:
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = get_phi_logger()
    handler = Capture()
    production_handlers = list(logger.handlers)
    logger.handlers.clear()
    logger.addHandler(handler)
    try:
        assert run(["python", "-c", "raise SystemExit(7)"]) == 7
    finally:
        logger.handlers.clear()
        logger.handlers.extend(production_handlers)

    assert records[-1].levelname == "CRITICAL"
    assert records[-1].event == "training_failed"
