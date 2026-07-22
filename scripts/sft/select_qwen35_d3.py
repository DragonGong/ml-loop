from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epoch_record(run_dir: Path, epoch: int) -> dict[str, Any]:
    pointer_path = run_dir / f"epoch-{epoch}.json"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Missing epoch pointer: {pointer_path}")
    pointer = json.loads(pointer_path.read_text())
    checkpoint = Path(pointer["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (run_dir / checkpoint).resolve()
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text())
    candidates = [
        row
        for row in state.get("log_history", [])
        if row.get("eval_loss") is not None
        and row.get("epoch") is not None
        and abs(float(row["epoch"]) - epoch) < 1e-5
    ]
    if not candidates:
        raise ValueError(f"No epoch-{epoch} validation loss in {state_path}")
    adapter = checkpoint / "lora"
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / filename).is_file():
            raise FileNotFoundError(f"Incomplete epoch adapter: {adapter / filename}")
    return {
        "epoch": epoch,
        "global_step": int(pointer["global_step"]),
        "validation_loss": float(candidates[-1]["eval_loss"]),
        "checkpoint": str(checkpoint),
        "adapter_path": str(adapter),
        "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
    }


def select(run_dir: Path) -> dict[str, Any]:
    epochs = [_epoch_record(run_dir, epoch) for epoch in (1, 2)]
    selected = min(epochs, key=lambda row: (row["validation_loss"], row["epoch"]))
    result = {
        "selection_rule": "lowest validation loss; ties choose epoch 1",
        "epochs": epochs,
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_adapter_path": selected["adapter_path"],
        "selected_adapter_sha256": selected["adapter_sha256"],
    }
    (run_dir / "selected_adapter.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(args.run_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
