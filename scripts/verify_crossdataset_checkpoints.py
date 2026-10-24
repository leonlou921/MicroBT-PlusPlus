from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for name in ("checkpoint_phase_300.pth", "checkpoint_final.pth"):
        path = args.model_folder / name
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint["current_epoch"]) != args.expected_epoch:
            raise ValueError(f"checkpoint epoch drift: {path}")
        records[name] = {
            "sha256": digest(path),
            "current_epoch": int(checkpoint["current_epoch"]),
            "optimizer_state_entries": len(checkpoint["optimizer_state"]["state"]),
        }
    payload = {"schema_version": 1, "status": "passed", "checkpoints": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
