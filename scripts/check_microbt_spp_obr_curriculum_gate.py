from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean, median


class GateFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _parse_training_log(text: str) -> tuple[dict[int, dict], list[dict]]:
    rows: dict[int, dict] = {}
    exact: list[dict] = []
    current_epoch: int | None = None
    for line in text.splitlines():
        match = re.search(r"(?:^|: )Epoch (\d+)\s*$", line)
        if match:
            current_epoch = int(match.group(1))
            rows.setdefault(current_epoch, {})
        if current_epoch is not None:
            match = re.search(r"train_loss ([0-9.eE+-]+)", line)
            if match:
                rows[current_epoch]["train_loss"] = float(match.group(1))
            match = re.search(r"Pseudo dice \[([^]]+)\]", line)
            if match:
                rows[current_epoch]["dice"] = [
                    float(value.strip()) for value in match.group(1).split(",")
                ]
            match = re.search(r"Epoch time: ([0-9.eE+-]+) s", line)
            if match:
                rows[current_epoch]["seconds"] = float(match.group(1))
        marker = "MicroBT-PF exact-fast telemetry: "
        if marker in line:
            payload = json.loads(line.split(marker, 1)[1])
            payload.setdefault("epoch", current_epoch)
            exact.append(payload)
    return rows, exact


def evaluate_gate(log_path: Path, telemetry_path: Path) -> dict:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    _require("checkpoint=none" in text, "gate was not initialized from scratch")
    rows, exact = _parse_training_log(text)
    completed = [rows.get(epoch, {}) for epoch in range(30)]
    _require(
        all(len(row.get("dice", ())) == 3 for row in completed),
        "gate is missing one or more three-region Dice rows",
    )
    _require(
        all(math.isfinite(float(row.get("train_loss", math.nan))) for row in completed),
        "gate has missing or nonfinite train loss",
    )
    _require(
        all(float(row.get("seconds", math.inf)) <= 360.0 for row in completed[1:]),
        "gate exceeded the six-minute ordinary-epoch limit",
    )
    _require(len(exact) >= 30, "gate is missing optimizer telemetry")
    _require(
        all(int(item.get("nonfinite_steps", 1)) == 0 for item in exact[-30:]),
        "gate contains nonfinite optimizer telemetry",
    )
    _require(
        all(int(item.get("skipped_steps", 1)) == 0 for item in exact[-30:]),
        "gate contains skipped optimizer steps",
    )

    components = [
        json.loads(line)
        for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_components = {
        int(item["epoch"]): item
        for item in components
        if item.get("phase") == "train" and 0 <= int(item["epoch"]) < 30
    }
    _require(len(train_components) == 30, "gate is missing train component telemetry")
    first5 = [float(train_components[epoch]["region_mean"]) for epoch in range(5)]
    last5 = [float(train_components[epoch]["region_mean"]) for epoch in range(25, 30)]
    existence = [
        float(train_components[epoch]["birth_existence_raw_all_mean"])
        for epoch in range(25, 30)
    ]
    prominence = [
        float(train_components[epoch]["birth_prominence_raw_all_mean"])
        for epoch in range(25, 30)
    ]
    _require(
        all(math.isfinite(value) for value in first5 + last5 + existence + prominence),
        "gate contains nonfinite learning telemetry",
    )
    first5_region_median = median(first5)
    last5_region_median = median(last5)
    existence_mean = mean(existence)
    prominence_mean = mean(prominence)
    _require(
        last5_region_median < first5_region_median,
        "region loss did not improve during the gate",
    )
    _require(
        existence_mean < 1.50,
        "flat-field birth existence signature remains at the gate",
    )
    _require(
        prominence_mean < 0.08,
        "birth-to-moat prominence did not separate during the gate",
    )
    _require(
        any(
            float(value) > 0.0
            for epoch in range(25, 30)
            for value in completed[epoch]["dice"][1:]
        ),
        "gate shows no TC or ET semantic signal in the last five epochs",
    )
    return {
        "status": "passed",
        "epochs": 30,
        "from_scratch": True,
        "seed": 42,
        "first5_region_median": first5_region_median,
        "last5_region_median": last5_region_median,
        "last5_birth_existence_raw_mean": existence_mean,
        "last5_birth_prominence_raw_mean": prominence_mean,
        "last_epoch_dice": completed[29]["dice"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate_gate(args.log, args.telemetry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SPP_OBR_CURRICULUM_GATE_OK", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

