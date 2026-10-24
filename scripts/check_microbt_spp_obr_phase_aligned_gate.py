from __future__ import annotations

import argparse
import csv
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


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_probe(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 52, f"fixed probe must contain 52 objects: {path}")
    required = {
        "subject_id",
        "instance_id",
        "size_bin",
        "has_moat",
        "gap",
        "existence_pass",
        "prominence_pass",
    }
    _require(bool(rows) and required.issubset(rows[0]), f"invalid probe schema: {path}")
    _require(all(_as_bool(row["has_moat"]) for row in rows), "probe has absent moat")
    return rows


def _probe_metrics(rows: list[dict[str, str]]) -> dict:
    gaps = [float(row["gap"]) for row in rows]
    existence = [_as_bool(row["existence_pass"]) for row in rows]
    prominence = [_as_bool(row["prominence_pass"]) for row in rows]
    by_size = {}
    for size_bin in ("le3mm", "3to10mm", "gt10mm"):
        selected = [row for row in rows if row["size_bin"] == size_bin]
        _require(bool(selected), f"missing size stratum: {size_bin}")
        by_size[size_bin] = {
            "object_count": len(selected),
            "existence_pass_rate": mean(
                _as_bool(row["existence_pass"]) for row in selected
            ),
            "prominence_pass_rate": mean(
                _as_bool(row["prominence_pass"]) for row in selected
            ),
        }
    return {
        "object_count": len(rows),
        "gap_median": median(gaps),
        "existence_pass_rate": mean(existence),
        "prominence_pass_rate": mean(prominence),
        "by_size": by_size,
    }


def _parse_health(log_path: Path, telemetry_path: Path) -> dict:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    _require("checkpoint=none" in text, "run lacks scratch initialization evidence")
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
            exact.append(json.loads(line.split(marker, 1)[1]))

    completed = [rows.get(epoch, {}) for epoch in range(60)]
    _require(all(len(row.get("dice", ())) == 3 for row in completed), "missing Dice row")
    _require(
        all(math.isfinite(float(row.get("train_loss", math.nan))) for row in completed),
        "missing or nonfinite train loss",
    )
    _require(
        all(float(row.get("seconds", math.inf)) <= 360.0 for row in completed[1:]),
        "ordinary epoch exceeded six minutes",
    )
    _require(len(exact) >= 60, "missing optimizer telemetry")
    _require(
        all(int(item.get("nonfinite_steps", 1)) == 0 for item in exact[-60:]),
        "nonfinite optimizer step",
    )
    _require(
        all(int(item.get("skipped_steps", 1)) == 0 for item in exact[-60:]),
        "skipped optimizer step",
    )
    last5_dice = [completed[epoch]["dice"] for epoch in range(55, 60)]
    _require(
        all(median(row[index] for row in last5_dice) > 0.0 for index in range(3)),
        "segmentation signal collapsed in the last five epochs",
    )

    components = [
        json.loads(line)
        for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = {
        int(item["epoch"]): item
        for item in components
        if item.get("phase") == "train" and 0 <= int(item["epoch"]) < 60
    }
    _require(len(train) == 60, "missing train component telemetry")
    _require(
        all(math.isfinite(float(train[epoch]["region_mean"])) for epoch in range(60)),
        "nonfinite region telemetry",
    )
    return {
        "last5_dice_median": [
            median(row[index] for row in last5_dice) for index in range(3)
        ],
        "max_epoch_seconds": max(float(row["seconds"]) for row in completed),
    }


def evaluate_gate(
    log_path: Path,
    telemetry_path: Path,
    probe45_path: Path,
    probe60_path: Path,
) -> dict:
    health = _parse_health(log_path, telemetry_path)
    rows45 = _read_probe(probe45_path)
    rows60 = _read_probe(probe60_path)
    ids45 = {(row["subject_id"], row["instance_id"]) for row in rows45}
    ids60 = {(row["subject_id"], row["instance_id"]) for row in rows60}
    _require(ids45 == ids60, "fixed probe object identities changed")
    metrics45 = _probe_metrics(rows45)
    metrics60 = _probe_metrics(rows60)

    _require(metrics60["existence_pass_rate"] >= 0.80, "existence pass below 80%")
    _require(metrics60["gap_median"] >= 0.05, "overall median gap below 0.05")
    _require(
        metrics60["prominence_pass_rate"] >= 0.20,
        "overall prominence pass below 20%",
    )
    _require(
        all(item["prominence_pass_rate"] > 0.0 for item in metrics60["by_size"].values()),
        "a size stratum has zero prominence pass",
    )
    micro = metrics60["by_size"]["le3mm"]
    _require(micro["existence_pass_rate"] >= 0.75, "micro-lesion existence below 75%")
    _require(
        micro["prominence_pass_rate"] >= 0.10,
        "micro-lesion prominence below 10%",
    )
    _require(
        metrics60["gap_median"] >= metrics45["gap_median"],
        "fixed-object gap regressed from epoch 45 to 60",
    )
    _require(
        metrics60["prominence_pass_rate"] >= metrics45["prominence_pass_rate"],
        "fixed-object prominence regressed from epoch 45 to 60",
    )
    return {
        "status": "passed",
        "epoch": 60,
        "from_scratch": True,
        "object_count": metrics60["object_count"],
        "overall_existence_pass_rate": metrics60["existence_pass_rate"],
        "overall_gap_median": metrics60["gap_median"],
        "overall_prominence_pass_rate": metrics60["prominence_pass_rate"],
        "micro_existence_pass_rate": micro["existence_pass_rate"],
        "micro_prominence_pass_rate": micro["prominence_pass_rate"],
        "epoch45": metrics45,
        "epoch60": metrics60,
        "health": health,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--probe-45", type=Path, required=True)
    parser.add_argument("--probe-60", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate_gate(
        args.log, args.telemetry, args.probe_45, args.probe_60
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SPP_OBR_PHASE_ALIGNED_GATE_OK", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

