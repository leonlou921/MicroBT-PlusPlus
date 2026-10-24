from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from check_microbt_spp_obr_h0off_gate import GateFailure, evaluate_gate  # noqa: E402


SIZE_COUNTS = {"le3mm": 18, "3to10mm": 22, "gt10mm": 12}


def _write_probe(path: Path, gap: float = 0.08) -> None:
    rows = []
    for size_bin, count in SIZE_COUNTS.items():
        prominence_passes = 3 if size_bin == "le3mm" else max(3, count // 4)
        for index in range(count):
            rows.append(
                {
                    "subject_id": f"case_{size_bin}_{index:02d}",
                    "instance_id": index + 1,
                    "size_bin": size_bin,
                    "has_moat": "True",
                    "gap": gap,
                    "existence_pass": "True",
                    "prominence_pass": str(index < prominence_passes),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_health(
    log_path: Path,
    telemetry_path: Path,
    *,
    late_dice_scale: float = 1.0,
    h0_active_steps: int = 0,
) -> None:
    lines = ["SPP scratch initialization: checkpoint=none"]
    telemetry = []
    for epoch in range(60):
        dice = [0.50, 0.40, 0.30]
        if epoch >= 55:
            dice = [value * late_dice_scale for value in dice]
        seconds = 400.0 if epoch == 0 else 250.0
        lines.extend(
            (
                f"Epoch {epoch}",
                "train_loss 0.8",
                f"Pseudo dice [{', '.join(str(x) for x in dice)}]",
                f"Epoch time: {seconds} s",
                "MicroBT-PF exact-fast telemetry: "
                + json.dumps({"nonfinite_steps": 0, "skipped_steps": 0}),
            )
        )
        telemetry.append(
            {
                "epoch": epoch,
                "phase": "train",
                "region_mean": 0.8 if epoch < 55 else 0.9,
                "h0_active_steps": h0_active_steps,
            }
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    telemetry_path.write_text(
        "\n".join(json.dumps(item) for item in telemetry) + "\n",
        encoding="utf-8",
    )


def test_h0off_gate_accepts_preserved_segmentation_and_objects(tmp_path: Path) -> None:
    log, telemetry = tmp_path / "train.log", tmp_path / "learning.jsonl"
    probe45, probe60 = tmp_path / "45.csv", tmp_path / "60.csv"
    _write_health(log, telemetry)
    _write_probe(probe45, gap=0.07)
    _write_probe(probe60, gap=0.08)

    result = evaluate_gate(log, telemetry, probe45, probe60)

    assert result["status"] == "passed"
    assert result["h0_active_steps_total"] == 0
    assert result["object_count"] == 52


def test_h0off_gate_rejects_any_exact_h0_execution(tmp_path: Path) -> None:
    log, telemetry = tmp_path / "train.log", tmp_path / "learning.jsonl"
    probe45, probe60 = tmp_path / "45.csv", tmp_path / "60.csv"
    _write_health(log, telemetry, h0_active_steps=1)
    _write_probe(probe45)
    _write_probe(probe60)
    with pytest.raises(GateFailure, match="exact H0 was active"):
        evaluate_gate(log, telemetry, probe45, probe60)


def test_h0off_gate_rejects_late_segmentation_collapse(tmp_path: Path) -> None:
    log, telemetry = tmp_path / "train.log", tmp_path / "learning.jsonl"
    probe45, probe60 = tmp_path / "45.csv", tmp_path / "60.csv"
    _write_health(log, telemetry, late_dice_scale=0.2)
    _write_probe(probe45)
    _write_probe(probe60)
    with pytest.raises(GateFailure, match="Dice retention"):
        evaluate_gate(log, telemetry, probe45, probe60)

