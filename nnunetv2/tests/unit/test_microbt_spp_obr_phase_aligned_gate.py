from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from check_microbt_spp_obr_phase_aligned_gate import (  # noqa: E402
    GateFailure,
    evaluate_gate,
)


SIZE_COUNTS = {"le3mm": 18, "3to10mm": 22, "gt10mm": 12}


def _write_probe(
    path: Path,
    *,
    gap: float,
    prominence_rate: float,
    micro_prominence_rate: float,
) -> None:
    rows: list[dict[str, object]] = []
    for size_bin, count in SIZE_COUNTS.items():
        rate = micro_prominence_rate if size_bin == "le3mm" else prominence_rate
        passes = max(1, round(count * rate))
        for index in range(count):
            rows.append(
                {
                    "subject_id": f"case_{size_bin}_{index:02d}",
                    "instance_id": index + 1,
                    "size_bin": size_bin,
                    "has_moat": "True",
                    "gap": gap,
                    "existence_pass": "True",
                    "prominence_pass": str(index < passes),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_health(log_path: Path, telemetry_path: Path) -> None:
    lines = ["SPP scratch initialization: checkpoint=none"]
    telemetry = []
    for epoch in range(60):
        lines.extend(
            (
                f"Epoch {epoch}",
                "train_loss 0.5000",
                "Pseudo dice [0.55, 0.40, 0.30]",
                "Epoch time: 250.0 s",
                "MicroBT-PF exact-fast telemetry: "
                + json.dumps({"nonfinite_steps": 0, "skipped_steps": 0}),
            )
        )
        telemetry.append(
            {
                "epoch": epoch,
                "phase": "train",
                "region_mean": 0.5,
                "nonfinite_steps": 0,
                "skipped_steps": 0,
            }
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    telemetry_path.write_text(
        "\n".join(json.dumps(item) for item in telemetry) + "\n",
        encoding="utf-8",
    )


def test_epoch60_gate_passes_only_complete_healthy_object_contract(tmp_path: Path) -> None:
    probe45 = tmp_path / "45.csv"
    probe60 = tmp_path / "60.csv"
    log = tmp_path / "training.txt"
    telemetry = tmp_path / "learning.jsonl"
    _write_probe(probe45, gap=0.07, prominence_rate=0.25, micro_prominence_rate=0.17)
    _write_probe(probe60, gap=0.08, prominence_rate=0.30, micro_prominence_rate=0.22)
    _write_health(log, telemetry)

    result = evaluate_gate(log, telemetry, probe45, probe60)

    assert result["status"] == "passed"
    assert result["epoch"] == 60
    assert result["object_count"] == 52
    assert result["overall_existence_pass_rate"] == pytest.approx(1.0)
    assert result["overall_gap_median"] == pytest.approx(0.08)
    assert result["micro_prominence_pass_rate"] >= 0.10


def test_epoch60_gate_rejects_zero_micro_prominence(tmp_path: Path) -> None:
    probe45 = tmp_path / "45.csv"
    probe60 = tmp_path / "60.csv"
    log = tmp_path / "training.txt"
    telemetry = tmp_path / "learning.jsonl"
    _write_probe(probe45, gap=0.07, prominence_rate=0.25, micro_prominence_rate=0.17)
    _write_probe(probe60, gap=0.08, prominence_rate=0.30, micro_prominence_rate=0.0)
    _write_health(log, telemetry)

    with pytest.raises(GateFailure, match="micro-lesion prominence"):
        evaluate_gate(log, telemetry, probe45, probe60)


def test_epoch60_gate_rejects_fixed_probe_regression(tmp_path: Path) -> None:
    probe45 = tmp_path / "45.csv"
    probe60 = tmp_path / "60.csv"
    log = tmp_path / "training.txt"
    telemetry = tmp_path / "learning.jsonl"
    _write_probe(probe45, gap=0.09, prominence_rate=0.30, micro_prominence_rate=0.22)
    _write_probe(probe60, gap=0.08, prominence_rate=0.30, micro_prominence_rate=0.22)
    _write_health(log, telemetry)

    with pytest.raises(GateFailure, match="gap regressed"):
        evaluate_gate(log, telemetry, probe45, probe60)
