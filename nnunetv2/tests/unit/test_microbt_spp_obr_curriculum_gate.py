from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_microbt_spp_obr_curriculum_gate import GateFailure, evaluate_gate


def _write_gate_fixture(
    root: Path,
    *,
    birth_existence_last5: float = 1.30,
    skipped_steps: int = 0,
) -> tuple[Path, Path]:
    log_path = root / "training_log.txt"
    telemetry_path = root / "learning_components.jsonl"
    log_lines = [
        "SPP scratch initialization: seed=42, optimizer_state_entries=0, checkpoint=none"
    ]
    telemetry = []
    first_region = [0.95, 0.93, 0.92, 0.91, 0.90]
    last_region = [0.84, 0.83, 0.82, 0.81, 0.80]
    for epoch in range(30):
        if epoch < 5:
            region = first_region[epoch]
        elif epoch >= 25:
            region = last_region[epoch - 25]
        else:
            region = 0.89 - 0.002 * epoch
        dice = [0.20, 0.05, 0.04] if epoch >= 25 else [0.08, 0.0, 0.0]
        log_lines.extend(
            [
                f"Epoch {epoch}",
                f"train_loss {region + 0.1:.4f}",
                f"Pseudo dice {dice}",
                "Epoch time: 240.0 s",
                "MicroBT-PF exact-fast telemetry: "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "nonfinite_steps": 0,
                        "skipped_steps": skipped_steps if epoch == 29 else 0,
                    }
                ),
            ]
        )
        telemetry.append(
            {
                "epoch": epoch,
                "phase": "train",
                "region_mean": region,
                "birth_existence_raw_all_mean": (
                    birth_existence_last5 if epoch >= 25 else 1.35
                ),
                "birth_prominence_raw_all_mean": 0.04,
            }
        )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    telemetry_path.write_text(
        "\n".join(json.dumps(row) for row in telemetry) + "\n", encoding="utf-8"
    )
    return log_path, telemetry_path


def test_gate_accepts_semantic_improvement_and_noncollapsed_births(tmp_path: Path) -> None:
    log_path, telemetry_path = _write_gate_fixture(tmp_path)
    observed = evaluate_gate(log_path, telemetry_path)

    assert observed["status"] == "passed"
    assert observed["epochs"] == 30
    assert observed["last5_birth_existence_raw_mean"] == pytest.approx(1.30)
    assert observed["last5_region_median"] == pytest.approx(0.82)


def test_gate_rejects_the_observed_flat_field_signature(tmp_path: Path) -> None:
    log_path, telemetry_path = _write_gate_fixture(
        tmp_path, birth_existence_last5=1.5625
    )
    with pytest.raises(GateFailure, match="flat-field birth existence"):
        evaluate_gate(log_path, telemetry_path)


def test_gate_rejects_skipped_optimizer_steps(tmp_path: Path) -> None:
    log_path, telemetry_path = _write_gate_fixture(tmp_path, skipped_steps=1)
    with pytest.raises(GateFailure, match="skipped optimizer"):
        evaluate_gate(log_path, telemetry_path)

