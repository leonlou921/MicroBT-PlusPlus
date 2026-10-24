from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from check_microbt_spp_obr_fieldcap_gate import (  # noqa: E402
    FieldCapContractFailure,
    audit_fieldcap_contract,
)


def _contract_log() -> str:
    lines = ["SPP scratch initialization: checkpoint=none"]
    for epoch in range(60):
        if epoch <= 4:
            field, birth = 0.0, 0.0
        elif epoch <= 14:
            progress = (epoch - 5) / 9
            field = 0.025 + progress * 0.225
            birth = 0.005 + progress * 0.045
        elif epoch <= 29:
            progress = (epoch - 14) / 15
            field = 0.25 + progress * 0.25
            birth = 0.05 + progress * 0.15
        else:
            progress = (epoch - 29) / 30
            field = 0.5
            birth = 0.2 + progress * 0.8
        lines.append(
            "SPP-OBR field-cap causal contract: "
            f"epoch={epoch}, field={field:.6f}, birth={birth:.6f}, "
            "h0=0.000000, exact_h0=disabled"
        )
    return "\n".join(lines) + "\n"


def test_fieldcap_contract_audit_accepts_the_only_allowed_schedule(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(_contract_log(), encoding="utf-8")
    result = audit_fieldcap_contract(log)
    assert result == {
        "birth_curriculum": "unchanged",
        "field_cap": 0.5,
        "field_cap_start_epoch": 30,
        "h0": 0.0,
        "marker_count": 60,
    }


def test_fieldcap_contract_audit_rejects_field_drift(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    text = _contract_log().replace(
        "epoch=30, field=0.500000", "epoch=30, field=0.516667"
    )
    log.write_text(text, encoding="utf-8")
    with pytest.raises(FieldCapContractFailure, match="field cap"):
        audit_fieldcap_contract(log)


def test_fieldcap_contract_audit_rejects_birth_drift(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    text = _contract_log().replace(
        "epoch=45, field=0.500000, birth=0.626667",
        "epoch=45, field=0.500000, birth=0.500000",
    )
    log.write_text(text, encoding="utf-8")
    with pytest.raises(FieldCapContractFailure, match="birth curriculum"):
        audit_fieldcap_contract(log)


def test_failed_gate_persists_a_machine_readable_manifest(tmp_path: Path) -> None:
    log = tmp_path / "bad.log"
    log.write_text("SPP scratch initialization: checkpoint=none\n", encoding="utf-8")
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_microbt_spp_obr_fieldcap_gate.py"),
            "--log",
            str(log),
            "--telemetry",
            str(tmp_path / "missing.jsonl"),
            "--probe-45",
            str(tmp_path / "missing45.csv"),
            "--probe-60",
            str(tmp_path / "missing60.csv"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "birth_curriculum": "unchanged",
        "exact_h0": False,
        "field_cap": 0.5,
        "field_cap_start_epoch": 30,
        "from_scratch": True,
        "reason": "field-cap contract must have epochs 0-59",
        "status": "failed",
    }
