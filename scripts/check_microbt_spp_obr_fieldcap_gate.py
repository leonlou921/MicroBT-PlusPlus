from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from check_microbt_spp_obr_h0off_gate import GateFailure, evaluate_gate
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRFieldCapCausalTrainer import (
    field_cap_curriculum_weights,
)


class FieldCapContractFailure(GateFailure):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FieldCapContractFailure(message)


def audit_fieldcap_contract(log_path: Path) -> dict:
    pattern = re.compile(
        r"SPP-OBR field-cap causal contract: epoch=(\d+), "
        r"field=([0-9.]+), birth=([0-9.]+), h0=([0-9.]+), "
        r"exact_h0=disabled"
    )
    rows = {}
    for match in pattern.finditer(Path(log_path).read_text(encoding="utf-8", errors="replace")):
        epoch = int(match.group(1))
        _require(epoch not in rows, f"duplicate field-cap marker at epoch {epoch}")
        rows[epoch] = tuple(float(match.group(index)) for index in (2, 3, 4))
    _require(set(rows) == set(range(60)), "field-cap contract must have epochs 0-59")
    for epoch in range(60):
        field, birth, h0 = rows[epoch]
        expected = field_cap_curriculum_weights(epoch)
        _require(abs(field - expected.field) <= 5e-7, f"field cap drift at epoch {epoch}")
        _require(abs(birth - expected.birth) <= 5e-7, f"birth curriculum drift at epoch {epoch}")
        _require(abs(h0) <= 5e-7, f"H0 contract drift at epoch {epoch}")
    return {
        "birth_curriculum": "unchanged",
        "field_cap": 0.5,
        "field_cap_start_epoch": 30,
        "h0": 0.0,
        "marker_count": len(rows),
    }


def evaluate_fieldcap_gate(
    log_path: Path,
    telemetry_path: Path,
    probe45_path: Path,
    probe60_path: Path,
) -> dict:
    contract = audit_fieldcap_contract(log_path)
    payload = evaluate_gate(log_path, telemetry_path, probe45_path, probe60_path)
    payload["fieldcap_contract"] = contract
    return payload


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--probe-45", type=Path, required=True)
    parser.add_argument("--probe-60", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = evaluate_fieldcap_gate(
            args.log, args.telemetry, args.probe_45, args.probe_60
        )
    except GateFailure as error:
        payload = {
            "status": "failed",
            "reason": str(error),
            "from_scratch": True,
            "exact_h0": False,
            "field_cap": 0.5,
            "field_cap_start_epoch": 30,
            "birth_curriculum": "unchanged",
        }
        _write(args.output, payload)
        print("SPP_OBR_FIELDCAP_CAUSAL_GATE_FAILED", json.dumps(payload, sort_keys=True))
        raise SystemExit(1)
    _write(args.output, payload)
    print("SPP_OBR_FIELDCAP_CAUSAL_GATE_OK", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
