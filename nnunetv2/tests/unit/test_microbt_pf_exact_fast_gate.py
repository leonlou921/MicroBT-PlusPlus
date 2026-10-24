from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nnunetv2.scripts.microbt_pf_exact_fast_gate import evaluate_exact_fast_gate


def _passing_payload(epoch_seconds: float = 300.0) -> dict:
    return {
        "status": "complete",
        "topology_hashes": {"precomputed": "abc", "reference": "abc"},
        "hundred_step": {
            "steps": 100,
            "exact_h0_active_count": 100,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
        },
        "full_epoch": {
            "training_iterations": 250,
            "validation_iterations": 50,
            "exact_h0_train_calls": 250,
            "validation_exact_h0_calls": 5,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
            "wall_seconds": epoch_seconds,
        },
        "protected_state_unchanged": True,
    }


@pytest.mark.parametrize("epoch_seconds", [300.0, 360.0])
def test_gate_admits_literal_complete_payload_at_or_below_limit(epoch_seconds) -> None:
    decision = evaluate_exact_fast_gate(_passing_payload(epoch_seconds))
    assert decision == {"admitted": True, "reasons": []}


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: payload["hundred_step"].update(exact_h0_active_count=99), "hundred_step_h0_incomplete"),
        (lambda payload: payload["full_epoch"].update(optimizer_skips=1), "full_epoch_optimizer_skip"),
        (lambda payload: payload["topology_hashes"].update(precomputed="different"), "topology_hash_mismatch"),
        (lambda payload: payload.update(protected_state_unchanged=False), "protected_state_drift"),
        (lambda payload: payload["full_epoch"].update(wall_seconds=360.0001), "epoch_wall_exceeds_360_seconds"),
    ],
)
def test_gate_rejects_each_scientific_or_safety_violation(mutation, reason) -> None:
    payload = deepcopy(_passing_payload())
    mutation(payload)
    decision = evaluate_exact_fast_gate(payload)
    assert decision["admitted"] is False
    assert reason in decision["reasons"]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_gate_launcher_safety_stops_before_runner(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    guard = tmp_path / "guard.sh"
    runner = tmp_path / "runner.sh"
    marker = tmp_path / "runner-called"
    guard_marker = tmp_path / "guard-called"
    guard.write_text(
        f"from pathlib import Path\nPath({str(guard_marker)!r}).touch()\nraise SystemExit(2)\n",
        encoding="utf-8",
    )
    _write_executable(runner, f"touch {marker!s}\n")
    environment = os.environ.copy()
    for inherited_name in (
        "MICROBT_PF_EXACT_FAST_GATE_ARTIFACT",
        "MICROBT_PF_CACHE",
        "nnUNet_results",
    ):
        environment.pop(inherited_name, None)
    environment.update(
        {
            "MICROBT_PF_ISOLATED_ROOT": str(repository.parent),
            "MICROBT_PF_GUARD": str(guard),
            "MICROBT_PF_GATE_RUNNER": str(runner),
            "MICROBT_PF_BASELINE": str(tmp_path / "baseline.json"),
            "MICROBT_PF_PYTHON": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(repository / "scripts" / "run_microbt_pf_exact_fast_gate.sh")],
        cwd=repository,
        env=environment,
        check=False,
    )

    assert completed.returncode == 2
    assert guard_marker.exists()
    assert not marker.exists()
