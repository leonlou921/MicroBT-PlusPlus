from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nnunetv2.scripts.microbt_pf_scheduled_exact_gate import (
    evaluate_scheduled_exact_gate,
    target_topology_hashes,
)
from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget


def _passing_payload(epoch_seconds: float = 330.0) -> dict:
    return {
        "status": "complete",
        "schedule": {
            "num_epochs": 200,
            "tail_start_epoch": 197,
            "sparse_interval": 10,
            "sparse_h0_weight": 10.0,
            "tail_h0_weight": 1.0,
        },
        "topology_hashes": {"precomputed": "abc", "reference": "abc"},
        "hundred_step": {
            "steps": 100,
            "schedule_mode": "sparse_exact",
            "exact_h0_active_count": 10,
            "schedule_skip_count": 90,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
        },
        "full_epoch": {
            "training_iterations": 250,
            "validation_iterations": 50,
            "schedule_mode": "sparse_exact",
            "exact_h0_train_calls": 25,
            "schedule_skip_count": 225,
            "validation_exact_h0_calls": 5,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
            "wall_seconds": epoch_seconds,
        },
        "tail_step": {
            "epoch": 197,
            "schedule_mode": "full_exact",
            "effective_h0_weight": 1.0,
            "exact_h0_active_count": 1,
            "schedule_skip_count": 0,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
            "finite_loss": True,
        },
        "protected_state_unchanged": True,
    }


@pytest.mark.parametrize("epoch_seconds", [300.0, 360.0])
def test_scheduled_gate_admits_complete_payload_at_or_below_limit(
    epoch_seconds: float,
) -> None:
    assert evaluate_scheduled_exact_gate(_passing_payload(epoch_seconds)) == {
        "admitted": True,
        "reasons": [],
    }


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda p: p.update(status="running"), "run_incomplete"),
        (
            lambda p: p["schedule"].update(sparse_interval=8),
            "schedule_contract_mismatch",
        ),
        (
            lambda p: p["topology_hashes"].update(precomputed="different"),
            "topology_hash_mismatch",
        ),
        (
            lambda p: p["hundred_step"].update(exact_h0_active_count=9),
            "hundred_step_h0_count",
        ),
        (
            lambda p: p["hundred_step"].update(schedule_skip_count=89),
            "hundred_step_schedule_skip_count",
        ),
        (
            lambda p: p["full_epoch"].update(exact_h0_train_calls=24),
            "full_epoch_h0_count",
        ),
        (
            lambda p: p["full_epoch"].update(schedule_skip_count=224),
            "full_epoch_schedule_skip_count",
        ),
        (
            lambda p: p["full_epoch"].update(validation_exact_h0_calls=4),
            "validation_h0_audit_count",
        ),
        (
            lambda p: p["tail_step"].update(effective_h0_weight=10.0),
            "tail_step_contract",
        ),
        (
            lambda p: p["hundred_step"].update(nonfinite_steps=1),
            "hundred_step_nonfinite",
        ),
        (
            lambda p: p["full_epoch"].update(optimizer_skips=1),
            "full_epoch_optimizer_skip",
        ),
        (
            lambda p: p["full_epoch"].update(wall_seconds=360.0001),
            "epoch_wall_exceeds_360_seconds",
        ),
        (
            lambda p: p.update(protected_state_unchanged=False),
            "protected_state_drift",
        ),
    ],
)
def test_scheduled_gate_rejects_each_scientific_or_safety_violation(
    mutation, reason: str
) -> None:
    payload = deepcopy(_passing_payload())
    mutation(payload)
    decision = evaluate_scheduled_exact_gate(payload)
    assert decision["admitted"] is False
    assert reason in decision["reasons"]


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_scheduled_gate_launcher_stops_before_runner_when_guard_rejects(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    guard = tmp_path / "guard.py"
    runner = tmp_path / "runner.sh"
    runner_marker = tmp_path / "runner-called"
    guard_marker = tmp_path / "guard-called"
    guard.write_text(
        f"from pathlib import Path\nPath({str(guard_marker)!r}).touch()\nraise SystemExit(2)\n",
        encoding="utf-8",
    )
    _write_executable(runner, f"touch {runner_marker!s}\n")
    environment = os.environ.copy()
    for inherited in (
        "MICROBT_PF_SCHEDULED_GATE_ARTIFACT",
        "MICROBT_PF_CACHE",
        "nnUNet_results",
    ):
        environment.pop(inherited, None)
    environment.update(
        {
            "MICROBT_PF_ISOLATED_ROOT": str(repository.parent),
            "MICROBT_PF_GUARD": str(guard),
            "MICROBT_PF_SCHEDULED_GATE_RUNNER": str(runner),
            "MICROBT_PF_BASELINE": str(tmp_path / "baseline.json"),
            "MICROBT_PF_PYTHON": sys.executable,
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(repository / "scripts" / "run_microbt_pf_scheduled_exact_gate.sh"),
        ],
        cwd=repository,
        env=environment,
        check=False,
    )

    assert completed.returncode == 2
    assert guard_marker.exists()
    assert not runner_marker.exists()


def test_target_topology_hashes_materializes_an_uncomputed_dense_target() -> None:
    shape = (3, 3, 3)
    target = PersistentFieldTarget(
        field=torch.linspace(-1.0, 1.0, 27).reshape(shape),
        instance_ids=torch.zeros(shape, dtype=torch.int32),
        dense_valid_mask=torch.ones(shape, dtype=torch.bool),
        valid_instance_ids=(),
        centers_zyx=(),
        topology=None,
        precompute_topology=False,
    )

    hashes = target_topology_hashes(target)

    assert hashes["precomputed"] == hashes["reference"]
    assert target.topology is None


def test_scheduled_gate_defaults_to_eight_augmentation_workers() -> None:
    repository = Path(__file__).resolve().parents[3]
    text = (
        repository / "scripts" / "run_microbt_pf_scheduled_exact_gate.sh"
    ).read_text(encoding="utf-8")
    assert 'nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"' in text
