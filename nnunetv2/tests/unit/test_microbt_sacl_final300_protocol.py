from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassTrainer import (
    ALLOWED_UMCTF_TARGET_EPOCHS,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSACLTrainer import (
    MicroBTUnitMassSACLFinal300Trainer,
    sacl_continuation_lambda,
    sacl_weights,
)


ROOT = Path(__file__).resolve().parents[3]


def test_umctf_target_epoch_contract_includes_final300() -> None:
    assert ALLOWED_UMCTF_TARGET_EPOCHS == (60, 200, 300)


def test_c2_final300_launcher_contract_is_from_scratch_horizon300() -> None:
    runner = ROOT / "scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh"
    completed = subprocess.run(
        ["bash", str(runner), "--print-contract"],
        check=True,
        text=True,
        capture_output=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["fold"] == "all"
    assert contract["formal_epochs"] == 300
    assert contract["lr_schedule_horizon"] == 300
    assert contract["from_scratch"] is True
    assert contract["test_case_count"] == 251
    assert contract["fixed8"] == "abolished"


def test_c2_final300_runner_saves_declared_formal_snapshots() -> None:
    text = (
        ROOT / "scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh"
    ).read_text(encoding="utf-8")
    for epoch in (60, 100, 150, 200, 250, 300):
        assert f"checkpoint_phase_{epoch:03d}.pth" in text
    assert MicroBTUnitMassSACLFinal300Trainer.continuation_snapshot_epochs == (
        100,
        150,
        200,
        250,
        300,
    )


def test_c2_final300_continuation_stays_at_frozen_endpoint_after_epoch_200() -> None:
    """The 300-epoch final run must keep the already-reached SACL endpoint."""
    assert sacl_continuation_lambda(199) == 1.0
    assert sacl_continuation_lambda(200) == 1.0
    assert sacl_continuation_lambda(299) == 1.0
    assert sacl_weights(299) == (0.25, 0.25)
    assert MicroBTUnitMassSACLFinal300Trainer.training_epochs == 300
    assert MicroBTUnitMassSACLFinal300Trainer.exact_h0_schedule(299, 0) == (
        "ctf_no_h0",
        False,
        0.0,
    )


def test_c2_final300_recovery_runner_requires_verified_epoch200_checkpoint() -> None:
    runner = ROOT / "scripts/continue_microbt_umctf_sacl_c2_final300_foldall.sh"
    completed = subprocess.run(
        ["bash", str(runner), "--print-contract"],
        check=True,
        text=True,
        capture_output=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["continue_training"] is True
    assert contract["required_current_epoch"] == 200
    assert contract["formal_epochs"] == 300
    text = runner.read_text(encoding="utf-8")
    assert "--continue-training" in text
    assert "checkpoint_latest.pth" in text
