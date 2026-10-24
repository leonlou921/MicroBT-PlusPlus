from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassMarginSurvivalTrainer import (
    MicroBTUnitMassMarginSurvivalTrainerScratch60,
)
from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassSurvivalTrainer import (
    MicroBTUnitMassSurvivalTrainerScratch60,
)


def test_margin_survival_changes_only_training_survival_target() -> None:
    assert issubclass(
        MicroBTUnitMassMarginSurvivalTrainerScratch60,
        MicroBTUnitMassSurvivalTrainerScratch60,
    )
    assert MicroBTUnitMassSurvivalTrainerScratch60.hard_survival_mass == 0.25
    assert MicroBTUnitMassMarginSurvivalTrainerScratch60.hard_survival_mass == 0.50
    criterion = MicroBTUnitMassMarginSurvivalTrainerScratch60._build_loss(
        MicroBTUnitMassMarginSurvivalTrainerScratch60.__new__(
            MicroBTUnitMassMarginSurvivalTrainerScratch60
        )
    )
    assert criterion.mass_conservation.hard_survival_mass == 0.50


def test_margin_launcher_contract_freezes_all_other_protocol_variables() -> None:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            "bash",
            str(root / "scripts" / "run_microbt_umctf_hsm05_scratch200.sh"),
            "--print-contract",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract == {
        "candidate_id": "UMCTF-HSM05-DA8",
        "causal_change": "decoder_aligned_survival_margin",
        "da_workers": 8,
        "da_workers_policy": "hard_locked",
        "decoder_min_atom_mass": 0.25,
        "formal_epochs": 200,
        "from_scratch": True,
        "full203_default": False,
        "isolated_root": "/path/to/example",
        "lr_schedule_horizon": 200,
        "pretrained": False,
        "random_seed": 42,
        "scientific_parent": "UMCTF-HS",
        "trainer": "MicroBTUnitMassMarginSurvivalTrainerScratch60",
        "training_survival_mass": 0.50,
    }

