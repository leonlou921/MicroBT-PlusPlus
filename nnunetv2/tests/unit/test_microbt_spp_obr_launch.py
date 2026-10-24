from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.MicroBTSPPTrainer import (
    MicroBTSPPOBRTrainerScratch200,
    MicroBTSPPOBRTrainerScratchGate5,
)


ROOT = Path(__file__).resolve().parents[3]


def test_scratch_trainer_uses_the_full_200_epoch_optimization_contract() -> None:
    assert MicroBTSPPOBRTrainerScratch200.training_epochs == 200
    assert MicroBTSPPOBRTrainerScratch200.initial_learning_rate == pytest.approx(1e-2)
    assert MicroBTSPPOBRTrainerScratch200.sparse_exact_interval == 10
    assert MicroBTSPPOBRTrainerScratch200.sparse_h0_weight == 10.0
    assert MicroBTSPPOBRTrainerScratch200.full_exact_tail_epochs == 0
    assert MicroBTSPPOBRTrainerScratch200.scratch_seed == 42
    assert MicroBTSPPOBRTrainerScratch200.object_morse_weight(0) == 0.0
    assert MicroBTSPPOBRTrainerScratch200.object_morse_weight(19) == 1.0


def test_obr_gate_is_five_epochs_with_the_same_200_epoch_lr_horizon() -> None:
    trainer = object.__new__(MicroBTSPPOBRTrainerScratchGate5)
    trainer.network = nn.Linear(2, 2)
    trainer.initial_lr = 1e-2
    trainer.weight_decay = 3e-5
    _, scheduler = trainer.configure_optimizers()

    assert trainer.gate_epochs == 5
    assert trainer.training_epochs == 200
    assert scheduler.max_steps == 200


def test_obr_launcher_shell_public_contract_and_safety_guards() -> None:
    launcher = ROOT / "scripts" / "run_microbt_spp_obr_scratch200.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(launcher)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run(
        ["bash", str(launcher), "--print-contract"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "auto_full203": True,
        "formal_epochs": 200,
        "formal_trainer": "MicroBTSPPOBRTrainerScratch200",
        "from_scratch": True,
        "fresh_optimizer": True,
        "gate_epochs": 5,
        "gate_trainer": "MicroBTSPPOBRTrainerScratchGate5",
        "isolated_root": "/path/to/example",
        "random_seed": 42,
    }

    text = launcher.read_text(encoding="utf-8")
    assert "refusing to overwrite" in text
    assert "active_fold0_trainer" in text
    assert "source_checkpoint" not in text
    assert "MICROBT_SPP_EXACT_INIT_CHECKPOINT" not in text
    assert "-pretrained_weights" not in text
    assert text.count("run_microbt_pf_full203.py") == 1
