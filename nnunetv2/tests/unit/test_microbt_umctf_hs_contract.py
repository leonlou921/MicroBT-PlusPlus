from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from nnunetv2.inference.microbt_umctf_predictor import (
    raw_logits_to_umctf,
)
from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassSurvivalTrainer import (
    MicroBTUnitMassSurvivalTrainerScratch60,
)


def test_raw_seven_channel_inference_contract() -> None:
    output = raw_logits_to_umctf(torch.zeros((7, 3, 3, 3)))
    assert output["transport"].shape == (3, 3, 3, 3)
    assert output["region_probabilities"].shape == (3, 3, 3, 3)
    assert output["object_density"].shape == (1, 3, 3, 3)


def test_scratch_trainers_share_one_200_epoch_schedule() -> None:
    assert MicroBTUnitMassSurvivalTrainerScratch60.scratch_seed == 42
    assert MicroBTUnitMassSurvivalTrainerScratch60.causal_epochs == 60
    assert MicroBTUnitMassSurvivalTrainerScratch60.training_epochs == 200
    assert MicroBTUnitMassSurvivalTrainerScratch60.snapshot_epochs == (15, 30, 40, 50, 60)
    assert MicroBTUnitMassSurvivalTrainerScratch60.continuation_snapshot_epochs == (100, 150, 200)
    assert MicroBTUnitMassSurvivalTrainerScratch60.hard_survival_mass == 0.25


def test_launcher_contract_is_isolated_scratch_without_pretraining() -> None:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["bash", str(root / "scripts" / "run_microbt_umctf_hs_scratch200.sh"), "--print-contract"],
        check=True, capture_output=True, text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["isolated_root"] == "/path/to/example"
    assert contract["trainer"] == "MicroBTUnitMassSurvivalTrainerScratch60"
    assert contract["causal_change"] == "decoder_aligned_hard_object_survival"
    assert contract["hard_survival_mass"] == 0.25
    assert contract["formal_epochs"] == 200
    assert contract["lr_schedule_horizon"] == 200
    assert contract["da_workers"] == 8
    assert contract["da_workers_policy"] == "hard_locked"
    assert contract["from_scratch"] is True
    assert contract["pretrained"] is False
    assert contract["full203_default"] is False


def test_fixed8_contract_freezes_four_comparable_snapshots() -> None:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["bash", str(root / "scripts" / "run_umctf_hs_fixed8_snapshots.sh"), "--print-contract"],
        check=True, capture_output=True, text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract == {
        "cases": 8,
        "checkpoint_epochs": [60, 100, 150, 200],
        "count_mae_lt": 15,
        "evaluator": "evaluate_microbt_umctf_full_gate.py",
        "frozen_thresholds": True,
        "trainer": "MicroBTUnitMassSurvivalTrainerScratch60",
    }


def test_gate_error_pattern_does_not_count_zero_nonfinite_telemetry() -> None:
    from scripts.evaluate_microbt_umctf_stage_gate import (
        count_real_log_errors,
        scored_instance_boxes,
    )

    assert count_real_log_errors('"nonfinite_steps": 0\n"skipped_steps": 0') == 0
    assert count_real_log_errors("Traceback (most recent call last):") == 1
    assert count_real_log_errors('"nonfinite_steps": 2') == 2
    labels = torch.zeros((4, 4, 4), dtype=torch.int32).numpy()
    labels[2:4, 1:3, 1:3] = 2
    boxes = scored_instance_boxes(labels, (0.25, 0.75))
    assert boxes == [((2, 1, 1, 4, 3, 3), 0.75)]
