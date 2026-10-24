from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from nnunetv2.inference.microbt_umctf_predictor import (
    raw_logits_to_umctf,
)
from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassTrainer import (
    MicroBTUnitMassTrainerScratch60,
)


def test_raw_seven_channel_inference_contract() -> None:
    output = raw_logits_to_umctf(torch.zeros((7, 3, 3, 3)))
    assert output["transport"].shape == (3, 3, 3, 3)
    assert output["region_probabilities"].shape == (3, 3, 3, 3)
    assert output["object_density"].shape == (1, 3, 3, 3)


def test_scratch_trainers_share_one_200_epoch_schedule() -> None:
    assert MicroBTUnitMassTrainerScratch60.scratch_seed == 42
    assert MicroBTUnitMassTrainerScratch60.causal_epochs == 60
    assert MicroBTUnitMassTrainerScratch60.training_epochs == 200
    assert MicroBTUnitMassTrainerScratch60.snapshot_epochs == (15, 30, 40, 50, 60)
    assert MicroBTUnitMassTrainerScratch60.continuation_snapshot_epochs == (100, 150, 200)


def test_launcher_contract_is_isolated_scratch_without_pretraining() -> None:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["bash", str(root / "scripts" / "run_microbt_umctf_scratch60.sh"), "--print-contract"],
        check=True, capture_output=True, text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["isolated_root"] == "/path/to/example"
    assert contract["trainer"] == "MicroBTUnitMassTrainerScratch60"
    assert contract["from_scratch"] is True
    assert contract["pretrained"] is False
    assert contract["full203_default"] is False
    assert contract["auto_continue_to200_on_pass"] is True


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
