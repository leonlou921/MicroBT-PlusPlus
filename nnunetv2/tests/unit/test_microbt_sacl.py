from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_sacl_losses import MicroBTSACLLoss
from nnunetv2.training.nnUNetTrainer.MicroBTSACLTrainer import (
    MicroBTUnitMassSACLTrainerScratch60,
    sacl_continuation_lambda,
    sacl_weights,
)


def _target(ids: torch.Tensor) -> PersistentFieldTarget:
    valid = tuple(int(value) for value in torch.unique(ids).tolist() if value > 0)
    centers = tuple((1, 1, 1) for _ in valid)
    return PersistentFieldTarget(
        field=torch.zeros_like(ids, dtype=torch.float32),
        instance_ids=ids.to(torch.int32),
        dense_valid_mask=torch.ones_like(ids, dtype=torch.bool),
        valid_instance_ids=valid,
        centers_zyx=centers,
        precompute_topology=False,
    )


def test_sacl_continuation_preserves_b0_endpoint_and_delays_transport() -> None:
    assert sacl_continuation_lambda(0) == 0.0
    assert sacl_continuation_lambda(4) == 0.0
    assert sacl_continuation_lambda(29) == 1.0
    assert sacl_continuation_lambda(199) == 1.0
    measure_5, transport_5 = sacl_weights(5)
    assert measure_5 == 0.01
    assert transport_5 == 0.0004
    for epoch in range(200):
        measure, transport = sacl_weights(epoch)
        assert 0.0 <= transport <= measure <= 0.25
    assert sacl_weights(29) == (0.25, 0.25)


def test_sacl_loss_weights_measure_and_transport_independently() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    ids[1, 1, 1] = 1
    target = _target(ids)
    output = {
        "transport": torch.zeros((1, 3, 3, 3, 3), requires_grad=True),
        "region_probabilities": torch.full((1, 3, 3, 3, 3), 0.5, requires_grad=True),
        "object_density": torch.full((1, 1, 3, 3, 3), 0.1, requires_grad=True),
    }
    region_target = torch.zeros((1, 3, 3, 3, 3))
    criterion = MicroBTSACLLoss(
        measure_weight=0.25,
        transport_weight=0.0,
        hard_survival_mass=0.25,
    )
    result = criterion(output, region_target, [target])
    torch.testing.assert_close(
        result["loss"],
        result["region"] + 0.25 * result["mass_conservation_raw"],
    )
    torch.testing.assert_close(result["transport_assignment_weighted"], torch.tensor(0.0))
    assert result["mass_conservation_weighted"].item() > 0

    criterion.set_sacl_weights(measure_weight=0.0, transport_weight=0.25)
    result = criterion(output, region_target, [target])
    torch.testing.assert_close(
        result["loss"],
        result["region"] + 0.25 * result["transport_assignment_raw"],
    )
    torch.testing.assert_close(result["mass_conservation_weighted"], torch.tensor(0.0))


def test_sacl_trainer_freezes_b0_scientific_contract() -> None:
    trainer = MicroBTUnitMassSACLTrainerScratch60
    assert trainer.hard_survival_mass == 0.25
    assert trainer.scratch_seed == 42
    assert trainer.training_epochs == 200
    assert trainer.continuation_snapshot_epochs == (100, 150, 200)
    assert trainer.sacl_schedule == "b0_linear_measure_quadratic_transport"


def test_sacl_formal_launcher_freezes_matched_c2_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["nnUNet_n_proc_DA"] = "2"
    completed = subprocess.run(
        [
            "bash",
            str(root / "scripts" / "run_microbt_umctf_sacl_c2_scratch200.sh"),
            "--print-contract",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    contract = json.loads(completed.stdout)
    assert contract == {
        "architecture_carrier": "B0",
        "causal_change": "support_to_atom_continuation_learning",
        "da_workers": 8,
        "da_workers_policy": "hard_locked",
        "decoder_min_atom_mass": 0.25,
        "formal_epochs": 200,
        "from_scratch": True,
        "hard_survival_mass": 0.25,
        "lr_schedule_horizon": 200,
        "measure_weight": "0.25*lambda",
        "pretrained": False,
        "random_seed": 42,
        "schedule": "b0_linear_measure_quadratic_transport",
        "transport_weight": "0.25*lambda^2",
        "trainer": "MicroBTUnitMassSACLTrainerScratch60",
    }
