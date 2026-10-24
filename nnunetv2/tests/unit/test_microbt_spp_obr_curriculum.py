from __future__ import annotations

import pytest
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    CurriculumWeights,
    MicroBTSPPOBRCurriculumTrainerScratch200,
    _MicroBTSPPOBRCurriculumCriterion,
    curriculum_weights,
)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    (
        (0, CurriculumWeights(0.0, 0.0, 0.0)),
        (4, CurriculumWeights(0.0, 0.0, 0.0)),
        (5, CurriculumWeights(0.025, 0.005, 0.0)),
        (14, CurriculumWeights(0.25, 0.05, 0.0)),
        (15, CurriculumWeights(0.2666666667, 0.06, 0.0)),
        (29, CurriculumWeights(0.5, 0.2, 0.0)),
        (30, CurriculumWeights(0.5166666667, 0.2266666667, 0.0333333333)),
        (59, CurriculumWeights(1.0, 1.0, 1.0)),
        (199, CurriculumWeights(1.0, 1.0, 1.0)),
    ),
)
def test_curriculum_enters_field_then_obr_then_exact_h0(
    epoch: int, expected: CurriculumWeights
) -> None:
    observed = curriculum_weights(epoch)
    assert observed.field == pytest.approx(expected.field)
    assert observed.birth == pytest.approx(expected.birth)
    assert observed.h0 == pytest.approx(expected.h0)


def test_curriculum_rejects_epochs_outside_the_formal_horizon() -> None:
    with pytest.raises(ValueError, match="epoch must be in"):
        curriculum_weights(-1)
    with pytest.raises(ValueError, match="epoch must be in"):
        curriculum_weights(200)


def test_component_weights_do_not_change_loss_formulas() -> None:
    values = {
        "loss": torch.tensor(99.0),
        "region": torch.tensor(2.0),
        "field": torch.tensor(3.0),
        "betti_0": torch.tensor(5.0),
        "false_persistence": torch.tensor(1.0),
        "birth_reservation": torch.tensor(7.0),
        "birth_existence": torch.tensor(4.0),
        "birth_prominence": torch.tensor(3.0),
        "object_morse": torch.tensor(15.0),
        "persistent_field": torch.tensor(15.0),
    }
    observed = _MicroBTSPPOBRCurriculumCriterion._apply_curriculum_weights(
        values, CurriculumWeights(field=0.5, birth=0.2, h0=0.0)
    )

    torch.testing.assert_close(observed["field_raw"], torch.tensor(3.0))
    torch.testing.assert_close(observed["field"], torch.tensor(1.5))
    torch.testing.assert_close(observed["birth_reservation_raw"], torch.tensor(7.0))
    torch.testing.assert_close(observed["birth_reservation"], torch.tensor(1.4))
    torch.testing.assert_close(observed["birth_existence"], torch.tensor(0.8))
    torch.testing.assert_close(observed["birth_prominence"], torch.tensor(0.6))
    torch.testing.assert_close(observed["betti_0"], torch.tensor(5.0))
    torch.testing.assert_close(observed["object_morse"], torch.tensor(7.9))
    torch.testing.assert_close(observed["loss"], torch.tensor(9.9))


def test_exact_h0_is_disabled_until_epoch_30_then_unbiased() -> None:
    trainer = MicroBTSPPOBRCurriculumTrainerScratch200
    assert trainer.exact_h0_schedule(29, 0) == ("curriculum_dense", False, 0.0)
    assert trainer.exact_h0_schedule(30, 1) == ("sparse_exact", False, 0.0)

    mode, active, weight = trainer.exact_h0_schedule(30, 0)
    assert mode == "sparse_exact"
    assert active is True
    assert weight == pytest.approx(10.0 / 30.0)

    assert trainer.exact_h0_schedule(59, 0) == ("sparse_exact", True, 10.0)
    assert trainer.exact_h0_schedule(60, 0) == ("sparse_exact", True, 10.0)


def test_formal_trainer_preserves_scratch_optimization_contract() -> None:
    trainer = MicroBTSPPOBRCurriculumTrainerScratch200
    assert trainer.training_epochs == 200
    assert trainer.initial_learning_rate == pytest.approx(1e-2)
    assert trainer.scratch_seed == 42
    assert trainer.sparse_exact_interval == 10
    assert trainer.full_exact_tail_epochs == 0


def test_gate_run_keeps_the_200_epoch_lr_horizon() -> None:
    trainer = object.__new__(MicroBTSPPOBRCurriculumTrainerScratch200)
    trainer.network = nn.Linear(2, 2)
    trainer.initial_lr = 1e-2
    trainer.weight_decay = 3e-5
    _, scheduler = trainer.configure_optimizers()

    assert scheduler.max_steps == 200
