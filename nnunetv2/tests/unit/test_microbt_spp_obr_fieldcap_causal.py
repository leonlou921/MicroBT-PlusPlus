from __future__ import annotations

import pytest

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    curriculum_weights,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRFieldCapCausalTrainer import (
    MicroBTSPPOBRFieldCapCausalTrainerScratch60,
    field_cap_curriculum_weights,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRH0OffCausalTrainer import (
    h0off_curriculum_weights,
)


def test_field_cap_changes_nothing_before_epoch_30() -> None:
    for epoch in (0, 15, 29):
        assert field_cap_curriculum_weights(epoch) == h0off_curriculum_weights(epoch)


def test_field_is_capped_and_birth_is_unchanged_after_epoch_29() -> None:
    for epoch in (30, 45, 59):
        actual = field_cap_curriculum_weights(epoch)
        original = curriculum_weights(epoch)
        assert actual.field == pytest.approx(0.5)
        assert actual.birth == pytest.approx(original.birth)
        assert actual.h0 == pytest.approx(0.0)


def test_field_cap_trainer_preserves_the_h0off_causal_contract() -> None:
    trainer = MicroBTSPPOBRFieldCapCausalTrainerScratch60
    assert trainer.causal_epochs == 60
    assert trainer.field_cap_start_epoch == 30
    assert trainer.field_cap == pytest.approx(0.5)
    assert trainer.snapshot_epochs == (15, 30, 35, 40, 45, 50, 55, 60)
    for epoch in (0, 30, 59):
        assert trainer.exact_h0_schedule(epoch, 0) == (
            "h0_off_causal",
            False,
            0.0,
        )

