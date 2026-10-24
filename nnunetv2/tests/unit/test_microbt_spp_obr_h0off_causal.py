from __future__ import annotations

import pytest

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    curriculum_weights,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRH0OffCausalTrainer import (
    MicroBTSPPOBRH0OffCausalTrainerScratch60,
    phase_checkpoint_name,
)


def test_exact_h0_is_inactive_for_the_entire_causal_run() -> None:
    trainer = MicroBTSPPOBRH0OffCausalTrainerScratch60
    for epoch in (0, 29, 30, 45, 59):
        assert trainer.exact_h0_schedule(epoch, 0) == ("h0_off_causal", False, 0.0)
        assert trainer.exact_h0_schedule(epoch, 249) == (
            "h0_off_causal",
            False,
            0.0,
        )


def test_field_and_birth_curriculum_remain_unchanged() -> None:
    assert curriculum_weights(30).field == pytest.approx(0.5166666667)
    assert curriculum_weights(30).birth == pytest.approx(0.2266666667)
    assert curriculum_weights(59).field == pytest.approx(1.0)
    assert curriculum_weights(59).birth == pytest.approx(1.0)


def test_causal_trainer_contract_is_continuous_and_from_scratch() -> None:
    trainer = MicroBTSPPOBRH0OffCausalTrainerScratch60
    assert trainer.causal_epochs == 60
    assert trainer.training_epochs == 200
    assert trainer.scratch_seed == 42
    assert trainer.phase_transition_epoch == 15
    assert trainer.phase_learning_rate == pytest.approx(3e-3)
    assert trainer.snapshot_epochs == (15, 30, 35, 40, 45, 50, 55, 60)


def test_phase_checkpoint_names_are_canonical() -> None:
    assert phase_checkpoint_name(15) == "checkpoint_phase_015.pth"
    assert phase_checkpoint_name(60) == "checkpoint_phase_060.pth"
    with pytest.raises(ValueError, match="completed epoch"):
        phase_checkpoint_name(0)

