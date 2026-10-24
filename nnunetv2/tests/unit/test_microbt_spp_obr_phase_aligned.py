from __future__ import annotations

import os

import pytest
import torch

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRPhaseAlignedTrainer import (
    MicroBTSPPOBRPhaseAlignedTrainerScratch200,
    PhaseAlignedPolyLRScheduler,
    phase_aligned_lr,
    reset_sgd_momentum_buffers,
)


def test_phase_aligned_lr_has_one_deliberate_drop_then_decays_to_zero() -> None:
    assert phase_aligned_lr(0) == pytest.approx(1e-2)
    assert phase_aligned_lr(14) > 0.009
    assert phase_aligned_lr(15) == pytest.approx(3e-3)
    assert phase_aligned_lr(16) < phase_aligned_lr(15)
    assert phase_aligned_lr(199) > 0.0
    assert phase_aligned_lr(200) == pytest.approx(0.0)


def test_phase_aligned_lr_rejects_epochs_outside_horizon() -> None:
    with pytest.raises(ValueError, match="epoch must be in"):
        phase_aligned_lr(-1)
    with pytest.raises(ValueError, match="epoch must be in"):
        phase_aligned_lr(201)


def test_scheduler_applies_literal_epoch_values() -> None:
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-2, momentum=0.99)
    scheduler = PhaseAlignedPolyLRScheduler(optimizer)
    for epoch in (0, 14, 15, 60, 199):
        scheduler.step(epoch)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            phase_aligned_lr(epoch)
        )


def test_momentum_reset_preserves_non_momentum_optimizer_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=1e-2, momentum=0.99)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    state = optimizer.state[parameter]
    state["sentinel"] = torch.tensor([7.0])

    cleared = reset_sgd_momentum_buffers(optimizer)

    assert cleared == 1
    assert "momentum_buffer" not in state
    torch.testing.assert_close(state["sentinel"], torch.tensor([7.0]))


def test_formal_trainer_contract_is_from_scratch_and_phase_aligned() -> None:
    trainer = MicroBTSPPOBRPhaseAlignedTrainerScratch200
    assert trainer.training_epochs == 200
    assert trainer.scratch_seed == 42
    assert trainer.phase_transition_epoch == 15
    assert trainer.phase_learning_rate == pytest.approx(3e-3)
    assert trainer.first_hard_gate_epoch == 60
    assert trainer.allowed_stop_epochs == (15, 30, 45, 60, 200)


def test_stop_epoch_parser_accepts_only_preregistered_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MICROBT_PHASE_ALIGNED_STOP_EPOCHS", "45")
    assert MicroBTSPPOBRPhaseAlignedTrainerScratch200.requested_stop_epoch() == 45
    monkeypatch.setenv("MICROBT_PHASE_ALIGNED_STOP_EPOCHS", "44")
    with pytest.raises(ValueError, match="15, 30, 45, 60, or 200"):
        MicroBTSPPOBRPhaseAlignedTrainerScratch200.requested_stop_epoch()
    monkeypatch.delenv("MICROBT_PHASE_ALIGNED_STOP_EPOCHS", raising=False)
    assert MicroBTSPPOBRPhaseAlignedTrainerScratch200.requested_stop_epoch() == 200

