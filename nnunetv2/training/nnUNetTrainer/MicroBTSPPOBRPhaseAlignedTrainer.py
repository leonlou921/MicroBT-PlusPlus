from __future__ import annotations

import os

import torch
from torch.optim.lr_scheduler import _LRScheduler

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    MicroBTSPPOBRCurriculumTrainerScratch200,
)


def phase_aligned_lr(
    epoch: int,
    *,
    initial_lr: float = 1e-2,
    phase_lr: float = 3e-3,
    transition_epoch: int = 15,
    max_epochs: int = 200,
    exponent: float = 0.9,
) -> float:
    """Original scratch decay through epoch 14, then a new phase decay."""
    epoch = int(epoch)
    if not 0 <= epoch <= int(max_epochs):
        raise ValueError(f"epoch must be in [0, {int(max_epochs)}]")
    if epoch < int(transition_epoch):
        return float(initial_lr) * (1.0 - epoch / float(max_epochs)) ** float(
            exponent
        )
    phase_horizon = int(max_epochs) - int(transition_epoch)
    phase_step = epoch - int(transition_epoch)
    return float(phase_lr) * (1.0 - phase_step / float(phase_horizon)) ** float(
        exponent
    )


class PhaseAlignedPolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float = 1e-2,
        phase_lr: float = 3e-3,
        transition_epoch: int = 15,
        max_epochs: int = 200,
        exponent: float = 0.9,
        current_step: int | None = None,
    ) -> None:
        self.initial_lr = float(initial_lr)
        self.phase_lr = float(phase_lr)
        self.transition_epoch = int(transition_epoch)
        self.max_epochs = int(max_epochs)
        self.exponent = float(exponent)
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step: int | None = None) -> None:
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        new_lr = phase_aligned_lr(
            int(current_step),
            initial_lr=self.initial_lr,
            phase_lr=self.phase_lr,
            transition_epoch=self.transition_epoch,
            max_epochs=self.max_epochs,
            exponent=self.exponent,
        )
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        return self._last_lr


def reset_sgd_momentum_buffers(optimizer: torch.optim.Optimizer) -> int:
    """Clear only SGD momentum memory; preserve parameters and other state."""
    cleared = 0
    for state in optimizer.state.values():
        if "momentum_buffer" in state:
            state.pop("momentum_buffer")
            cleared += 1
    return cleared


class MicroBTSPPOBRPhaseAlignedTrainerScratch200(
    MicroBTSPPOBRCurriculumTrainerScratch200
):
    """Unchanged SPP-OBR method with a phase-aligned optimizer and epoch-60 gate."""

    phase_transition_epoch = 15
    phase_learning_rate = 3e-3
    first_hard_gate_epoch = 60
    allowed_stop_epochs = (15, 30, 45, 60, 200)

    @classmethod
    def requested_stop_epoch(cls) -> int:
        requested = int(
            os.getenv("MICROBT_PHASE_ALIGNED_STOP_EPOCHS", str(cls.training_epochs))
        )
        if requested not in cls.allowed_stop_epochs:
            raise ValueError(
                "phase-aligned stop epoch must be 15, 30, 45, 60, or 200"
            )
        return requested

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        requested = self.requested_stop_epoch()
        previous_gate = os.environ.get("MICROBT_CURRICULUM_GATE_EPOCHS")
        os.environ["MICROBT_CURRICULUM_GATE_EPOCHS"] = str(self.training_epochs)
        try:
            super().__init__(plans, configuration, fold, dataset_json, device)
        finally:
            if previous_gate is None:
                os.environ.pop("MICROBT_CURRICULUM_GATE_EPOCHS", None)
            else:
                os.environ["MICROBT_CURRICULUM_GATE_EPOCHS"] = previous_gate
        self.num_epochs = requested
        self.save_every = 5 if requested <= self.first_hard_gate_epoch else 25
        self._phase_transition_applied = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PhaseAlignedPolyLRScheduler(
            optimizer,
            initial_lr=self.initial_lr,
            phase_lr=self.phase_learning_rate,
            transition_epoch=self.phase_transition_epoch,
            max_epochs=self.training_epochs,
        )
        return optimizer, scheduler

    def on_train_epoch_start(self):
        result = super().on_train_epoch_start()
        if (
            int(self.current_epoch) == self.phase_transition_epoch
            and not self._phase_transition_applied
        ):
            cleared = reset_sgd_momentum_buffers(self.optimizer)
            self._phase_transition_applied = True
            self.print_to_log_file(
                "SPP-OBR phase transition: "
                f"epoch={self.phase_transition_epoch}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.8f}, "
                f"momentum_buffers_cleared={cleared}"
            )
        return result

