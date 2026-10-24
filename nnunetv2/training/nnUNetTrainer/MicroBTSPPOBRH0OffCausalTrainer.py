from __future__ import annotations

from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    CurriculumWeights,
    _MicroBTSPPOBRCurriculumCriterion,
    curriculum_weights,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRPhaseAlignedTrainer import (
    MicroBTSPPOBRPhaseAlignedTrainerScratch200,
)


def phase_checkpoint_name(completed_epoch: int) -> str:
    completed_epoch = int(completed_epoch)
    if completed_epoch <= 0:
        raise ValueError("completed epoch must be positive")
    return f"checkpoint_phase_{completed_epoch:03d}.pth"


def h0off_curriculum_weights(epoch: int) -> CurriculumWeights:
    """Preserve field/birth ramps while making the H0 contract explicit."""
    original = curriculum_weights(epoch)
    return CurriculumWeights(original.field, original.birth, 0.0)


class MicroBTSPPOBRH0OffCausalTrainerScratch60(
    MicroBTSPPOBRPhaseAlignedTrainerScratch200
):
    """Continuous 60-epoch scratch control with exact H0 disabled."""

    causal_epochs = 60
    snapshot_epochs = (15, 30, 35, 40, 45, 50, 55, 60)
    contract_label = "SPP-OBR H0-off causal contract"

    @classmethod
    def effective_curriculum_weights(cls, epoch: int) -> CurriculumWeights:
        return h0off_curriculum_weights(epoch)

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.causal_epochs
        self.save_every = 5

    @classmethod
    def exact_h0_schedule(
        cls, epoch: int, step: int
    ) -> tuple[str, bool, float]:
        epoch = int(epoch)
        step = int(step)
        if not 0 <= epoch < cls.training_epochs:
            raise ValueError(f"epoch must be in [0, {cls.training_epochs})")
        if not 0 <= step < 250:
            raise ValueError("training step must be in [0, 250)")
        return "h0_off_causal", False, 0.0

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weights = self.effective_curriculum_weights(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, _MicroBTSPPOBRCurriculumCriterion):
                raise TypeError("H0-off SPP-OBR requires the curriculum criterion")
            criterion.set_curriculum_weights(weights)
        self._current_curriculum_weights = weights
        return max(weights)

    def on_train_epoch_start(self):
        epoch = int(self.current_epoch)
        weights = self.effective_curriculum_weights(epoch)
        self.print_to_log_file(
            f"{self.contract_label}: epoch={epoch}, "
            f"field={weights.field:.6f}, birth={weights.birth:.6f}, "
            "h0=0.000000, exact_h0=disabled"
        )
        return super().on_train_epoch_start()

    def on_epoch_end(self):
        completed_epoch = int(self.current_epoch) + 1
        result = super().on_epoch_end()
        if completed_epoch in self.snapshot_epochs:
            # nnU-Net increments current_epoch inside super(). Save with the
            # same completed-epoch semantics used by checkpoint_final.
            self.current_epoch -= 1
            try:
                destination = Path(self.output_folder) / phase_checkpoint_name(
                    completed_epoch
                )
                self.save_checkpoint(str(destination))
                self.print_to_log_file(
                    f"SPP-OBR causal snapshot saved: {destination.name}"
                )
            finally:
                self.current_epoch += 1
        return result
