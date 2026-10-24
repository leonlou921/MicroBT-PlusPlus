from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    CurriculumWeights,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRH0OffCausalTrainer import (
    MicroBTSPPOBRH0OffCausalTrainerScratch60,
    h0off_curriculum_weights,
)


def field_cap_curriculum_weights(epoch: int) -> CurriculumWeights:
    """Cap only field after epoch 29; preserve birth and H0-off exactly."""
    epoch = int(epoch)
    original = h0off_curriculum_weights(epoch)
    field = min(float(original.field), 0.5) if epoch >= 30 else float(original.field)
    return CurriculumWeights(field, original.birth, original.h0)


class MicroBTSPPOBRFieldCapCausalTrainerScratch60(
    MicroBTSPPOBRH0OffCausalTrainerScratch60
):
    """Single-variable Field-Cap control derived from the H0-off run."""

    field_cap_start_epoch = 30
    field_cap = 0.5
    contract_label = "SPP-OBR field-cap causal contract"

    @classmethod
    def effective_curriculum_weights(cls, epoch: int) -> CurriculumWeights:
        return field_cap_curriculum_weights(epoch)
