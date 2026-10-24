from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget


def asymmetric_birth_fields(
    field_logits: torch.Tensor,
    persistent_targets: Sequence[PersistentFieldTarget],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return forward-identical positive and negative SPP training fields."""
    if field_logits.ndim != 5 or field_logits.shape[1] != 4:
        raise ValueError("SPP field_logits must have shape [B,4,D,H,W]")
    if len(persistent_targets) != field_logits.shape[0]:
        raise ValueError("persistent target count must equal batch size")

    logits = field_logits.float()
    support = torch.sigmoid(logits[:, 0:1])
    raw_birth = torch.tanh(logits[:, 1:2])
    core_masks: list[torch.Tensor] = []
    for target in persistent_targets:
        if target.birth_core_ids.shape != field_logits.shape[2:]:
            raise ValueError("birth core target shape mismatch")
        core_masks.append(
            target.birth_core_ids.to(device=field_logits.device).ne(0)
        )
    core_mask = torch.stack(core_masks, dim=0).unsqueeze(1)
    positive_support = torch.where(core_mask, support, support.detach())
    negative_support = support.detach()
    positive_field = -1.0 + positive_support * (1.0 + raw_birth)
    negative_field = -1.0 + negative_support * (1.0 + raw_birth)
    return positive_field, negative_field


def _normalized_smooth_max(values: torch.Tensor, temperature: float) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("smooth maximum requires a nonempty vector")
    maximum = values.max()
    return maximum + float(temperature) * torch.log(
        torch.exp((values - maximum) / float(temperature)).mean()
    )


class OneBirthReservationLoss(nn.Module):
    """Reserve one sufficiently prominent persistent birth per valid object."""

    def __init__(
        self,
        birth_threshold: float = 0.25,
        prominence_margin: float = 0.25,
        temperature: float = 0.10,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.birth_threshold = float(birth_threshold)
        self.prominence_margin = float(prominence_margin)
        self.temperature = float(temperature)

    def forward(
        self,
        positive_field: torch.Tensor,
        negative_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positive_field.shape != negative_field.shape:
            raise ValueError("positive and negative birth fields must have equal shape")
        if positive_field.ndim != 5 or positive_field.shape[1] != 1:
            raise ValueError("birth fields must have shape [B,1,D,H,W]")
        if len(persistent_targets) != positive_field.shape[0]:
            raise ValueError("persistent target count must equal batch size")

        existence_terms: list[torch.Tensor] = []
        prominence_terms: list[torch.Tensor] = []
        for batch_index, target in enumerate(persistent_targets):
            positive = positive_field[batch_index, 0]
            negative = negative_field[batch_index, 0]
            if target.birth_core_ids.shape != positive.shape:
                raise ValueError("birth core target shape mismatch")
            if target.birth_moat_ids.shape != positive.shape:
                raise ValueError("birth moat target shape mismatch")
            core_ids = target.birth_core_ids.to(device=positive.device)
            moat_ids = target.birth_moat_ids.to(device=positive.device)
            for instance_id in target.valid_instance_ids:
                core_mask = core_ids == int(instance_id)
                if not bool(core_mask.any()):
                    raise RuntimeError(
                        f"valid instance ID {instance_id} has no birth-core voxels"
                    )
                birth = _normalized_smooth_max(
                    positive[core_mask], self.temperature
                )
                existence = torch.relu(self.birth_threshold - birth).square()
                moat_mask = moat_ids == int(instance_id)
                if bool(moat_mask.any()):
                    moat = _normalized_smooth_max(
                        negative[moat_mask], self.temperature
                    )
                    prominence = torch.relu(
                        self.prominence_margin - (birth - moat)
                    ).square()
                else:
                    prominence = birth * 0.0
                existence_terms.append(existence)
                prominence_terms.append(prominence)

        zero = positive_field.sum() * 0.0 + negative_field.sum() * 0.0
        if not existence_terms:
            return zero, zero, zero
        existence = torch.stack(existence_terms).mean()
        prominence = torch.stack(prominence_terms).mean()
        return existence + prominence, existence, prominence
