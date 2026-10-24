from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_ctf_losses import (
    EqualInstanceAssignmentConsistencyLoss,
    MicroBTCTFLoss,
)


class UnitMassConservationLoss(nn.Module):
    """Give every valid lesion one integrated mass and background zero mass."""

    def __init__(
        self,
        background_weight: float = 0.10,
        beta: float = 0.50,
        hard_survival_mass: float | None = None,
    ) -> None:
        super().__init__()
        if background_weight < 0:
            raise ValueError("background_weight must be nonnegative")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if hard_survival_mass is not None and not 0 < hard_survival_mass <= 1:
            raise ValueError("hard_survival_mass must be in (0, 1]")
        self.background_weight = float(background_weight)
        self.beta = float(beta)
        self.hard_survival_mass = (
            None if hard_survival_mass is None else float(hard_survival_mass)
        )

    def components(
        self,
        density: torch.Tensor,
        targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if density.ndim != 5 or density.shape[1] != 1:
            raise ValueError("density must have shape [B,1,D,H,W]")
        if len(targets) != density.shape[0]:
            raise ValueError("target count must equal batch size")
        if not bool(torch.isfinite(density).all()):
            raise ValueError("density must be finite")
        if bool((density < 0).any()):
            raise ValueError("density must be nonnegative")

        object_terms: list[torch.Tensor] = []
        background_terms: list[torch.Tensor] = []
        predicted_masses: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            ids = target.instance_ids.to(device=density.device)
            if ids.shape != density.shape[2:]:
                raise ValueError("instance target shape mismatch")
            values = density[batch_index, 0]
            for instance_id in target.valid_instance_ids:
                mass = values[ids == int(instance_id)].sum()
                # A detached scale gives large objects O(1) aggregate gradient,
                # while a missing micro object keeps an undiminished gradient.
                scale = mass.detach().clamp_min(1.0)
                error = (mass - 1.0) / scale
                object_terms.append(
                    F.smooth_l1_loss(
                        error, torch.zeros_like(error), beta=self.beta
                    )
                )
                predicted_masses.append(mass)
            background = values[ids == 0]
            if background.numel():
                # log1p constrains accumulated false object mass without a
                # volume-proportional gradient spike on million-voxel patches.
                log_total_background_mass = torch.log1p(background.sum())
                background_terms.append(
                    F.smooth_l1_loss(
                        log_total_background_mass,
                        torch.zeros_like(log_total_background_mass),
                        beta=self.beta,
                    )
                )

        zero = density.sum() * 0.0
        object_loss = torch.stack(object_terms).mean() if object_terms else zero
        if self.hard_survival_mass is not None and predicted_masses:
            masses = torch.stack(predicted_masses)
            hard_masses = masses[masses.detach() < self.hard_survival_mass]
            if hard_masses.numel():
                # Decoder-aligned hard-object mining: already viable objects do
                # not dilute the restoring gradient of an object below the
                # frozen minimum atom mass.
                object_loss = object_loss + (
                    self.hard_survival_mass - hard_masses
                ).clamp_min(0).mean()
        background_loss = (
            torch.stack(background_terms).mean() if background_terms else zero
        )
        mean_mass = torch.stack(predicted_masses).mean() if predicted_masses else zero
        total = object_loss + self.background_weight * background_loss
        return total, object_loss, background_loss, mean_mass

    def forward(
        self,
        density: torch.Tensor,
        targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total, object_loss, background_loss, _ = self.components(density, targets)
        return total, object_loss, background_loss


class MicroBTUnitMassLoss(MicroBTCTFLoss):
    """Two-family objective: region fidelity plus one transported object measure."""

    def __init__(
        self,
        object_weight: float = 0.0,
        hard_survival_mass: float | None = None,
    ) -> None:
        super().__init__(object_weight=object_weight)
        self.object_transport = EqualInstanceAssignmentConsistencyLoss()
        self.mass_conservation = UnitMassConservationLoss(
            hard_survival_mass=hard_survival_mass
        )

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        required = {"transport", "region_probabilities", "object_density"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"Unit-Mass CTF output is missing {sorted(missing)}")
        region = self.region_fidelity(output["region_probabilities"], region_target)
        support = output["region_probabilities"][:, 0:1]
        density = output["object_density"]
        with torch.autocast(device_type=output["transport"].device.type, enabled=False):
            if self.object_weight == 0.0:
                zero = (
                    output["transport"].float().sum()
                    + support.float().sum()
                    + density.float().sum()
                ) * 0.0
                assignment_raw = endpoint = support_assignment = endpoint_error = zero
                mass_raw = object_mass = background_mass = mean_predicted_mass = zero
            else:
                assignment_raw, endpoint, support_assignment, endpoint_error = (
                    self.object_transport(
                        output["transport"].float(), support.float(), persistent_targets
                    )
                )
                mass_raw, object_mass, background_mass, mean_predicted_mass = (
                    self.mass_conservation.components(
                        density.float(), persistent_targets
                    )
                )
            object_raw = assignment_raw + mass_raw
            object_weighted = object_raw * self.object_weight
            total = region.float() + object_weighted
        zero = total * 0.0
        return {
            "loss": total,
            "region": region,
            "object_transport": object_weighted,
            "object_transport_raw": object_raw,
            "transport_offset": endpoint,
            "center_anchor": support_assignment,
            "endpoint_error_vox": endpoint_error,
            "mass_conservation_raw": mass_raw,
            "object_mass_error": object_mass,
            "background_mass_error": background_mass,
            "mean_predicted_object_mass": mean_predicted_mass,
            "object_weight": torch.as_tensor(
                self.object_weight, device=total.device, dtype=total.dtype
            ),
            "object_morse": object_weighted,
            "persistent_field": object_weighted,
            "field": object_weighted,
            "betti_0": zero,
            "false_persistence": zero,
            "birth_reservation": zero,
            "birth_reservation_raw": zero,
            "birth_existence": zero,
            "birth_existence_raw": zero,
            "birth_prominence": zero,
            "birth_prominence_raw": zero,
        }
