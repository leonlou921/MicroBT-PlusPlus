from __future__ import annotations

from collections.abc import Sequence

import torch

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_umctf_losses import MicroBTUnitMassLoss


class MicroBTSACLLoss(MicroBTUnitMassLoss):
    """UMCTF objective with independently continued measure and transport terms."""

    max_object_weight = 0.25

    def __init__(
        self,
        measure_weight: float = 0.0,
        transport_weight: float = 0.0,
        hard_survival_mass: float | None = None,
    ) -> None:
        super().__init__(
            object_weight=0.0,
            hard_survival_mass=hard_survival_mass,
        )
        self.set_sacl_weights(measure_weight, transport_weight)

    def set_sacl_weights(
        self, measure_weight: float, transport_weight: float
    ) -> None:
        measure_weight = float(measure_weight)
        transport_weight = float(transport_weight)
        for name, value in (
            ("measure_weight", measure_weight),
            ("transport_weight", transport_weight),
        ):
            if not 0.0 <= value <= self.max_object_weight:
                raise ValueError(
                    f"{name} must be in [0, {self.max_object_weight}]"
                )
        self.measure_weight = measure_weight
        self.transport_weight = transport_weight
        # Preserve the parent telemetry contract without using this value to
        # aggregate the two independently continued terms.
        self.object_weight = measure_weight

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        required = {"transport", "region_probabilities", "object_density"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"SACL output is missing {sorted(missing)}")
        region = self.region_fidelity(output["region_probabilities"], region_target)
        support = output["region_probabilities"][:, 0:1]
        density = output["object_density"]
        with torch.autocast(
            device_type=output["transport"].device.type,
            enabled=False,
        ):
            zero = (
                output["transport"].float().sum()
                + support.float().sum()
                + density.float().sum()
            ) * 0.0
            if self.transport_weight == 0.0:
                assignment_raw = endpoint = support_assignment = endpoint_error = zero
            else:
                assignment_raw, endpoint, support_assignment, endpoint_error = (
                    self.object_transport(
                        output["transport"].float(),
                        support.float(),
                        persistent_targets,
                    )
                )
            if self.measure_weight == 0.0:
                mass_raw = object_mass = background_mass = mean_predicted_mass = zero
            else:
                mass_raw, object_mass, background_mass, mean_predicted_mass = (
                    self.mass_conservation.components(
                        density.float(), persistent_targets
                    )
                )
            transport_weighted = assignment_raw * self.transport_weight
            mass_weighted = mass_raw * self.measure_weight
            object_weighted = transport_weighted + mass_weighted
            object_raw = assignment_raw + mass_raw
            total = region.float() + object_weighted
        scalar = lambda value: torch.as_tensor(
            value, device=total.device, dtype=total.dtype
        )
        return {
            "loss": total,
            "region": region,
            "object_transport": object_weighted,
            "object_transport_raw": object_raw,
            "transport_assignment_raw": assignment_raw,
            "transport_assignment_weighted": transport_weighted,
            "transport_offset": endpoint,
            "center_anchor": support_assignment,
            "endpoint_error_vox": endpoint_error,
            "mass_conservation_raw": mass_raw,
            "mass_conservation_weighted": mass_weighted,
            "object_mass_error": object_mass,
            "background_mass_error": background_mass,
            "mean_predicted_object_mass": mean_predicted_mass,
            "object_weight": scalar(self.measure_weight),
            "sacl_measure_weight": scalar(self.measure_weight),
            "sacl_transport_weight": scalar(self.transport_weight),
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
