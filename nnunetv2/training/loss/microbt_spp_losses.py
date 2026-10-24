from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from nnunetv2.topology.persistent_field_0d import (
    critical_values,
    induced_matching_0d_precomputed_target,
)
from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_pf_losses import (
    EqualObjectFieldLoss,
    RegionFidelityLoss,
    _resolved_branch_terms,
    _validate_field_batch,
)
from nnunetv2.training.loss.microbt_spp_obr import (
    OneBirthReservationLoss,
    asymmetric_birth_fields,
)


class SupportPreservingOBRPersistentFieldLoss(nn.Module):
    """Dense equal-object field plus count-sensitive spatial H0 matching."""

    def __init__(self) -> None:
        super().__init__()
        self.equal_object_field = EqualObjectFieldLoss()

    @staticmethod
    def _topology_components(
        positive_field: torch.Tensor,
        negative_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_field_batch(positive_field, persistent_targets)
        _validate_field_batch(negative_field, persistent_targets)
        if positive_field.shape != negative_field.shape:
            raise ValueError("positive and negative birth fields must have equal shape")
        if any(target.topology is None for target in persistent_targets):
            raise ValueError("persistent target topology is required for H0 matching")

        sample_betti: list[torch.Tensor] = []
        sample_false: list[torch.Tensor] = []
        for batch_index, target in enumerate(persistent_targets):
            positive_prediction = positive_field[batch_index, 0]
            negative_prediction = negative_field[batch_index, 0]
            with torch.no_grad():
                matching = induced_matching_0d_precomputed_target(
                    negative_prediction.detach(), target.field, target.topology.pairs
                )
            positive_birth, positive_death = critical_values(
                positive_prediction, matching.prediction, floor=-1.0
            )
            negative_birth, negative_death = critical_values(
                negative_prediction, matching.prediction, floor=-1.0
            )
            target_birth = torch.tensor(
                target.topology.birth_values,
                device=positive_prediction.device,
                dtype=positive_prediction.dtype,
            )
            target_death = torch.tensor(
                target.topology.death_values,
                device=positive_prediction.device,
                dtype=positive_prediction.dtype,
            )
            matched, unmatched_prediction, unmatched_target = _resolved_branch_terms(
                matching, target
            )
            gt_count = len(matched) + len(unmatched_target)

            matched_value = positive_prediction.sum() * 0.0
            if matched:
                prediction_indices = torch.as_tensor(
                    [item[0] for item in matched],
                    device=positive_prediction.device,
                    dtype=torch.long,
                )
                target_indices = torch.as_tensor(
                    [item[1] for item in matched],
                    device=positive_prediction.device,
                    dtype=torch.long,
                )
                matched_value = matched_value + (
                    positive_birth.index_select(0, prediction_indices)
                    - target_birth.index_select(0, target_indices)
                ).square().sum()
                matched_value = matched_value + (
                    positive_death.index_select(0, prediction_indices)
                    - target_death.index_select(0, target_indices)
                ).square().sum()
            matched_value = matched_value / max(1, gt_count)

            false_value = negative_prediction.sum() * 0.0
            if unmatched_prediction:
                prediction_indices = torch.as_tensor(
                    unmatched_prediction,
                    device=negative_prediction.device,
                    dtype=torch.long,
                )
                persistence = (
                    negative_birth.index_select(0, prediction_indices)
                    - negative_death.index_select(0, prediction_indices)
                )
                false_value = persistence.square().sum() / float(gt_count + 1)

            sample_false.append(false_value)
            sample_betti.append(matched_value + false_value)

        return (
            torch.stack(sample_betti).mean(),
            torch.stack(sample_false).mean(),
        )

    def components(
        self,
        positive_field: torch.Tensor,
        negative_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self.equal_object_field(negative_field, persistent_targets)
        betti_0, false_persistence = self._topology_components(
            positive_field, negative_field, persistent_targets
        )
        return field, betti_0, false_persistence


class MicroBTSPPOBRLoss(nn.Module):
    """Region fidelity plus the OBR support-preserving object family."""

    def __init__(self) -> None:
        super().__init__()
        self.region_fidelity = RegionFidelityLoss()
        self.object_morse = SupportPreservingOBRPersistentFieldLoss()
        self.birth_reservation = OneBirthReservationLoss()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        required = {"field_logits", "birth_field", "region_probabilities"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"MicroBT-SPP output is missing {sorted(missing)}")
        field_logits = output["field_logits"]
        if field_logits.ndim != 5 or field_logits.shape[1] != 4:
            raise ValueError("SPP field_logits must have shape [B,4,D,H,W]")
        region = self.region_fidelity(output["region_probabilities"], region_target)
        with torch.autocast(device_type=field_logits.device.type, enabled=False):
            positive_field, negative_field = asymmetric_birth_fields(
                field_logits, persistent_targets
            )
            field, betti_0, false_persistence = self.object_morse.components(
                positive_field, negative_field, persistent_targets
            )
            birth_reservation, birth_existence, birth_prominence = (
                self.birth_reservation(
                    positive_field, negative_field, persistent_targets
                )
            )
            object_morse = field + betti_0 + birth_reservation
            total = region.float() + object_morse
        return {
            "loss": total,
            "region": region,
            "object_morse": object_morse,
            "field": field,
            "betti_0": betti_0,
            "false_persistence": false_persistence,
            "birth_reservation": birth_reservation,
            "birth_existence": birth_existence,
            "birth_prominence": birth_prominence,
        }


class SupportPreservingPersistentFieldLoss(nn.Module):
    """Original SPP object loss retained as the causal control."""

    def __init__(self) -> None:
        super().__init__()
        self.equal_object_field = EqualObjectFieldLoss()

    @staticmethod
    def _topology_components(
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_field_batch(predicted_field, persistent_targets)
        if any(target.topology is None for target in persistent_targets):
            raise ValueError("persistent target topology is required for H0 matching")

        sample_betti: list[torch.Tensor] = []
        sample_false: list[torch.Tensor] = []
        sample_completeness: list[torch.Tensor] = []
        for batch_index, target in enumerate(persistent_targets):
            prediction = predicted_field[batch_index, 0]
            with torch.no_grad():
                matching = induced_matching_0d_precomputed_target(
                    prediction.detach(), target.field, target.topology.pairs
                )
            prediction_birth, prediction_death = critical_values(
                prediction, matching.prediction, floor=-1.0
            )
            target_birth = torch.tensor(
                target.topology.birth_values,
                device=prediction.device,
                dtype=prediction.dtype,
            )
            target_death = torch.tensor(
                target.topology.death_values,
                device=prediction.device,
                dtype=prediction.dtype,
            )
            matched, unmatched_prediction, unmatched_target = _resolved_branch_terms(
                matching, target
            )
            gt_count = len(matched) + len(unmatched_target)

            matched_value = prediction.sum() * 0.0
            if matched:
                prediction_indices = torch.as_tensor(
                    [item[0] for item in matched],
                    device=prediction.device,
                    dtype=torch.long,
                )
                target_indices = torch.as_tensor(
                    [item[1] for item in matched],
                    device=prediction.device,
                    dtype=torch.long,
                )
                matched_value = matched_value + (
                    prediction_birth.index_select(0, prediction_indices)
                    - target_birth.index_select(0, target_indices)
                ).square().sum()
                matched_value = matched_value + (
                    prediction_death.index_select(0, prediction_indices)
                    - target_death.index_select(0, target_indices)
                ).square().sum()
            matched_value = matched_value / max(1, gt_count)

            false_value = prediction.sum() * 0.0
            if unmatched_prediction:
                prediction_indices = torch.as_tensor(
                    unmatched_prediction,
                    device=prediction.device,
                    dtype=torch.long,
                )
                persistence = (
                    prediction_birth.index_select(0, prediction_indices)
                    - prediction_death.index_select(0, prediction_indices)
                )
                false_value = persistence.square().sum() / float(gt_count + 1)

            completeness = prediction.sum() * 0.0
            if unmatched_target:
                target_indices = torch.as_tensor(
                    unmatched_target,
                    device=prediction.device,
                    dtype=torch.long,
                )
                birth_flat = torch.as_tensor(
                    matching.target.birth_flat[
                        np.asarray(unmatched_target, dtype=np.int64)
                    ],
                    device=prediction.device,
                    dtype=torch.long,
                )
                predicted_at_target_birth = prediction.reshape(-1).index_select(
                    0, birth_flat
                )
                completeness = (
                    predicted_at_target_birth
                    - target_birth.index_select(0, target_indices)
                ).square().mean()

            sample_false.append(false_value)
            sample_betti.append(matched_value + false_value)
            sample_completeness.append(completeness)

        return (
            torch.stack(sample_betti).mean(),
            torch.stack(sample_false).mean(),
            torch.stack(sample_completeness).mean(),
        )

    def components(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self.equal_object_field(predicted_field, persistent_targets)
        betti_0, false_persistence, birth_completeness = self._topology_components(
            predicted_field, persistent_targets
        )
        return field, betti_0, false_persistence, birth_completeness


class MicroBTSPPLoss(nn.Module):
    """Original SPP loss retained unchanged for the continuation control."""

    def __init__(self) -> None:
        super().__init__()
        self.region_fidelity = RegionFidelityLoss()
        self.object_morse = SupportPreservingPersistentFieldLoss()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        required = {"field_logits", "birth_field", "region_probabilities"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"MicroBT-SPP output is missing {sorted(missing)}")
        field_logits = output["field_logits"]
        if field_logits.ndim != 5 or field_logits.shape[1] != 4:
            raise ValueError("SPP field_logits must have shape [B,4,D,H,W]")
        region = self.region_fidelity(output["region_probabilities"], region_target)
        with torch.autocast(device_type=field_logits.device.type, enabled=False):
            field, betti_0, false_persistence, birth_completeness = (
                self.object_morse.components(
                    output["birth_field"].float(), persistent_targets
                )
            )
            object_morse = field + betti_0 + birth_completeness
            total = region.float() + object_morse
        return {
            "loss": total,
            "region": region,
            "object_morse": object_morse,
            "field": field,
            "betti_0": betti_0,
            "false_persistence": false_persistence,
            "birth_completeness": birth_completeness,
        }
