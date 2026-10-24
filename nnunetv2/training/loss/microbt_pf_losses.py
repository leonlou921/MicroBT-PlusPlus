from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.topology.persistent_field_0d import (
    InducedMatching0D,
    PersistencePairs0D,
    critical_values,
    induced_matching_0d,
    induced_matching_0d_precomputed_target,
)
from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget


class RegionFidelityLoss(nn.Module):
    """Mean-region soft Dice plus binary cross entropy."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)

    def forward(
        self, region_probabilities: torch.Tensor, region_target: torch.Tensor
    ) -> torch.Tensor:
        if region_probabilities.shape != region_target.shape:
            raise ValueError("region probabilities and target must have equal shapes")
        if region_probabilities.ndim < 3 or region_probabilities.shape[1] == 0:
            raise ValueError("region tensors must have shape [B,R,...] with R > 0")
        probability = region_probabilities.float()
        target = region_target.to(device=probability.device, dtype=torch.float32)
        intersection = (probability * target).flatten(2).sum(-1)
        denominator = probability.flatten(2).sum(-1) + target.flatten(2).sum(-1)
        dice = 1.0 - (2.0 * intersection + self.epsilon) / (
            denominator + self.epsilon
        )
        # Probability-space BCE is deliberately retained by the registered loss,
        # but PyTorch forbids it inside CUDA autocast. Evaluate only this primitive
        # in FP32 with autocast disabled; gradients still flow to the live logits.
        with torch.autocast(device_type=probability.device.type, enabled=False):
            binary_cross_entropy = F.binary_cross_entropy(
                probability, target, reduction="none"
            ).flatten(2).mean(-1)
        return dice.mean() + binary_cross_entropy.mean()


def _validate_field_batch(
    predicted_field: torch.Tensor,
    persistent_targets: Sequence[PersistentFieldTarget],
) -> None:
    if predicted_field.ndim != 5 or predicted_field.shape[1] != 1:
        raise ValueError("predicted field must have shape [B,1,D,H,W]")
    if len(persistent_targets) != predicted_field.shape[0]:
        raise ValueError("persistent target count must equal batch size")
    for target in persistent_targets:
        if tuple(target.field.shape) != tuple(predicted_field.shape[2:]):
            raise ValueError("persistent field target shape mismatch")
        if target.instance_ids.shape != target.field.shape:
            raise ValueError("instance_ids shape mismatch")
        if target.dense_valid_mask.shape != target.field.shape:
            raise ValueError("dense_valid_mask shape mismatch")


class EqualObjectFieldLoss(nn.Module):
    """Equal-object foreground L1 balanced equally with valid background L1."""

    def forward(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> torch.Tensor:
        _validate_field_batch(predicted_field, persistent_targets)
        sample_losses: list[torch.Tensor] = []
        for batch_index, target in enumerate(persistent_targets):
            prediction = predicted_field[batch_index, 0]
            target_field = target.field.to(
                device=prediction.device, dtype=prediction.dtype
            )
            instance_ids = target.instance_ids.to(device=prediction.device)
            valid_mask = target.dense_valid_mask.to(
                device=prediction.device, dtype=torch.bool
            )
            absolute_error = torch.abs(prediction - target_field)
            zero = prediction.sum() * 0.0

            object_errors: list[torch.Tensor] = []
            for instance_id in target.valid_instance_ids:
                instance_mask = valid_mask & (instance_ids == int(instance_id))
                if not bool(instance_mask.any()):
                    raise RuntimeError(
                        f"valid instance ID {instance_id} has no dense-valid voxels"
                    )
                object_errors.append(absolute_error[instance_mask].mean())

            background_mask = valid_mask & (instance_ids == 0)
            background_error = (
                absolute_error[background_mask].mean()
                if bool(background_mask.any())
                else zero
            )
            if object_errors:
                foreground_error = torch.stack(object_errors).mean()
                sample_losses.append(0.5 * foreground_error + 0.5 * background_error)
            else:
                sample_losses.append(background_error)
        return torch.stack(sample_losses).mean()


def _inside_topology_margin(flat_index: int, shape: tuple[int, int, int]) -> bool:
    coordinate = np.unravel_index(int(flat_index), shape)
    return all(0 < int(value) < shape[axis] - 1 for axis, value in enumerate(coordinate))


def _target_branch_validity(
    pairs: PersistencePairs0D, target: PersistentFieldTarget
) -> np.ndarray:
    topology_pairs = target.topology.pairs
    if (
        np.array_equal(pairs.birth_flat, topology_pairs.birth_flat)
        and np.array_equal(pairs.death_flat, topology_pairs.death_flat)
        and np.array_equal(pairs.essential, topology_pairs.essential)
    ):
        return target.topology.branch_valid
    instance_ids = target.instance_ids.detach().cpu().numpy().ravel()
    valid_ids = frozenset(int(value) for value in target.valid_instance_ids)
    return np.asarray(
        [int(instance_ids[int(index)]) in valid_ids for index in pairs.birth_flat],
        dtype=bool,
    )


def _prediction_branch_validity(
    pairs: PersistencePairs0D, target: PersistentFieldTarget
) -> np.ndarray:
    shape = tuple(int(value) for value in target.instance_ids.shape)
    instance_ids = target.instance_ids.detach().cpu().numpy().ravel()
    valid_ids = frozenset(int(value) for value in target.valid_instance_ids)
    validity: list[bool] = []
    for birth_flat in pairs.birth_flat:
        index = int(birth_flat)
        instance_id = int(instance_ids[index])
        invalid_fragment = instance_id > 0 and instance_id not in valid_ids
        validity.append(
            _inside_topology_margin(index, shape) and not invalid_fragment
        )
    return np.asarray(validity, dtype=bool)


def _resolved_branch_terms(
    matching: InducedMatching0D,
    target: PersistentFieldTarget,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    prediction_valid = _prediction_branch_validity(matching.prediction, target)
    target_valid = _target_branch_validity(matching.target, target)
    matched: list[tuple[int, int]] = []
    unmatched_prediction: list[int] = []
    unmatched_target: list[int] = []

    for prediction_index, target_index in zip(
        matching.matched_prediction.tolist(), matching.matched_target.tolist()
    ):
        prediction_is_valid = bool(prediction_valid[prediction_index])
        target_is_valid = bool(target_valid[target_index])
        if prediction_is_valid and target_is_valid:
            matched.append((int(prediction_index), int(target_index)))
        elif prediction_is_valid:
            unmatched_prediction.append(int(prediction_index))
        elif target_is_valid:
            unmatched_target.append(int(target_index))

    unmatched_prediction.extend(
        int(index)
        for index in matching.unmatched_prediction
        if prediction_valid[int(index)]
    )
    unmatched_target.extend(
        int(index)
        for index in matching.unmatched_target
        if target_valid[int(index)]
    )
    return matched, unmatched_prediction, unmatched_target


class BettiMatching0Loss(nn.Module):
    """Spatial induced H0 matching loss with live critical-value gradients."""

    def forward(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> torch.Tensor:
        _validate_field_batch(predicted_field, persistent_targets)
        if any(target.topology is None for target in persistent_targets):
            raise ValueError("persistent target topology is required for H0 matching")
        sample_losses: list[torch.Tensor] = []
        for batch_index, target in enumerate(persistent_targets):
            prediction = predicted_field[batch_index, 0]
            with torch.no_grad():
                matching = induced_matching_0d_precomputed_target(
                    prediction.detach(),
                    target.field,
                    target.topology.pairs,
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

            value = prediction.sum() * 0.0
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
                value = value + (
                    prediction_birth.index_select(0, prediction_indices)
                    - target_birth.index_select(0, target_indices)
                ).square().sum()
                value = value + (
                    prediction_death.index_select(0, prediction_indices)
                    - target_death.index_select(0, target_indices)
                ).square().sum()
            if unmatched_prediction:
                prediction_indices = torch.as_tensor(
                    unmatched_prediction,
                    device=prediction.device,
                    dtype=torch.long,
                )
                value = value + 0.5 * (
                    prediction_birth.index_select(0, prediction_indices)
                    - prediction_death.index_select(0, prediction_indices)
                ).square().sum()
            if unmatched_target:
                target_indices = torch.as_tensor(
                    unmatched_target,
                    device=prediction.device,
                    dtype=torch.long,
                )
                value = value + 0.5 * (
                    target_birth.index_select(0, target_indices)
                    - target_death.index_select(0, target_indices)
                ).square().sum()
            count = len(matched) + len(unmatched_prediction) + len(unmatched_target)
            sample_losses.append(value / max(1, count))
        return torch.stack(sample_losses).mean()


class PersistentFieldLoss(nn.Module):
    """The single persistent-field family: dense field plus H0 matching."""

    def __init__(self) -> None:
        super().__init__()
        self.equal_object_field = EqualObjectFieldLoss()
        self.betti_matching_0 = BettiMatching0Loss()

    def components(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        field = self.equal_object_field(predicted_field, persistent_targets)
        betti_0 = self.betti_matching_0(predicted_field, persistent_targets)
        return field, betti_0

    def forward(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> torch.Tensor:
        field, betti_0 = self.components(predicted_field, persistent_targets)
        return field + betti_0


class ObjectMorsePersistentFieldLoss(nn.Module):
    """SB-PF object field, exact H0, and gradient-complete missing births."""

    def __init__(self) -> None:
        super().__init__()
        self.equal_object_field = EqualObjectFieldLoss()

    @staticmethod
    def _topology_components(
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_field_batch(predicted_field, persistent_targets)
        if any(target.topology is None for target in persistent_targets):
            raise ValueError("persistent target topology is required for H0 matching")

        sample_betti: list[torch.Tensor] = []
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

            betti = prediction.sum() * 0.0
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
                betti = betti + (
                    prediction_birth.index_select(0, prediction_indices)
                    - target_birth.index_select(0, target_indices)
                ).square().sum()
                betti = betti + (
                    prediction_death.index_select(0, prediction_indices)
                    - target_death.index_select(0, target_indices)
                ).square().sum()
            if unmatched_prediction:
                prediction_indices = torch.as_tensor(
                    unmatched_prediction,
                    device=prediction.device,
                    dtype=torch.long,
                )
                betti = betti + 0.5 * (
                    prediction_birth.index_select(0, prediction_indices)
                    - prediction_death.index_select(0, prediction_indices)
                ).square().sum()
            branch_count = len(matched) + len(unmatched_prediction) + len(unmatched_target)
            sample_betti.append(betti / max(1, branch_count))

            completeness = prediction.sum() * 0.0
            if unmatched_target:
                target_indices = torch.as_tensor(
                    unmatched_target,
                    device=prediction.device,
                    dtype=torch.long,
                )
                birth_flat = torch.as_tensor(
                    matching.target.birth_flat[np.asarray(unmatched_target, dtype=np.int64)],
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
            sample_completeness.append(completeness)

        return torch.stack(sample_betti).mean(), torch.stack(sample_completeness).mean()

    def components(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        field = self.equal_object_field(predicted_field, persistent_targets)
        betti_0, birth_completeness = self._topology_components(
            predicted_field, persistent_targets
        )
        return field, betti_0, birth_completeness

    def forward(
        self,
        predicted_field: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> torch.Tensor:
        field, betti_0, birth_completeness = self.components(
            predicted_field, persistent_targets
        )
        return field + betti_0 + birth_completeness


class MicroBTPFLoss(nn.Module):
    """Exactly region fidelity plus one persistent-field family at weight one."""

    lambda_pf = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.region_fidelity = RegionFidelityLoss()
        self.persistent_field = PersistentFieldLoss()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        if "field_logits" not in output or "region_probabilities" not in output:
            raise KeyError("MicroBT-PF output requires field logits and region probabilities")
        region = self.region_fidelity(output["region_probabilities"], region_target)
        field_logits = output["field_logits"]
        if field_logits.ndim != 5 or field_logits.shape[1] < 1:
            raise ValueError("field_logits must have shape [B,C,D,H,W]")
        with torch.autocast(device_type=field_logits.device.type, enabled=False):
            field_fp32 = torch.tanh(field_logits[:, 0:1].float())
            field, betti_0 = self.persistent_field.components(
                field_fp32, persistent_targets
            )
            persistent_field = field + betti_0
            total = region.float() + self.lambda_pf * persistent_field
        return {
            "loss": total,
            "region": region,
            "persistent_field": persistent_field,
            "field": field,
            "betti_0": betti_0,
        }


class MicroBTSBPFLoss(nn.Module):
    """Exactly region fidelity plus the SB-PF object-Morse loss family."""

    lambda_object_morse = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.region_fidelity = RegionFidelityLoss()
        self.object_morse = ObjectMorsePersistentFieldLoss()

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        required = {"field_logits", "birth_field", "region_probabilities"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"MicroBT SB-PF output is missing {sorted(missing)}")
        field_logits = output["field_logits"]
        if field_logits.ndim != 5 or field_logits.shape[1] != 4:
            raise ValueError("SB-PF field_logits must have shape [B,4,D,H,W]")
        birth_field = output["birth_field"]
        region = self.region_fidelity(output["region_probabilities"], region_target)
        with torch.autocast(device_type=field_logits.device.type, enabled=False):
            field, betti_0, birth_completeness = self.object_morse.components(
                birth_field.float(), persistent_targets
            )
            object_morse = field + betti_0 + birth_completeness
            total = region.float() + self.lambda_object_morse * object_morse
        return {
            "loss": total,
            "region": region,
            "object_morse": object_morse,
            "field": field,
            "betti_0": betti_0,
            "birth_completeness": birth_completeness,
        }
