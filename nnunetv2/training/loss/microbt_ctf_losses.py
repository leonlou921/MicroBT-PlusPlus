from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_pf_losses import RegionFidelityLoss


class EqualInstanceCentroidTransportLoss(nn.Module):
    """Robust centroid-offset regression with one equal contribution per lesion."""

    def __init__(
        self,
        offset_scale_zyx: tuple[float, float, float] = (56.0, 80.0, 64.0),
        beta: float = 0.10,
        center_support_threshold: float = 0.50,
        anchor_weight: float = 0.25,
    ) -> None:
        super().__init__()
        scale = torch.as_tensor(offset_scale_zyx, dtype=torch.float32)
        if scale.shape != (3,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("offset_scale_zyx must contain three positive finite values")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 <= center_support_threshold <= 1.0:
            raise ValueError("center_support_threshold must be in [0, 1]")
        if anchor_weight < 0.0:
            raise ValueError("anchor_weight must be nonnegative")
        self.register_buffer("offset_scale_zyx", scale, persistent=False)
        self.beta = float(beta)
        self.center_support_threshold = float(center_support_threshold)
        self.anchor_weight = float(anchor_weight)

    def forward(
        self,
        transport: torch.Tensor,
        support: torch.Tensor,
        targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if transport.ndim != 5 or transport.shape[1] != 3:
            raise ValueError("transport must have shape [B,3,D,H,W]")
        if support.shape != (transport.shape[0], 1, *transport.shape[2:]):
            raise ValueError("support must have shape [B,1,D,H,W]")
        if len(targets) != transport.shape[0]:
            raise ValueError("target count must equal batch size")

        scale = self.offset_scale_zyx.to(device=transport.device, dtype=transport.dtype)
        offset_terms: list[torch.Tensor] = []
        anchor_terms: list[torch.Tensor] = []
        endpoint_terms: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            ids = target.instance_ids.to(device=transport.device)
            if ids.shape != transport.shape[2:]:
                raise ValueError("instance target shape mismatch")
            if len(target.valid_instance_ids) != len(target.centers_zyx):
                raise ValueError("valid instance IDs and centers must be aligned")
            if not target.valid_instance_ids:
                continue

            valid_ids = torch.as_tensor(
                target.valid_instance_ids, device=transport.device, dtype=torch.long
            )
            centers = torch.as_tensor(
                target.centers_zyx, device=transport.device, dtype=transport.dtype
            )
            max_id = int(ids.max().item())
            compact_lookup = torch.full(
                (max_id + 1,), -1, device=transport.device, dtype=torch.long
            )
            compact_lookup[valid_ids] = torch.arange(
                len(target.valid_instance_ids), device=transport.device
            )
            center_lookup = torch.zeros(
                (max_id + 1, 3), device=transport.device, dtype=transport.dtype
            )
            center_lookup[valid_ids] = centers

            coords = torch.nonzero(ids > 0, as_tuple=False)
            voxel_instance_ids = ids[
                coords[:, 0], coords[:, 1], coords[:, 2]
            ].long()
            compact_ids = compact_lookup[voxel_instance_ids]
            keep = compact_ids >= 0
            coords = coords[keep]
            voxel_instance_ids = voxel_instance_ids[keep]
            compact_ids = compact_ids[keep]
            target_offset = (
                center_lookup[voxel_instance_ids] - coords.to(transport.dtype)
            ) / scale
            target_offset = target_offset.clamp(min=-1.0, max=1.0)
            predicted = transport[
                batch_index, :, coords[:, 0], coords[:, 1], coords[:, 2]
            ].transpose(0, 1)

            instance_count = len(target.valid_instance_ids)
            counts = torch.zeros(
                instance_count, device=transport.device, dtype=transport.dtype
            )
            counts.scatter_add_(
                0, compact_ids, torch.ones_like(compact_ids, dtype=transport.dtype)
            )
            if bool((counts == 0).any()):
                missing = [
                    instance_id
                    for index, instance_id in enumerate(target.valid_instance_ids)
                    if counts[index].item() == 0
                ]
                raise RuntimeError(f"valid instance IDs {missing} have no voxels")

            voxel_offset = F.smooth_l1_loss(
                predicted, target_offset, reduction="none", beta=self.beta
            ).mean(dim=1)
            offset_sums = torch.zeros_like(counts)
            offset_sums.scatter_add_(0, compact_ids, voxel_offset)
            offset_terms.append((offset_sums / counts).mean())

            voxel_endpoint = torch.linalg.vector_norm(
                (predicted - target_offset) * scale, dim=1
            )
            endpoint_sums = torch.zeros_like(counts)
            endpoint_sums.scatter_add_(0, compact_ids, voxel_endpoint)
            endpoint_terms.append((endpoint_sums / counts).mean())

            center_indices = centers.long().unbind(dim=1)
            center_support = support[batch_index, 0][center_indices]
            anchor_terms.append(
                torch.relu(self.center_support_threshold - center_support)
                .square()
                .mean()
            )

        zero = transport.sum() * 0.0 + support.sum() * 0.0
        if not offset_terms:
            return zero, zero, zero, zero
        offset = torch.stack(offset_terms).mean()
        anchor = torch.stack(anchor_terms).mean()
        endpoint_error = torch.stack(endpoint_terms).mean()
        total = offset + self.anchor_weight * anchor
        return total, offset, anchor, endpoint_error


class EqualInstanceAssignmentConsistencyLoss(nn.Module):
    """Make each lesion an equal-weight support-to-centroid assignment."""

    def __init__(
        self,
        offset_scale_zyx: tuple[float, float, float] = (56.0, 80.0, 64.0),
        spacing_zyx: tuple[float, float, float] = (1.0, 0.9375, 0.9375),
        endpoint_beta_mm: float = 1.0,
        endpoint_scale_mm: float = 5.0,
        support_target: float = 0.50,
    ) -> None:
        super().__init__()
        scale = torch.as_tensor(offset_scale_zyx, dtype=torch.float32)
        spacing = torch.as_tensor(spacing_zyx, dtype=torch.float32)
        if scale.shape != (3,) or bool((scale <= 0).any()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("offset_scale_zyx must contain three positive finite values")
        if spacing.shape != (3,) or bool((spacing <= 0).any()) or not bool(torch.isfinite(spacing).all()):
            raise ValueError("spacing_zyx must contain three positive finite values")
        if endpoint_beta_mm <= 0 or endpoint_scale_mm <= 0:
            raise ValueError("endpoint beta and scale must be positive")
        if not 0.0 <= support_target <= 1.0:
            raise ValueError("support_target must be in [0, 1]")
        self.register_buffer("offset_scale_zyx", scale, persistent=False)
        self.register_buffer("spacing_zyx", spacing, persistent=False)
        self.endpoint_beta_mm = float(endpoint_beta_mm)
        self.endpoint_scale_mm = float(endpoint_scale_mm)
        self.support_target = float(support_target)

    def forward(
        self,
        transport: torch.Tensor,
        support: torch.Tensor,
        targets: Sequence[PersistentFieldTarget],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if transport.ndim != 5 or transport.shape[1] != 3:
            raise ValueError("transport must have shape [B,3,D,H,W]")
        if support.shape != (transport.shape[0], 1, *transport.shape[2:]):
            raise ValueError("support must have shape [B,1,D,H,W]")
        if len(targets) != transport.shape[0]:
            raise ValueError("target count must equal batch size")
        scale = self.offset_scale_zyx.to(transport)
        spacing = self.spacing_zyx.to(transport)
        endpoint_terms: list[torch.Tensor] = []
        support_terms: list[torch.Tensor] = []
        endpoint_error_terms: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            ids = target.instance_ids.to(device=transport.device)
            if not target.valid_instance_ids:
                continue
            valid_ids = torch.as_tensor(target.valid_instance_ids, device=transport.device, dtype=torch.long)
            centers = torch.as_tensor(target.centers_zyx, device=transport.device, dtype=transport.dtype)
            max_id = int(ids.max().item())
            compact_lookup = torch.full((max_id + 1,), -1, device=transport.device, dtype=torch.long)
            compact_lookup[valid_ids] = torch.arange(len(valid_ids), device=transport.device)
            coords = torch.nonzero(ids > 0, as_tuple=False)
            voxel_ids = ids[coords[:, 0], coords[:, 1], coords[:, 2]].long()
            compact_ids = compact_lookup[voxel_ids]
            keep = compact_ids >= 0
            coords = coords[keep]
            compact_ids = compact_ids[keep]
            instance_count = len(valid_ids)
            counts = torch.zeros(instance_count, device=transport.device, dtype=transport.dtype)
            counts.scatter_add_(0, compact_ids, torch.ones_like(compact_ids, dtype=transport.dtype))
            if bool((counts == 0).any()):
                raise RuntimeError("at least one valid instance has no voxels")
            predicted = transport[
                batch_index, :, coords[:, 0], coords[:, 1], coords[:, 2]
            ].transpose(0, 1)
            endpoint_error_mm = (
                coords.to(transport.dtype) + predicted * scale - centers[compact_ids]
            ) * spacing
            voxel_endpoint = F.smooth_l1_loss(
                endpoint_error_mm,
                torch.zeros_like(endpoint_error_mm),
                beta=self.endpoint_beta_mm,
                reduction="none",
            ).mean(dim=1) / self.endpoint_scale_mm
            voxel_endpoint_error = torch.linalg.vector_norm(endpoint_error_mm, dim=1)
            voxel_support = support[batch_index, 0, coords[:, 0], coords[:, 1], coords[:, 2]]
            voxel_assignment = torch.relu(self.support_target - voxel_support).square()
            for voxel_values, destination in (
                (voxel_endpoint, endpoint_terms),
                (voxel_assignment, support_terms),
                (voxel_endpoint_error, endpoint_error_terms),
            ):
                sums = torch.zeros_like(counts)
                sums.scatter_add_(0, compact_ids, voxel_values)
                destination.append((sums / counts).mean())
        zero = transport.sum() * 0.0 + support.sum() * 0.0
        if not endpoint_terms:
            return zero, zero, zero, zero
        endpoint = torch.stack(endpoint_terms).mean()
        assignment = torch.stack(support_terms).mean()
        endpoint_error = torch.stack(endpoint_error_terms).mean()
        return endpoint + assignment, endpoint, assignment, endpoint_error


class MicroBTCTFLoss(nn.Module):
    """Two-family CTF objective: nested region fidelity plus object transport."""

    def __init__(self, object_weight: float = 0.0) -> None:
        super().__init__()
        self.region_fidelity = RegionFidelityLoss()
        self.object_transport = EqualInstanceCentroidTransportLoss()
        self.set_object_weight(object_weight)

    def set_object_weight(self, weight: float) -> None:
        weight = float(weight)
        if not 0.0 <= weight <= 0.25:
            raise ValueError("CTF object weight must be in [0, 0.25]")
        self.object_weight = weight

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        required = {"transport", "region_probabilities"}
        missing = required.difference(output)
        if missing:
            raise KeyError(f"MicroBT-CTF output is missing {sorted(missing)}")
        region = self.region_fidelity(output["region_probabilities"], region_target)
        support = output["region_probabilities"][:, 0:1]
        with torch.autocast(device_type=output["transport"].device.type, enabled=False):
            if self.object_weight == 0.0:
                zero_object = output["transport"].float().sum() * 0.0 + support.float().sum() * 0.0
                object_raw = offset = anchor = endpoint_error = zero_object
            else:
                object_raw, offset, anchor, endpoint_error = self.object_transport(
                    output["transport"].float(), support.float(), persistent_targets
                )
            object_weighted = object_raw * self.object_weight
            total = region.float() + object_weighted
        zero = total * 0.0
        return {
            "loss": total,
            "region": region,
            "object_transport": object_weighted,
            "object_transport_raw": object_raw,
            "transport_offset": offset,
            "center_anchor": anchor,
            "endpoint_error_vox": endpoint_error,
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


class MicroBTCTFAssignmentLoss(MicroBTCTFLoss):
    """CTF loss with equal-instance physical assignment consistency."""

    def __init__(self, object_weight: float = 0.0) -> None:
        super().__init__(object_weight=object_weight)
        self.object_transport = EqualInstanceAssignmentConsistencyLoss()
