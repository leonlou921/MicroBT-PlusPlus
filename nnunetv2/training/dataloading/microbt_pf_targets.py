from __future__ import annotations

from dataclasses import InitVar, dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label
from skimage.graph import MCP_Geometric

from nnunetv2.topology.persistent_field_0d import (
    PersistencePairs0D,
    persistence_pairs_0d,
)


@dataclass(frozen=True)
class PersistentTargetTopology:
    pairs: PersistencePairs0D
    birth_values: np.ndarray
    death_values: np.ndarray
    branch_valid: np.ndarray

    def __post_init__(self) -> None:
        pair_count = int(self.pairs.birth_flat.size)
        arrays = (
            ("birth_values", self.birth_values, np.dtype(np.float32)),
            ("death_values", self.death_values, np.dtype(np.float32)),
            ("branch_valid", self.branch_valid, np.dtype(bool)),
        )
        for name, value, dtype in arrays:
            array = np.ascontiguousarray(value, dtype=dtype)
            if array.ndim != 1 or array.size != pair_count:
                raise ValueError(f"{name} must have one entry per target pair")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        for array in (
            self.pairs.birth_flat,
            self.pairs.death_flat,
            self.pairs.essential,
        ):
            array.flags.writeable = False

    def __reduce__(self):
        return (
            type(self),
            (self.pairs, self.birth_values, self.death_values, self.branch_valid),
        )


@dataclass(frozen=True)
class PersistentFieldTarget:
    field: torch.Tensor
    instance_ids: torch.Tensor
    dense_valid_mask: torch.Tensor
    valid_instance_ids: tuple[int, ...]
    centers_zyx: tuple[tuple[int, int, int], ...]
    topology: PersistentTargetTopology | None = None
    birth_core_ids: torch.Tensor | None = None
    birth_moat_ids: torch.Tensor | None = None
    precompute_topology: InitVar[bool] = True

    def __post_init__(self, precompute_topology: bool) -> None:
        core_ids = self.birth_core_ids
        moat_ids = self.birth_moat_ids
        if core_ids is None and moat_ids is None:
            core_ids, moat_ids = _build_birth_reservation_ids(self)
            object.__setattr__(self, "birth_core_ids", core_ids)
            object.__setattr__(self, "birth_moat_ids", moat_ids)
        elif core_ids is None or moat_ids is None:
            raise ValueError("birth core and moat ID maps must be provided together")
        for name, value in (
            ("birth_core_ids", self.birth_core_ids),
            ("birth_moat_ids", self.birth_moat_ids),
        ):
            if value.shape != self.instance_ids.shape:
                raise ValueError(f"{name} shape mismatch")
            if value.dtype not in (torch.int16, torch.int32, torch.int64):
                raise ValueError(f"{name} must use an integer dtype")
        if self.topology is None and precompute_topology:
            object.__setattr__(self, "topology", _build_target_topology(self))


def _build_birth_reservation_ids(
    target: PersistentFieldTarget,
) -> tuple[torch.Tensor, torch.Tensor]:
    field = target.field.detach().to(device="cpu", dtype=torch.float32)
    instance_ids = target.instance_ids.detach().to(device="cpu", dtype=torch.int32)
    if field.shape != instance_ids.shape:
        raise ValueError("field and instance_ids shape mismatch")
    core_ids = torch.zeros_like(instance_ids)
    moat_ids = torch.zeros_like(instance_ids)
    centers_by_id = (
        dict(zip(target.valid_instance_ids, target.centers_zyx, strict=True))
        if len(target.valid_instance_ids) == len(target.centers_zyx)
        else {}
    )
    for instance_id in target.valid_instance_ids:
        instance_mask = instance_ids == int(instance_id)
        core_mask = instance_mask & (field >= 0.75)
        if not bool(core_mask.any()) and int(instance_id) in centers_by_id:
            center = tuple(int(value) for value in centers_by_id[int(instance_id)])
            if not instance_mask[center]:
                raise ValueError(
                    f"center for valid instance ID {instance_id} lies outside the instance"
                )
            core_mask[center] = True
        moat_mask = instance_mask & (field >= 0.25) & (field <= 0.50)
        core_ids[core_mask] = int(instance_id)
        moat_ids[moat_mask] = int(instance_id)
    return core_ids.contiguous(), moat_ids.contiguous()


def _build_target_topology(target: PersistentFieldTarget) -> PersistentTargetTopology:
    field = target.field.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    pairs = persistence_pairs_0d(field, floor=-1.0)
    flat = field.ravel()
    birth_values = flat[pairs.birth_flat.astype(np.int64)].copy()
    death_values = np.full(pairs.death_flat.size, -1.0, dtype=np.float32)
    finite = pairs.death_flat >= 0
    death_values[finite] = flat[pairs.death_flat[finite]]
    instance_ids = target.instance_ids.detach().cpu().numpy().ravel()
    valid_ids = frozenset(int(value) for value in target.valid_instance_ids)
    branch_valid = np.asarray(
        [int(instance_ids[int(index)]) in valid_ids for index in pairs.birth_flat],
        dtype=bool,
    )
    return PersistentTargetTopology(
        pairs=pairs,
        birth_values=birth_values,
        death_values=death_values,
        branch_valid=branch_valid,
    )


def precompute_persistent_target_topology(
    target: PersistentFieldTarget,
) -> PersistentFieldTarget:
    if target.topology is not None:
        return target
    topology = _build_target_topology(target)
    return PersistentFieldTarget(
        field=target.field,
        instance_ids=target.instance_ids,
        dense_valid_mask=target.dense_valid_mask,
        valid_instance_ids=target.valid_instance_ids,
        centers_zyx=target.centers_zyx,
        topology=topology,
        birth_core_ids=target.birth_core_ids,
        birth_moat_ids=target.birth_moat_ids,
        precompute_topology=False,
    )


def _spacing_array(spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive finite values")
    return spacing


def label_instances_26(wt: np.ndarray) -> np.ndarray:
    if np.asarray(wt).ndim != 3:
        raise ValueError("WT mask must be three-dimensional")
    instances, _ = label(
        np.asarray(wt, dtype=bool),
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    return instances.astype(np.int32, copy=False)


def deterministic_interior_center(
    mask: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> tuple[int, int, int]:
    component = np.asarray(mask, dtype=bool)
    if component.ndim != 3 or not component.any():
        raise ValueError("mask must be a nonempty three-dimensional component")
    spacing = _spacing_array(spacing_zyx)
    interior_distance = distance_transform_edt(component, sampling=spacing)
    maximum = float(interior_distance[component].max())
    candidates = np.argwhere(component & np.isclose(interior_distance, maximum))
    centroid = np.argwhere(component).mean(axis=0)
    squared_physical_distance = np.sum(((candidates - centroid) * spacing) ** 2, axis=1)
    nearest = candidates[
        np.isclose(squared_physical_distance, squared_physical_distance.min())
    ]
    return min(tuple(int(value) for value in point) for point in nearest)


def spacing_aware_geodesic(
    mask: np.ndarray,
    center_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    component = np.asarray(mask, dtype=bool)
    if component.ndim != 3 or not component.any():
        raise ValueError("mask must be a nonempty three-dimensional component")
    center = tuple(int(value) for value in center_zyx)
    if any(value < 0 or value >= component.shape[axis] for axis, value in enumerate(center)):
        raise ValueError("center_zyx lies outside the mask volume")
    if not component[center]:
        raise ValueError("center_zyx must lie inside the component")
    spacing = _spacing_array(spacing_zyx)
    coordinates = np.argwhere(component)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    cropped_component = component[slices]
    local_center = tuple(int(value - lower[axis]) for axis, value in enumerate(center))
    costs = np.where(cropped_component, 1.0, np.inf)
    cropped_distances, _ = MCP_Geometric(
        costs,
        fully_connected=True,
        sampling=tuple(float(value) for value in spacing),
    ).find_costs([local_center])
    distances = np.full(component.shape, np.inf, dtype=np.float64)
    distances[slices] = np.asarray(cropped_distances, dtype=np.float64)
    distances[~component] = np.inf
    return distances


def transform_centers_affine(
    centers_zyx: np.ndarray,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    affine: np.ndarray | None,
    center_location_in_pixels: tuple[float, float, float],
    mirror_axes: tuple[int, ...],
) -> np.ndarray:
    centers = np.asarray(centers_zyx, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_zyx must have shape [N, 3]")
    matrix = np.eye(3, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("affine must have shape [3, 3]")
    input_size = np.asarray(input_shape, dtype=np.float64)
    output_size = np.asarray(output_shape, dtype=np.float64)
    if input_size.shape != (3,) or output_size.shape != (3,):
        raise ValueError("input_shape and output_shape must be three-dimensional")

    input_centered = centers - (input_size - 1.0) / 2.0
    translation = np.asarray(center_location_in_pixels, dtype=np.float64) - input_size / 2.0
    output_centered = (input_centered - translation) @ np.linalg.inv(matrix)
    output_points = output_centered + (output_size - 1.0) / 2.0
    for axis in mirror_axes:
        if axis not in (0, 1, 2):
            raise ValueError("mirror_axes entries must be 0, 1, or 2")
        output_points[:, axis] = output_size[axis] - 1.0 - output_points[:, axis]
    return output_points


def _inside_topology_interior(
    point: np.ndarray, shape: tuple[int, int, int], margin: int
) -> bool:
    lower = np.full(3, margin, dtype=np.float64)
    upper = np.asarray(shape, dtype=np.float64) - margin
    return bool(np.isfinite(point).all() and np.all(point >= lower) and np.all(point < upper))


def _nearest_same_id(
    point: np.ndarray,
    instance_ids: np.ndarray,
    instance_id: int,
    spacing: np.ndarray,
) -> tuple[int, int, int]:
    candidates = np.argwhere(instance_ids == instance_id)
    if candidates.size == 0:
        raise ValueError(f"instance ID {instance_id} is absent from instance_ids")
    squared_distance = np.sum(((candidates - point) * spacing) ** 2, axis=1)
    nearest = candidates[np.isclose(squared_distance, squared_distance.min())]
    return min(tuple(int(value) for value in candidate) for candidate in nearest)


def build_persistent_field_target(
    instance_ids: np.ndarray,
    centers_by_id: dict[int, tuple[float, float, float]],
    spacing_zyx: tuple[float, float, float],
    topology_margin: int = 1,
    precompute_topology: bool = True,
) -> PersistentFieldTarget:
    instances = np.asarray(instance_ids)
    if instances.ndim != 3 or not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("instance_ids must be a three-dimensional integer array")
    if topology_margin < 0:
        raise ValueError("topology_margin must be nonnegative")
    spacing = _spacing_array(spacing_zyx)
    instances = instances.astype(np.int32, copy=False)
    field = np.full(instances.shape, -1.0, dtype=np.float32)
    dense_valid_mask = np.ones(instances.shape, dtype=bool)
    valid_ids: list[int] = []
    valid_centers: list[tuple[int, int, int]] = []

    present_ids = tuple(int(value) for value in np.unique(instances) if value > 0)
    for instance_id in present_ids:
        if instance_id not in centers_by_id:
            raise KeyError(f"Missing transformed center for instance ID {instance_id}")
        point = np.asarray(centers_by_id[instance_id], dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"Center for instance ID {instance_id} must have three coordinates")
        component = instances == instance_id
        if not _inside_topology_interior(point, instances.shape, topology_margin):
            dense_valid_mask[component] = False
            continue

        center = _nearest_same_id(point, instances, instance_id, spacing)
        if not _inside_topology_interior(np.asarray(center), instances.shape, topology_margin):
            dense_valid_mask[component] = False
            continue

        geodesic = spacing_aware_geodesic(component, center, spacing_zyx)
        reachable = component & np.isfinite(geodesic)
        unreachable = component & ~reachable
        dense_valid_mask[unreachable] = False
        maximum = float(geodesic[reachable].max())
        if maximum == 0.0:
            field[center] = 1.0
        else:
            field[reachable] = (
                0.25 + 0.75 * (1.0 - geodesic[reachable] / maximum)
            ).astype(np.float32)
        valid_ids.append(instance_id)
        valid_centers.append(center)

    return PersistentFieldTarget(
        field=torch.from_numpy(field),
        instance_ids=torch.from_numpy(instances.copy()),
        dense_valid_mask=torch.from_numpy(dense_valid_mask),
        valid_instance_ids=tuple(valid_ids),
        centers_zyx=tuple(valid_centers),
        precompute_topology=precompute_topology,
    )
