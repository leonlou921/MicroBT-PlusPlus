from __future__ import annotations

import numpy as np
import torch

from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    _inside_topology_interior,
    _spacing_array,
)


def build_centroid_transport_target(
    instance_ids: np.ndarray,
    centers_by_id: dict[int, tuple[float, float, float]],
    spacing_zyx: tuple[float, float, float],
    topology_margin: int = 1,
    precompute_topology: bool = False,
) -> PersistentFieldTarget:
    """Build only the instance IDs and interior centres consumed by CTF.

    The valid IDs and nearest same-instance centres exactly follow the legacy
    persistent-field target contract, but no geodesic scalar field or topology
    structure is constructed.
    """
    instances = np.asarray(instance_ids)
    if instances.ndim != 3 or not np.issubdtype(instances.dtype, np.integer):
        raise ValueError("instance_ids must be a three-dimensional integer array")
    if topology_margin < 0:
        raise ValueError("topology_margin must be nonnegative")
    if precompute_topology:
        raise ValueError("centroid transport targets never precompute topology")
    spacing = _spacing_array(spacing_zyx)
    instances = instances.astype(np.int32, copy=False)
    dense_valid_mask = np.ones(instances.shape, dtype=bool)
    valid_ids: list[int] = []
    valid_centers: list[tuple[int, int, int]] = []

    positive_coords = np.argwhere(instances > 0)
    positive_ids = instances[tuple(positive_coords.T)] if positive_coords.size else np.empty(0, dtype=np.int32)
    present_ids = tuple(int(value) for value in np.unique(positive_ids))
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

        candidates = positive_coords[positive_ids == instance_id]
        squared_distance = np.sum(((candidates - point) * spacing) ** 2, axis=1)
        nearest_candidates = candidates[
            np.isclose(squared_distance, squared_distance.min())
        ]
        center = min(
            tuple(int(value) for value in candidate)
            for candidate in nearest_candidates
        )
        if not _inside_topology_interior(
            np.asarray(center), instances.shape, topology_margin
        ):
            dense_valid_mask[component] = False
            continue
        valid_ids.append(instance_id)
        valid_centers.append(center)

    return PersistentFieldTarget(
        field=torch.zeros(instances.shape, dtype=torch.float32),
        instance_ids=torch.from_numpy(instances.copy()),
        dense_valid_mask=torch.from_numpy(dense_valid_mask),
        valid_instance_ids=tuple(valid_ids),
        centers_zyx=tuple(valid_centers),
        precompute_topology=False,
    )
