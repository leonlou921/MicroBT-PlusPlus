from __future__ import annotations

import numpy as np
from scipy.ndimage import label, watershed_ift

from nnunetv2.topology.persistent_field_0d import (
    PersistencePairs0D,
    persistence_pairs_0d,
)


def _field(value: np.ndarray, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be three-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def decode_support_preserving_birth_field(
    birth_field: np.ndarray,
    whole_tumor_probability: np.ndarray,
    field_min: float = 0.25,
    persistence_min: float = 0.25,
    support_threshold: float = 0.5,
) -> dict:
    """Partition semantic support without ever adding or removing support voxels."""
    birth = _field(birth_field, "birth_field")
    support_probability = _field(whole_tumor_probability, "whole_tumor_probability")
    if support_probability.shape != birth.shape:
        raise ValueError("birth field and whole-tumor probability must have equal shapes")

    pairs = persistence_pairs_0d(birth, floor=-1.0)
    flat = birth.ravel()
    birth_values = flat[pairs.birth_flat.astype(np.int64)]
    death_values = np.full(pairs.death_flat.size, -1.0, dtype=np.float32)
    finite = pairs.death_flat >= 0
    death_values[finite] = flat[pairs.death_flat[finite]]
    persistence = birth_values - death_values
    retained = np.flatnonzero(
        (birth_values >= float(field_min))
        & (persistence >= float(persistence_min))
    )
    retained = retained[np.argsort(pairs.birth_flat[retained], kind="stable")]

    support = support_probability >= float(support_threshold)
    retained_birth_flat = pairs.birth_flat[retained].astype(np.int64, copy=False)
    marker_outside_support = int((~support.ravel()[retained_birth_flat]).sum())
    if marker_outside_support:
        raise RuntimeError(
            f"{marker_outside_support} retained SPP marker(s) are outside semantic support"
        )

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    support_components, component_count = label(support, structure=structure)
    markers = np.zeros(birth.shape, dtype=np.int32)
    records: list[dict] = []
    for pair_index in retained.tolist():
        output_id = len(records) + 1
        birth_flat = int(pairs.birth_flat[pair_index])
        markers.ravel()[birth_flat] = output_id
        records.append(
            {
                "output_id": output_id,
                "pair_index": int(pair_index),
                "birth_flat": birth_flat,
                "fallback_marker": False,
            }
        )

    fallback_marker_count = 0
    for component_id in range(1, int(component_count) + 1):
        component = support_components == component_id
        if np.any(markers[component] > 0):
            continue
        component_flat = np.flatnonzero(component.ravel())
        component_values = flat[component_flat]
        maximum = float(component_values.max())
        birth_flat = int(component_flat[np.flatnonzero(component_values == maximum)[0]])
        output_id = len(records) + 1
        markers.ravel()[birth_flat] = output_id
        records.append(
            {
                "output_id": output_id,
                "pair_index": None,
                "birth_flat": birth_flat,
                "fallback_marker": True,
            }
        )
        fallback_marker_count += 1

    surface = np.rint((1.0 - np.clip(birth, -1.0, 1.0)) * 32767.0).astype(
        np.uint16
    )
    instance_ids = np.zeros(birth.shape, dtype=np.int32)
    for component_id in range(1, int(component_count) + 1):
        component = support_components == component_id
        component_markers = np.unique(markers[component])
        component_markers = component_markers[component_markers > 0]
        if component_markers.size == 1:
            instance_ids[component] = int(component_markers[0])
            continue
        coordinates = np.argwhere(component)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        local_component = component[crop]
        local_surface = surface[crop].copy()
        local_surface[~local_component] = np.iinfo(np.uint16).max
        local_markers = markers[crop].copy()
        local_labels = watershed_ift(local_surface, local_markers, structure=structure)
        destination = instance_ids[crop]
        destination[local_component] = local_labels[local_component]

    if not np.array_equal(instance_ids > 0, support):
        raise RuntimeError("SPP watershed failed support-conservation invariant")

    masks: list[np.ndarray] = []
    objects: list[dict] = []
    for record in records:
        output_id = int(record["output_id"])
        mask = instance_ids == output_id
        if not bool(mask.any()):
            raise RuntimeError("SPP marker received no watershed support")
        masks.append(mask)
        coordinates = np.argwhere(mask)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        birth_flat = int(record["birth_flat"])
        pair_index = record["pair_index"]
        if pair_index is None:
            birth_value = float(flat[birth_flat])
            death_value = -1.0
            persistence_value = birth_value + 1.0
        else:
            birth_value = float(birth_values[pair_index])
            death_value = float(death_values[pair_index])
            persistence_value = float(persistence[pair_index])
        centroid = np.unravel_index(birth_flat, birth.shape)
        objects.append(
            {
                "instance_id": output_id,
                "birth_flat": birth_flat,
                "birth": birth_value,
                "death": death_value,
                "persistence": persistence_value,
                "fallback_marker": bool(record["fallback_marker"]),
                "centroid_zyx": [int(value) for value in centroid],
                "box_zyx": [
                    int(lower[0]),
                    int(lower[1]),
                    int(lower[2]),
                    int(upper[0]),
                    int(upper[1]),
                    int(upper[2]),
                ],
                "volume": int(mask.sum()),
            }
        )
    return {
        "count": len(objects),
        "instance_ids": instance_ids,
        "masks": tuple(masks),
        "objects": tuple(objects),
        "marker_outside_support": marker_outside_support,
        "retained_marker_count": int(retained.size),
        "fallback_marker_count": int(fallback_marker_count),
        "support_voxel_count": int(support.sum()),
        "support_conserved": True,
    }
