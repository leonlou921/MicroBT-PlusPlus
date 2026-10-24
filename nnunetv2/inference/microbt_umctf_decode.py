from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, maximum_filter

from nnunetv2.inference.microbt_ctf_decode import instance_boxes


@dataclass(frozen=True)
class UnitMassDecodedInstances:
    centers_zyx: tuple[tuple[int, int, int], ...]
    center_scores: tuple[float, ...]
    instance_labels: np.ndarray
    atom_map: np.ndarray
    total_input_mass: float
    total_atom_mass: float

    @property
    def count(self) -> int:
        return len(self.centers_zyx)


def _trilinear_splat(
    shape: tuple[int, int, int], endpoints: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    atom_map = np.zeros(shape, dtype=np.float32)
    lower = np.floor(endpoints).astype(np.int64)
    fraction = endpoints - lower
    upper_bound = np.asarray(shape, dtype=np.int64) - 1
    for corner in range(8):
        indices = lower.copy()
        weight = np.ones(len(endpoints), dtype=np.float32)
        for axis in range(3):
            take_upper = bool(corner & (1 << axis))
            if take_upper:
                indices[:, axis] += 1
                weight *= fraction[:, axis]
            else:
                weight *= 1.0 - fraction[:, axis]
        indices = np.clip(indices, 0, upper_bound)
        np.add.at(
            atom_map,
            (indices[:, 0], indices[:, 1], indices[:, 2]),
            masses * weight,
        )
    return atom_map


def decode_umctf_instances(
    support: np.ndarray,
    density: np.ndarray,
    transport: np.ndarray,
    *,
    offset_scale_zyx: tuple[float, float, float] = (56.0, 80.0, 64.0),
    support_threshold: float = 0.10,
    density_threshold: float = 1e-6,
    min_atom_mass: float = 0.25,
    nms_radius: int = 2,
    max_centers: int = 64,
) -> UnitMassDecodedInstances:
    """Push conserved density to endpoints and decode one local atom per lesion."""
    support = np.asarray(support, dtype=np.float32)
    density = np.asarray(density, dtype=np.float32)
    transport = np.asarray(transport, dtype=np.float32)
    if support.ndim != 3 or density.shape != support.shape:
        raise ValueError("support and density must be aligned three-dimensional arrays")
    if transport.shape != (3, *support.shape):
        raise ValueError("transport must have shape [3,D,H,W]")
    if not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("density must be finite and nonnegative")
    if not np.isfinite(transport).all():
        raise ValueError("transport must be finite")
    scale = np.asarray(offset_scale_zyx, dtype=np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("offset_scale_zyx must contain three positive finite values")
    if not 0 <= support_threshold <= 1 or density_threshold < 0:
        raise ValueError("support and density thresholds are invalid")
    if min_atom_mass <= 0 or nms_radius < 0 or max_centers <= 0:
        raise ValueError("atom, NMS, and centre limits must be positive")

    mass_mask = density > float(density_threshold)
    mass_coords = np.argwhere(mass_mask)
    labels = np.zeros(support.shape, dtype=np.int32)
    empty_map = np.zeros(support.shape, dtype=np.float32)
    if mass_coords.size == 0:
        return UnitMassDecodedInstances((), (), labels, empty_map, 0.0, 0.0)

    mass_values = density[mass_mask]
    mass_endpoints = (
        mass_coords.astype(np.float32) + transport[:, mass_mask].T * scale[None]
    )
    mass_endpoints = np.clip(
        mass_endpoints, 0.0, (np.asarray(support.shape) - 1).astype(np.float32)
    )
    atom_map = _trilinear_splat(support.shape, mass_endpoints, mass_values)

    window = 2 * int(nms_radius) + 1
    local_max = maximum_filter(atom_map, size=window, mode="constant")
    plateau = (atom_map > 0) & np.isclose(atom_map, local_max)
    plateau_labels, plateau_count = label(plateau)
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for plateau_id in range(1, plateau_count + 1):
        coords = np.argwhere(plateau_labels == plateau_id)
        weights = atom_map[tuple(coords.T)]
        center_float = np.average(coords.astype(np.float64), axis=0, weights=weights)
        center = tuple(
            int(v) for v in np.clip(np.rint(center_float), 0, np.asarray(support.shape) - 1)
        )
        lo = np.maximum(np.asarray(center) - int(nms_radius), 0)
        hi = np.minimum(np.asarray(center) + int(nms_radius) + 1, np.asarray(support.shape))
        score = float(atom_map[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum())
        if score >= float(min_atom_mass):
            candidates.append((score, center))
    candidates.sort(key=lambda item: (-item[0], *item[1]))
    candidates = candidates[: int(max_centers)]
    if not candidates:
        return UnitMassDecodedInstances(
            (), (), labels, atom_map, float(mass_values.sum()), float(atom_map.sum())
        )

    centers_array = np.asarray([item[1] for item in candidates], dtype=np.float32)
    support_mask = np.isfinite(support) & (support >= float(support_threshold))
    support_coords = np.argwhere(support_mask)
    if support_coords.size:
        support_endpoints = (
            support_coords.astype(np.float32)
            + transport[:, support_mask].T * scale[None]
        )
        distance = ((support_endpoints[:, None] - centers_array[None]) ** 2).sum(2)
        assignments = distance.argmin(1).astype(np.int32) + 1
        labels[tuple(support_coords.T)] = assignments
    return UnitMassDecodedInstances(
        tuple(item[1] for item in candidates),
        tuple(item[0] for item in candidates),
        labels,
        atom_map,
        float(mass_values.sum()),
        float(atom_map.sum()),
    )


__all__ = ["UnitMassDecodedInstances", "decode_umctf_instances", "instance_boxes"]
