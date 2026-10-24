from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class CTFDecodedInstances:
    centers_zyx: tuple[tuple[int, int, int], ...]
    center_scores: tuple[float, ...]
    instance_labels: np.ndarray

    @property
    def count(self) -> int:
        return len(self.centers_zyx)


def decode_ctf_instances(
    support: np.ndarray,
    transport: np.ndarray,
    *,
    offset_scale_zyx: tuple[float, float, float] = (56.0, 80.0, 64.0),
    support_threshold: float = 0.10,
    min_vote_mass: float = 0.25,
    nms_radius: int = 2,
    max_centers: int = 64,
) -> CTFDecodedInstances:
    """Convert support-weighted centroid endpoints into instances and centres."""
    support = np.asarray(support, dtype=np.float32)
    transport = np.asarray(transport, dtype=np.float32)
    if support.ndim != 3:
        raise ValueError("support must be three-dimensional")
    if transport.shape != (3, *support.shape):
        raise ValueError("transport must have shape [3,D,H,W]")
    scale = np.asarray(offset_scale_zyx, dtype=np.float32)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("offset_scale_zyx must contain three positive finite values")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be in [0, 1]")
    if min_vote_mass <= 0.0 or nms_radius < 0 or max_centers <= 0:
        raise ValueError("vote, NMS, and centre limits must be positive")

    foreground = np.isfinite(support) & (support >= float(support_threshold))
    coords = np.argwhere(foreground)
    labels = np.zeros(support.shape, dtype=np.int32)
    if coords.size == 0:
        return CTFDecodedInstances((), (), labels)

    offsets = transport[:, foreground].T * scale[None]
    endpoints = np.rint(coords.astype(np.float32) + offsets).astype(np.int64)
    upper = np.asarray(support.shape, dtype=np.int64) - 1
    endpoints = np.clip(endpoints, 0, upper)
    vote_map = np.zeros(support.shape, dtype=np.float32)
    np.add.at(
        vote_map,
        (endpoints[:, 0], endpoints[:, 1], endpoints[:, 2]),
        support[foreground],
    )

    window = 2 * int(nms_radius) + 1
    local_max = maximum_filter(vote_map, size=window, mode="constant")
    peak_coords = np.argwhere(
        (vote_map >= float(min_vote_mass)) & (vote_map == local_max)
    )
    if peak_coords.size == 0:
        return CTFDecodedInstances((), (), labels)
    scores = vote_map[peak_coords[:, 0], peak_coords[:, 1], peak_coords[:, 2]]
    order = np.lexsort(
        (peak_coords[:, 2], peak_coords[:, 1], peak_coords[:, 0], -scores)
    )[: int(max_centers)]
    peak_coords = peak_coords[order]
    scores = scores[order]

    distance = ((endpoints[:, None, :] - peak_coords[None, :, :]) ** 2).sum(axis=2)
    assignments = distance.argmin(axis=1).astype(np.int32) + 1
    labels[coords[:, 0], coords[:, 1], coords[:, 2]] = assignments
    centers = tuple(tuple(int(v) for v in row) for row in peak_coords)
    return CTFDecodedInstances(centers, tuple(float(v) for v in scores), labels)


def instance_boxes(instance_labels: np.ndarray) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return exclusive-max ZYX boxes derived from assigned instance voxels."""
    labels = np.asarray(instance_labels)
    if labels.ndim != 3:
        raise ValueError("instance_labels must be three-dimensional")
    boxes: list[tuple[int, int, int, int, int, int]] = []
    for instance_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        coords = np.argwhere(labels == instance_id)
        lower = coords.min(axis=0)
        upper = coords.max(axis=0) + 1
        boxes.append(tuple(int(v) for v in np.concatenate((lower, upper))))
    return tuple(boxes)
