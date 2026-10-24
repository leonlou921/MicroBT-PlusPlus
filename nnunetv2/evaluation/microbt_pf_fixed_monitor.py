from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label
from scipy.optimize import linear_sum_assignment

from scripts.evaluate_microbt_pf_causal_matrix import evaluate_case


def label_instances_26(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 3:
        raise ValueError("instance labeling requires one 3D mask")
    instances, _ = label(array, structure=np.ones((3, 3, 3), dtype=np.uint8))
    return instances.astype(np.int32, copy=False)


def _instance_masks(instance_ids: np.ndarray) -> list[tuple[int, np.ndarray]]:
    array = np.asarray(instance_ids)
    return [
        (identifier, array == identifier)
        for identifier in sorted(int(value) for value in np.unique(array) if value > 0)
    ]


def _equivalent_diameter_mm(mask: np.ndarray, spacing_zyx: Sequence[float]) -> float:
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    return (6.0 * float(np.asarray(mask, dtype=bool).sum()) * voxel_volume / math.pi) ** (1.0 / 3.0)


@dataclass(frozen=True)
class TargetGeometry:
    identifiers: tuple[int, ...]
    centers_zyx: np.ndarray
    equivalent_diameters_mm: tuple[float, ...]
    component_slices: tuple[tuple[slice, ...], ...]


def build_target_geometry(
    target_instances: np.ndarray, spacing_zyx: Sequence[float]
) -> TargetGeometry:
    """Compute reusable GT geometry with exact component-local EDTs."""
    array = np.asarray(target_instances)
    if array.ndim != 3:
        raise ValueError("target geometry requires one 3D instance volume")
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("target geometry spacing must contain three positive values")
    voxel_volume = float(spacing.prod())
    identifiers: list[int] = []
    centers: list[np.ndarray] = []
    diameters: list[float] = []
    component_slices: list[tuple[slice, ...]] = []
    for zero_based, component_slice in enumerate(find_objects(array)):
        if component_slice is None:
            continue
        identifier = zero_based + 1
        expanded_slice = tuple(
            slice(
                max(0, int(axis.start or 0) - 1),
                min(array.shape[dimension], int(axis.stop) + 1),
            )
            for dimension, axis in enumerate(component_slice)
        )
        component = array[expanded_slice] == identifier
        if not component.any():
            continue
        distance = distance_transform_edt(component, sampling=spacing)
        maximum = float(distance[component].max())
        local_center = np.argwhere(component & (distance == maximum))[0]
        origin = np.asarray([int(axis.start or 0) for axis in expanded_slice])
        identifiers.append(identifier)
        centers.append((local_center + origin).astype(np.float64))
        diameters.append(
            (6.0 * float(component.sum()) * voxel_volume / math.pi) ** (1.0 / 3.0)
        )
        component_slices.append(tuple(component_slice))
    return TargetGeometry(
        identifiers=tuple(identifiers),
        centers_zyx=np.stack(centers) if centers else np.empty((0, 3), dtype=np.float64),
        equivalent_diameters_mm=tuple(float(value) for value in diameters),
        component_slices=tuple(component_slices),
    )


def _target_centers(
    target_instances: np.ndarray, spacing_zyx: Sequence[float]
) -> tuple[list[int], np.ndarray]:
    geometry = build_target_geometry(target_instances, spacing_zyx)
    return list(geometry.identifiers), geometry.centers_zyx


def proposal_recall_at_radius(
    target_instances: np.ndarray,
    proposal_centroids_zyx: Sequence[Sequence[float]],
    *,
    spacing_zyx: Sequence[float],
    radius_mm: float,
    eligible_target_ids: set[int] | None = None,
    target_geometry: TargetGeometry | None = None,
) -> float:
    if float(radius_mm) <= 0:
        raise ValueError("proposal radius must be positive")
    geometry = target_geometry or build_target_geometry(target_instances, spacing_zyx)
    identifiers = list(geometry.identifiers)
    target_centers = geometry.centers_zyx
    if eligible_target_ids is not None:
        selection = [
            index for index, identifier in enumerate(identifiers)
            if identifier in eligible_target_ids
        ]
        target_centers = target_centers[selection]
    proposals = np.asarray(proposal_centroids_zyx, dtype=np.float64)
    if proposals.size == 0:
        proposals = np.empty((0, 3), dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[1] != 3:
        raise ValueError("proposal centroids must have shape [N, 3]")
    if target_centers.shape[0] == 0:
        return float("nan") if eligible_target_ids is not None else float(proposals.shape[0] == 0)
    if proposals.shape[0] == 0:
        return 0.0
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    distances = np.linalg.norm(
        (proposals[:, None, :] - target_centers[None, :, :]) * spacing,
        axis=2,
    )
    proposal_indices, target_indices = linear_sum_assignment(distances)
    matched = sum(
        float(distances[prediction, target]) <= float(radius_mm)
        for prediction, target in zip(proposal_indices, target_indices)
    )
    return float(matched) / float(target_centers.shape[0])


def _field_metrics(
    field: np.ndarray,
    target_instances: np.ndarray,
    spacing_zyx: Sequence[float],
    *,
    target_geometry: TargetGeometry | None = None,
) -> dict[str, float]:
    field = np.asarray(field, dtype=np.float32)
    geometry = target_geometry or build_target_geometry(target_instances, spacing_zyx)
    identifiers = list(geometry.identifiers)
    centers = geometry.centers_zyx
    center_values = [float(field[tuple(center.astype(int))]) for center in centers]
    peak_distances: list[float] = []
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    for identifier, center, component_slice in zip(
        identifiers, centers, geometry.component_slices
    ):
        component = np.asarray(target_instances)[component_slice] == identifier
        local_field = field[component_slice]
        maximum = float(local_field[component].max())
        local_peak = np.argwhere(component & (local_field == maximum))[0]
        origin = np.asarray([int(axis.start or 0) for axis in component_slice])
        peak = (local_peak + origin).astype(np.float64)
        peak_distances.append(float(np.linalg.norm((peak - center) * spacing)))
    background = field[np.asarray(target_instances) == 0]
    return {
        "field_gt_center_mean": float(np.mean(center_values)) if center_values else float("nan"),
        "field_background_p95": float(np.percentile(background, 95.0)) if background.size else float("nan"),
        "field_peak_center_distance_mm": float(np.mean(peak_distances)) if peak_distances else float("nan"),
    }


def evaluate_fixed_case(
    subject_id: str,
    prediction: Mapping[str, Any],
    target_semantic: np.ndarray,
    *,
    spacing_zyx: Sequence[float],
    expected_count: int | None = None,
    expected_small_count: int | None = None,
    small_diameter_mm: float = 3.0,
) -> dict[str, Any]:
    target_semantic = np.asarray(target_semantic)
    target_instances = label_instances_26(target_semantic > 0)
    target_geometry = build_target_geometry(target_instances, spacing_zyx)
    predicted_semantic = np.asarray(prediction["segmentation"])
    predicted_instances = np.asarray(prediction["instance_ids"])
    field = np.asarray(prediction["field"])
    if not (
        predicted_semantic.shape
        == predicted_instances.shape
        == field.shape
        == target_semantic.shape
    ):
        raise ValueError("fixed-case prediction and target shapes differ")

    target_count = len(target_geometry.identifiers)
    small_target_ids = {
        identifier
        for identifier, diameter in zip(
            target_geometry.identifiers,
            target_geometry.equivalent_diameters_mm,
        )
        if diameter <= float(small_diameter_mm)
    }
    if expected_count is not None and target_count != int(expected_count):
        raise ValueError(
            f"{subject_id} target count changed: {target_count} != {expected_count}"
        )
    if expected_small_count is not None and len(small_target_ids) != int(expected_small_count):
        raise ValueError(
            f"{subject_id} small target count changed: "
            f"{len(small_target_ids)} != {expected_small_count}"
        )

    objects = tuple(prediction.get("objects", ()))
    centroids = tuple(object_["centroid_zyx"] for object_ in objects)
    scores = tuple(
        float(object_["birth"]) - float(object_["death"]) for object_ in objects
    )
    predicted_count = int(np.count_nonzero(np.unique(predicted_instances) > 0))
    if len(objects) != predicted_count or int(prediction.get("count", predicted_count)) != predicted_count:
        raise ValueError("decoded object and instance counts differ")

    row = evaluate_case(
        subject_id,
        predicted_semantic,
        predicted_instances,
        target_semantic,
        target_instances,
        spacing_zyx=spacing_zyx,
        object_scores=scores,
    )
    row.update(
        {
            "target_count": int(target_count),
            "predicted_count": int(predicted_count),
            "signed_count_error": int(predicted_count - target_count),
            "target_small_count_3mm": int(len(small_target_ids)),
            "decoded_persistence_mean": float(np.mean(scores)) if scores else float("nan"),
            "proposal_recall_3mm": proposal_recall_at_radius(
                target_instances,
                centroids,
                spacing_zyx=spacing_zyx,
                radius_mm=3.0,
                target_geometry=target_geometry,
            ),
            "proposal_recall_5mm": proposal_recall_at_radius(
                target_instances,
                centroids,
                spacing_zyx=spacing_zyx,
                radius_mm=5.0,
                target_geometry=target_geometry,
            ),
            "small_proposal_recall_3mm": proposal_recall_at_radius(
                target_instances,
                centroids,
                spacing_zyx=spacing_zyx,
                radius_mm=3.0,
                eligible_target_ids=small_target_ids,
                target_geometry=target_geometry,
            ),
            "small_proposal_recall_5mm": proposal_recall_at_radius(
                target_instances,
                centroids,
                spacing_zyx=spacing_zyx,
                radius_mm=5.0,
                eligible_target_ids=small_target_ids,
                target_geometry=target_geometry,
            ),
        }
    )
    row.update(
        _field_metrics(
            field,
            target_instances,
            spacing_zyx,
            target_geometry=target_geometry,
        )
    )
    return row


def summarize_fixed_cases(
    rows: Sequence[Mapping[str, Any]], *, epoch: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("fixed-case rows must not be empty")
    metric_names = sorted(
        set().union(*(row.keys() for row in rows)) - {"subject_id"}
    )
    macro: dict[str, float | None] = {}
    for metric in metric_names:
        values: list[float] = []
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
                values.append(float(value))
        macro[metric] = float(np.mean(values)) if values else None
    return {
        "schema_version": 1,
        "epoch": int(epoch),
        "case_count": len(rows),
        "totals": {
            "target_count": int(sum(int(row["target_count"]) for row in rows)),
            "predicted_count": int(sum(int(row["predicted_count"]) for row in rows)),
        },
        "macro": macro,
    }


def validate_fixed_case_manifest(
    manifest: Mapping[str, Any],
    *,
    split_sha256: str,
    validation_subjects: set[str],
) -> dict[str, Any]:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported fixed-case manifest schema")
    if int(manifest.get("connectivity", -1)) != 26:
        raise ValueError("fixed-case monitor requires 26-connectivity")
    if float(manifest.get("small_diameter_mm", -1)) != 3.0:
        raise ValueError("fixed-case monitor requires the 3mm small-lesion contract")
    decoder = manifest.get("decoder", {})
    if decoder != {"field_min": 0.25, "foreground_level": 0.0}:
        raise ValueError("fixed-case decoder contract changed")
    if str(manifest.get("split_sha256")) != str(split_sha256):
        raise ValueError("fixed-case split SHA256 mismatch")
    milestone = int(manifest.get("milestone_every_epochs", 0))
    if milestone <= 0:
        raise ValueError("milestone interval must be positive")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixed-case manifest must contain cases")
    case_ids = [str(case.get("subject_id", "")).strip() for case in cases]
    if any(not subject for subject in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("fixed-case IDs must be unique and nonblank")
    missing = sorted(set(case_ids) - set(validation_subjects))
    if missing:
        raise ValueError(f"fixed cases are outside fold validation: {missing}")
    for case in cases:
        if int(case.get("expected_count", -1)) < 0:
            raise ValueError("fixed case expected_count must be nonnegative")
        if int(case.get("expected_small_count", -1)) < 0:
            raise ValueError("fixed case expected_small_count must be nonnegative")
    return {
        "case_ids": case_ids,
        "milestone_every_epochs": milestone,
        "small_diameter_mm": 3.0,
        "decoder": dict(decoder),
    }
