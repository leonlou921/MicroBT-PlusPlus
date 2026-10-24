#!/usr/bin/env python3
"""Build the frozen MicroBT-PF A0/A0-D/A1/A2/A3 causal matrix.

The evaluator consumes one raw per-case metric table per variant. Producing the
full-volume predictions is intentionally separate: this script only accepts a
complete, subject-aligned panel and never selects checkpoints from validation
scores. Utility functions below define the registered per-case metrics so an
inference driver can produce rows without duplicating metric semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


VARIANTS = ("A0", "A0-D", "A1", "A2", "A3")
REQUIRED_METRICS = (
    "wt_dice",
    "tc_dice",
    "et_dice",
    "lesion_mask_dice",
    "hd95_mm",
    "branch_precision",
    "branch_recall",
    "branch_f1",
    "count_mae",
    "exact_count",
    "split_count",
    "merge_count",
    "false_positives_per_case",
    "small_lesion_recall_3mm",
    "small_lesion_f1_3mm",
    "centroid_inside",
    "box_ap50",
    "box_ap75",
    "box_tightness",
    "volume_inflation",
)
COMPARISONS = (
    ("A0-D_minus_A0", "A0", "A0-D"),
    ("A1_minus_A0", "A0", "A1"),
    ("A2_minus_A1", "A1", "A2"),
    ("A3_minus_A2", "A2", "A3"),
)
HD95_QUERY_CHUNK_POINTS = 65_536


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_value(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def align_variant_rows(
    variant_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_count: int = 203,
) -> dict[str, list[dict[str, Any]]]:
    if tuple(variant_rows) != VARIANTS and set(variant_rows) != set(VARIANTS):
        raise ValueError(f"variants must be exactly {VARIANTS}")
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        rows = variant_rows[variant]
        if len(rows) != expected_count:
            raise ValueError(
                f"{variant} has {len(rows)} cases; exactly {expected_count} required"
            )
        by_subject: dict[str, dict[str, Any]] = {}
        for source_row in rows:
            row = dict(source_row)
            subject = str(row.get("subject_id", "")).strip()
            if not subject:
                raise ValueError(f"{variant} contains a blank subject_id")
            if subject in by_subject:
                raise ValueError(f"{variant} contains duplicate subject_id {subject}")
            row["subject_id"] = subject
            by_subject[subject] = row
        indexed[variant] = by_subject

    reference = set(indexed["A0"])
    for variant in VARIANTS[1:]:
        if set(indexed[variant]) != reference:
            missing = sorted(reference - set(indexed[variant]))
            extra = sorted(set(indexed[variant]) - reference)
            raise ValueError(
                f"{variant} subject IDs differ from A0; missing={missing[:5]}, extra={extra[:5]}"
            )
    subjects = sorted(reference)
    return {
        variant: [indexed[variant][subject] for subject in subjects]
        for variant in VARIANTS
    }


def summarize_metric(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, int | float | None]:
    values = [value for row in rows if (value := _finite_value(row, metric)) is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
    }


def paired_bootstrap_delta(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    seed: int = 42,
    samples: int = 10_000,
) -> dict[str, int | float | None]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    reference = {str(row["subject_id"]): row for row in reference_rows}
    candidate = {str(row["subject_id"]): row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise ValueError("paired rows must have identical subject IDs")
    differences: list[float] = []
    for subject in sorted(reference):
        before = _finite_value(reference[subject], metric)
        after = _finite_value(candidate[subject], metric)
        if before is not None and after is not None:
            differences.append(after - before)
    if not differences:
        return {
            "n": 0,
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    delta = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, delta.size, size=(samples, delta.size))
    bootstrap = delta[indices].mean(axis=1)
    low, high = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "n": int(delta.size),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_fixed_checkpoints(
    checkpoint_paths: Mapping[str, str | Path]
) -> dict[str, dict[str, Any]]:
    if set(checkpoint_paths) != set(VARIANTS):
        raise ValueError(f"checkpoint variants must be exactly {VARIANTS}")
    resolved: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        path = Path(checkpoint_paths[variant]).expanduser().resolve()
        if path.name != "checkpoint_final.pth":
            raise ValueError(f"{variant} must use checkpoint_final.pth")
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved[variant] = {
            "path": str(path),
            "name": path.name,
            "size": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    if (
        resolved["A0-D"]["path"] != resolved["A0"]["path"]
        or resolved["A0-D"]["sha256"] != resolved["A0"]["sha256"]
    ):
        raise ValueError("A0-D must reuse the exact A0 checkpoint path and bytes")
    return resolved


def build_causal_manifest(
    variant_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    checkpoint_paths: Mapping[str, str | Path],
    *,
    expected_count: int = 203,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    aligned = align_variant_rows(variant_rows, expected_count=expected_count)
    for variant, rows in aligned.items():
        for row in rows:
            missing = [metric for metric in REQUIRED_METRICS if metric not in row]
            if missing:
                raise ValueError(
                    f"{variant}/{row['subject_id']} missing metrics: {missing}"
                )
    raw = _native(aligned)
    summaries = {
        variant: {
            metric: summarize_metric(aligned[variant], metric)
            for metric in REQUIRED_METRICS
        }
        for variant in VARIANTS
    }
    comparisons = {
        name: {
            metric: paired_bootstrap_delta(
                aligned[reference],
                aligned[candidate],
                metric,
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            )
            for metric in REQUIRED_METRICS
        }
        for name, reference, candidate in COMPARISONS
    }
    return {
        "schema_version": 1,
        "raw_per_case": raw,
        "summaries": summaries,
        "comparisons": comparisons,
        "fixed_checkpoints": resolve_fixed_checkpoints(checkpoint_paths),
        "contract": {
            "variants": list(VARIANTS),
            "expected_subject_count": int(expected_count),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_samples": int(bootstrap_samples),
            "checkpoint_policy": "checkpoint_final_only",
            "a0d_policy": "exact_A0_checkpoint_and_logits",
            "metrics": list(REQUIRED_METRICS),
        },
    }


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum() + target.sum())
    return 1.0 if denominator == 0 else 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def _hd95(prediction: np.ndarray, target: np.ndarray, spacing: Sequence[float]) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("nan")
    union_coordinates = np.argwhere(prediction | target)
    lower = union_coordinates.min(axis=0)
    upper = union_coordinates.max(axis=0) + 1
    component_slice = tuple(
        slice(int(start), int(stop)) for start, stop in zip(lower, upper)
    )
    prediction = np.pad(
        prediction[component_slice], 1, mode="constant", constant_values=False
    )
    target = np.pad(
        target[component_slice], 1, mode="constant", constant_values=False
    )
    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    physical_spacing = np.asarray(spacing, dtype=np.float64)
    if physical_spacing.shape != (prediction.ndim,):
        raise ValueError(
            f"spacing must contain {prediction.ndim} values, got {tuple(physical_spacing)}"
        )
    if not np.all(np.isfinite(physical_spacing)) or np.any(physical_spacing <= 0):
        raise ValueError("spacing values must be positive and finite")
    pred_coordinates = np.argwhere(pred_surface).astype(np.float64, copy=False)
    target_coordinates = np.argwhere(target_surface).astype(np.float64, copy=False)
    pred_coordinates *= physical_spacing
    target_coordinates *= physical_spacing

    def query_in_chunks(tree: cKDTree, coordinates: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(coordinates), HD95_QUERY_CHUNK_POINTS):
            stop = min(start + HD95_QUERY_CHUNK_POINTS, len(coordinates))
            distances_chunk, _ = tree.query(
                coordinates[start:stop], k=1, workers=1
            )
            chunks.append(np.asarray(distances_chunk, dtype=np.float64))
        return np.concatenate(chunks)

    distances = np.concatenate(
        (
            query_in_chunks(cKDTree(target_coordinates), pred_coordinates),
            query_in_chunks(cKDTree(pred_coordinates), target_coordinates),
        )
    )
    return float(np.percentile(distances, 95.0))


def _instance_masks(instance_ids: np.ndarray) -> list[np.ndarray]:
    return [
        instance_ids == identifier
        for identifier in sorted(int(value) for value in np.unique(instance_ids) if value > 0)
    ]


@dataclass(frozen=True)
class _InstanceTable:
    identifiers: np.ndarray
    voxel_counts: np.ndarray
    boxes: np.ndarray
    rounded_centroids: np.ndarray

    def __len__(self) -> int:
        return int(self.identifiers.size)


def _instance_table(instance_ids: np.ndarray) -> _InstanceTable:
    """Return exact per-instance sufficient statistics without K full volumes."""
    array = np.asarray(instance_ids)
    foreground = array > 0
    if not foreground.any():
        return _InstanceTable(
            identifiers=np.empty(0, dtype=array.dtype),
            voxel_counts=np.empty(0, dtype=np.int64),
            boxes=np.empty((0, 6), dtype=np.float64),
            rounded_centroids=np.empty((0, 3), dtype=np.int64),
        )

    coordinates = np.nonzero(foreground)
    identifiers, inverse, counts = np.unique(
        array[foreground], return_inverse=True, return_counts=True
    )
    count = int(identifiers.size)
    lower = np.full((count, 3), np.iinfo(np.int64).max, dtype=np.int64)
    upper = np.full((count, 3), -1, dtype=np.int64)
    sums = np.empty((count, 3), dtype=np.float64)
    for axis, coordinate in enumerate(coordinates):
        coordinate = np.asarray(coordinate, dtype=np.int64)
        np.minimum.at(lower[:, axis], inverse, coordinate)
        np.maximum.at(upper[:, axis], inverse, coordinate)
        sums[:, axis] = np.bincount(
            inverse, weights=coordinate, minlength=count
        )
    boxes = np.concatenate((lower, upper + 1), axis=1).astype(np.float64)
    centroids = np.rint(sums / counts[:, None]).astype(np.int64)
    return _InstanceTable(
        identifiers=np.asarray(identifiers),
        voxel_counts=np.asarray(counts, dtype=np.int64),
        boxes=boxes,
        rounded_centroids=centroids,
    )


def _intersection_and_iou(
    predicted_instances: np.ndarray,
    target_instances: np.ndarray,
    predicted: _InstanceTable,
    target: _InstanceTable,
) -> tuple[np.ndarray, np.ndarray]:
    intersections = np.zeros((len(predicted), len(target)), dtype=np.int64)
    if len(predicted) and len(target):
        joint = (predicted_instances > 0) & (target_instances > 0)
        predicted_dense = np.searchsorted(
            predicted.identifiers, predicted_instances[joint]
        )
        target_dense = np.searchsorted(target.identifiers, target_instances[joint])
        pair_indices = predicted_dense * len(target) + target_dense
        intersections = np.bincount(
            pair_indices, minlength=len(predicted) * len(target)
        ).reshape(len(predicted), len(target))
    unions = (
        predicted.voxel_counts[:, None]
        + target.voxel_counts[None, :]
        - intersections
    )
    overlaps = np.divide(
        intersections,
        unions,
        out=np.zeros(intersections.shape, dtype=np.float64),
        where=unions > 0,
    )
    return intersections, overlaps


def _box(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    lower = coordinates.min(axis=0).astype(np.float64)
    upper = (coordinates.max(axis=0) + 1).astype(np.float64)
    return np.concatenate((lower, upper))


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    lower = np.maximum(first[:3], second[:3])
    upper = np.minimum(first[3:], second[3:])
    intersection = float(np.maximum(upper - lower, 0.0).prod())
    first_volume = float((first[3:] - first[:3]).prod())
    second_volume = float((second[3:] - second[:3]).prod())
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(
    boxes_prediction: Sequence[np.ndarray],
    boxes_target: Sequence[np.ndarray],
    scores: Sequence[float],
    threshold: float,
) -> float:
    if not len(boxes_target):
        return 1.0 if not len(boxes_prediction) else 0.0
    matched: set[int] = set()
    true_positive: list[float] = []
    false_positive: list[float] = []
    for index in sorted(range(len(boxes_prediction)), key=lambda item: (-float(scores[item]), item)):
        overlaps = [_box_iou(boxes_prediction[index], target_box) for target_box in boxes_target]
        best = int(np.argmax(overlaps)) if overlaps else -1
        is_true = best >= 0 and overlaps[best] >= threshold and best not in matched
        if is_true:
            matched.add(best)
        true_positive.append(float(is_true))
        false_positive.append(float(not is_true))
    if not true_positive:
        return 0.0
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / len(boxes_target)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def evaluate_case(
    subject_id: str,
    predicted_semantic: np.ndarray,
    predicted_instances: np.ndarray,
    target_semantic: np.ndarray,
    target_instances: np.ndarray | None = None,
    *,
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    object_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute the registered full-volume semantic and instance row."""
    predicted_semantic = np.asarray(predicted_semantic)
    target_semantic = np.asarray(target_semantic)
    predicted_instances = np.asarray(predicted_instances)
    if not (
        predicted_semantic.shape == target_semantic.shape == predicted_instances.shape
    ):
        raise ValueError("semantic and instance volumes must have identical shapes")
    if target_instances is None:
        target_instances = label(target_semantic > 0)[0]
    target_instances = np.asarray(target_instances)
    if target_instances.shape != predicted_instances.shape:
        raise ValueError("target instance volume shape mismatch")

    predicted = _instance_table(predicted_instances)
    target = _instance_table(target_instances)
    scores = list(object_scores) if object_scores is not None else [1.0] * len(predicted)
    if len(scores) != len(predicted):
        raise ValueError("object score count must equal predicted instance count")

    intersections, overlap = _intersection_and_iou(
        predicted_instances, target_instances, predicted, target
    )
    matches: list[tuple[int, int]] = []
    if overlap.size:
        pred_indices, target_indices = linear_sum_assignment(-overlap)
        matches = [
            (int(pred), int(target))
            for pred, target in zip(pred_indices, target_indices)
            if overlap[pred, target] > 0.0
        ]
    matched_pred = {pred for pred, _ in matches}
    matched_target = {target for _, target in matches}
    precision = len(matches) / len(predicted) if len(predicted) else float(not len(target))
    recall = len(matches) / len(target) if len(target) else float(not len(predicted))
    branch_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    voxel_volume = float(spacing.prod())
    target_diameters = (
        6.0 * target.voxel_counts.astype(np.float64) * voxel_volume / math.pi
    ) ** (1.0 / 3.0)
    pred_diameters = (
        6.0 * predicted.voxel_counts.astype(np.float64) * voxel_volume / math.pi
    ) ** (1.0 / 3.0)
    small_target = {index for index, diameter in enumerate(target_diameters) if diameter <= 3.0}
    small_pred = {index for index, diameter in enumerate(pred_diameters) if diameter <= 3.0}
    small_matches = [(pred, target) for pred, target in matches if target in small_target]
    small_recall = len(small_matches) / len(small_target) if small_target else float("nan")
    small_precision = len({pred for pred, _ in small_matches if pred in small_pred}) / len(small_pred) if small_pred else (1.0 if not small_target else 0.0)
    small_f1 = 2.0 * small_precision * small_recall / (small_precision + small_recall) if math.isfinite(small_recall) and small_precision + small_recall else (0.0 if math.isfinite(small_recall) else float("nan"))

    centroid_inside_values = []
    inflation = []
    for pred_index, target_index in matches:
        centroid = predicted.rounded_centroids[pred_index]
        centroid_inside_values.append(
            float(
                target_instances[tuple(centroid)]
                == target.identifiers[target_index]
            )
        )
        inflation.append(
            float(predicted.voxel_counts[pred_index])
            / float(target.voxel_counts[target_index])
        )
    box_volumes = np.prod(predicted.boxes[:, 3:] - predicted.boxes[:, :3], axis=1)
    tightness = predicted.voxel_counts / box_volumes

    prediction_wt = predicted_semantic > 0
    target_wt = target_semantic > 0
    row = {
        "subject_id": str(subject_id),
        "wt_dice": _dice(prediction_wt, target_wt),
        "tc_dice": _dice(predicted_semantic >= 2, target_semantic >= 2),
        "et_dice": _dice(predicted_semantic == 3, target_semantic == 3),
        "lesion_mask_dice": _dice(predicted_instances > 0, target_instances > 0),
        "hd95_mm": _hd95(predicted_instances > 0, target_instances > 0, spacing),
        "branch_precision": float(precision),
        "branch_recall": float(recall),
        "branch_f1": float(branch_f1),
        "count_mae": float(abs(len(predicted) - len(target))),
        "exact_count": float(len(predicted) == len(target)),
        "split_count": float(np.sum(np.sum(intersections > 0, axis=0) > 1)),
        "merge_count": float(np.sum(np.sum(intersections > 0, axis=1) > 1)),
        "false_positives_per_case": float(len(predicted) - len(matched_pred)),
        "small_lesion_recall_3mm": float(small_recall),
        "small_lesion_f1_3mm": float(small_f1),
        "centroid_inside": float(np.mean(centroid_inside_values)) if centroid_inside_values else float("nan"),
        "box_ap50": _average_precision(predicted.boxes, target.boxes, scores, 0.50),
        "box_ap75": _average_precision(predicted.boxes, target.boxes, scores, 0.75),
        "box_tightness": float(np.mean(tightness)) if len(tightness) else float("nan"),
        "volume_inflation": float(np.mean(inflation)) if inflation else float("nan"),
    }
    return row


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=203)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    variant_rows = json.loads(arguments.rows.read_text(encoding="utf-8"))
    checkpoint_paths = json.loads(arguments.checkpoints.read_text(encoding="utf-8"))
    manifest = build_causal_manifest(
        variant_rows,
        checkpoint_paths,
        expected_count=arguments.expected_count,
        bootstrap_seed=arguments.bootstrap_seed,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
