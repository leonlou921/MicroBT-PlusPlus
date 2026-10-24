from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nnunetv2.evaluation.microbt_pf_fixed_monitor import (
    evaluate_fixed_case,
    label_instances_26,
)


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_native(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def _instance_masks(instance_ids: np.ndarray) -> list[tuple[int, np.ndarray]]:
    array = np.asarray(instance_ids)
    return [
        (identifier, array == identifier)
        for identifier in sorted(int(value) for value in np.unique(array) if value > 0)
    ]


def _box(mask: np.ndarray) -> list[int]:
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        raise ValueError("cannot make a box from an empty mask")
    return [
        *coordinates.min(axis=0).astype(int).tolist(),
        *(coordinates.max(axis=0) + 1).astype(int).tolist(),
    ]


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    lower = np.maximum(a[:3], b[:3])
    upper = np.minimum(a[3:], b[3:])
    intersection = float(np.maximum(upper - lower, 0.0).prod())
    volume_a = float(np.maximum(a[3:] - a[:3], 0.0).prod())
    volume_b = float(np.maximum(b[3:] - b[:3], 0.0).prod())
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def build_case_record(
    subject_id: str,
    prediction: Mapping[str, Any],
    target_semantic: np.ndarray,
    *,
    spacing_zyx: Sequence[float],
    checkpoint_sha256: str,
    evaluator_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    target_instances = label_instances_26(np.asarray(target_semantic) > 0)
    metrics = evaluate_fixed_case(
        subject_id,
        prediction,
        target_semantic,
        spacing_zyx=spacing_zyx,
        small_diameter_mm=3.0,
    )
    objects = list(prediction.get("objects", ()))
    predicted = _instance_masks(np.asarray(prediction["instance_ids"]))
    if len(objects) != len(predicted):
        raise ValueError("decoded objects and instance masks differ")
    detections = [
        {
            "box_zyx": _box(mask),
            "score": float(objects[index]["birth"]) - float(objects[index]["death"]),
        }
        for index, (_, mask) in enumerate(predicted)
    ]
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    targets = [
        {
            "box_zyx": _box(mask),
            "equivalent_diameter_mm": float(
                (6.0 * float(mask.sum()) * voxel_volume / math.pi) ** (1.0 / 3.0)
            ),
        }
        for _, mask in _instance_masks(target_instances)
    ]
    return {
        "schema_version": 1,
        "subject_id": str(subject_id),
        "checkpoint_sha256": str(checkpoint_sha256),
        "evaluator_sha256": str(evaluator_sha256),
        "elapsed_seconds": float(elapsed_seconds),
        "metrics": _native(metrics),
        "detections": detections,
        "targets": targets,
    }


def validate_case_record(
    record: Mapping[str, Any],
    subject_id: str,
    checkpoint_sha256: str,
    evaluator_sha256: str,
) -> None:
    if int(record.get("schema_version", -1)) != 1:
        raise ValueError("case record schema mismatch")
    if str(record.get("subject_id")) != str(subject_id):
        raise ValueError("case record subject mismatch")
    if str(record.get("checkpoint_sha256")) != str(checkpoint_sha256):
        raise ValueError("case record checkpoint mismatch")
    if str(record.get("evaluator_sha256")) != str(evaluator_sha256):
        raise ValueError("case record evaluator mismatch")
    if not isinstance(record.get("metrics"), Mapping):
        raise ValueError("case record metrics are missing")
    if not isinstance(record.get("detections"), list) or not isinstance(record.get("targets"), list):
        raise ValueError("case record detection payload is missing")
    elapsed = float(record.get("elapsed_seconds", -1))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("case record elapsed time is invalid")


def _ranked_matches(
    records: Sequence[Mapping[str, Any]], threshold: float, *, small_only: bool = False
) -> tuple[list[float], list[float], int]:
    targets: dict[str, list[Mapping[str, Any]]] = {}
    total_targets = 0
    detections: list[tuple[float, str, int, Mapping[str, Any]]] = []
    for record in records:
        subject = str(record["subject_id"])
        subject_targets = [
            target
            for target in record["targets"]
            if not small_only or float(target["equivalent_diameter_mm"]) <= 3.0
        ]
        targets[subject] = subject_targets
        total_targets += len(subject_targets)
        detections.extend(
            (float(detection["score"]), subject, index, detection)
            for index, detection in enumerate(record["detections"])
        )
    matched = {subject: set() for subject in targets}
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _, subject, _, detection in sorted(detections, key=lambda item: (-item[0], item[1], item[2])):
        overlaps = [
            _box_iou(detection["box_zyx"], target["box_zyx"])
            for target in targets[subject]
        ]
        best = int(np.argmax(overlaps)) if overlaps else -1
        accepted = best >= 0 and overlaps[best] >= float(threshold) and best not in matched[subject]
        if accepted:
            matched[subject].add(best)
        true_positive.append(float(accepted))
        false_positive.append(float(not accepted))
    return true_positive, false_positive, total_targets


def dataset_average_precision(records: Sequence[Mapping[str, Any]], iou_threshold: float) -> float:
    true_positive, false_positive, total_targets = _ranked_matches(records, iou_threshold)
    if total_targets == 0:
        return 1.0 if not true_positive else 0.0
    if not true_positive:
        return 0.0
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / float(total_targets)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _small_recall(records: Sequence[Mapping[str, Any]], threshold: float) -> tuple[int, float | None]:
    true_positive, _, total_targets = _ranked_matches(records, threshold, small_only=True)
    return total_targets, (float(sum(true_positive)) / total_targets if total_targets else None)


def summarize_full203(
    records: Sequence[Mapping[str, Any]],
    expected_subjects: Sequence[str],
    *,
    expected_count: int = 203,
) -> dict[str, Any]:
    if len(records) != int(expected_count):
        raise ValueError(f"exactly {expected_count} case records are required")
    subjects = [str(record.get("subject_id", "")) for record in records]
    if len(set(subjects)) != len(subjects):
        raise ValueError("duplicate subject IDs in case records")
    expected = tuple(sorted(str(subject) for subject in expected_subjects))
    if len(expected) != int(expected_count) or tuple(sorted(subjects)) != expected:
        raise ValueError(f"exactly {expected_count} aligned subjects are required")
    metric_names = sorted(set().union(*(record["metrics"].keys() for record in records)))
    macro: dict[str, dict[str, float | int | None]] = {}
    for metric in metric_names:
        values = []
        for record in records:
            value = record["metrics"].get(metric)
            if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
                values.append(float(value))
        macro[metric] = {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
        }
    small_count, small_recall = _small_recall(records, 0.5)
    return {
        "schema_version": 1,
        "case_count": len(records),
        "subject_ids": list(expected),
        "macro": macro,
        "dataset_ap50": dataset_average_precision(records, 0.50),
        "dataset_ap75": dataset_average_precision(records, 0.75),
        "target_small_count_3mm": int(small_count),
        "small_lesion_box_recall50": small_recall,
        "total_target_count": int(sum(len(record["targets"]) for record in records)),
        "total_predicted_count": int(sum(len(record["detections"]) for record in records)),
        "total_elapsed_seconds": float(sum(float(record["elapsed_seconds"]) for record in records)),
    }
