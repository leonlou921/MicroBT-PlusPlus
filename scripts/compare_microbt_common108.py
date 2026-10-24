#!/usr/bin/env python3
"""Compare persisted MicroBT results on one frozen 108-subject cohort.

This is deliberately an offline records-only analyzer. It reads legacy
per-case CSV tables and PF-family per-case JSON records; it does not import the
training stack, load checkpoints, or launch inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 10_000
TIE_TOLERANCE = 1e-8

LEGACY_METRICS = (
    "dice_wt",
    "dice_tc",
    "dice_et",
    "dice_mean",
    "hd95_wt",
    "hd95_tc",
    "hd95_et",
    "ap50",
    "ap75",
    "precision_50",
    "recall_50",
    "precision_75",
    "recall_75",
    "mean_best_gt_iou",
    "mean_best_pred_iou",
    "det_centroid_inside_rate",
    "count_mae",
)

PF_METRICS = (
    "wt_dice",
    "et_dice",
    "lesion_mask_dice",
    "hd95_mm",
    "branch_precision",
    "branch_recall",
    "branch_f1",
    "count_mae",
    "exact_count",
    "false_positives_per_case",
    "split_count",
    "merge_count",
    "box_ap50",
    "box_ap75",
    "small_lesion_recall_3mm",
    "small_lesion_f1_3mm",
    "centroid_inside",
    "box_tightness",
    "volume_inflation",
    "volume_inflation_abs_error",
)

EXCLUDED_PF_METRICS = {
    "tc_dice": (
        "invalid persisted definition: evaluator used label >= 2 while the dataset "
        "defines tumor core as labels [1, 3]; semantic predictions were not retained"
    )
}

LOWER_IS_BETTER = {
    "hd95_wt",
    "hd95_tc",
    "hd95_et",
    "hd95_mm",
    "count_mae",
    "false_positives_per_case",
    "split_count",
    "merge_count",
    "volume_inflation_abs_error",
}


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


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_fingerprint(directory: Path) -> tuple[str, int]:
    files = sorted((directory / "cases").glob("*.json"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def subject_list_sha256(subjects: Sequence[str]) -> str:
    payload = "".join(f"{subject}\n" for subject in subjects).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_cohort(case_directory: Path, *, expected_count: int = 108) -> tuple[list[str], str]:
    paths = sorted(Path(case_directory).glob("*.json"))
    if len(paths) != expected_count:
        raise ValueError(
            f"cohort directory has {len(paths)} JSON records; exactly {expected_count} required"
        )
    subjects: list[str] = []
    seen: set[str] = set()
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        subject = str(record.get("subject_id", "")).strip()
        if not subject:
            raise ValueError(f"blank subject_id in {path}")
        if subject in seen:
            raise ValueError(f"duplicate subject_id {subject}")
        seen.add(subject)
        subjects.append(subject)
    subjects.sort()
    return subjects, subject_list_sha256(subjects)


def align_rows(
    rows: Sequence[Mapping[str, Any]], cohort: Sequence[str], version: str
) -> list[dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        subject = str(row.get("subject_id", row.get("case_id", ""))).strip()
        if not subject:
            raise ValueError(f"{version} contains a blank subject_id")
        if subject in indexed:
            raise ValueError(f"{version} contains duplicate subject_id {subject}")
        row["subject_id"] = subject
        indexed[subject] = row
    required = set(cohort)
    missing = sorted(required - set(indexed))
    if missing:
        raise ValueError(f"{version} missing {len(missing)} cohort subjects: {missing[:5]}")
    return [indexed[subject] for subject in cohort]


def summarize_metric(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, int | float | None]:
    values = [_finite(row.get(metric)) for row in rows]
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
    }


def paired_comparison(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    higher_is_better: bool,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
    tie_tolerance: float = TIE_TOLERANCE,
) -> dict[str, int | float | None]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    reference = {str(row["subject_id"]): row for row in reference_rows}
    candidate = {str(row["subject_id"]): row for row in candidate_rows}
    if set(reference) != set(candidate):
        raise ValueError("paired rows must have identical subject IDs")
    differences: list[float] = []
    wins = ties = losses = 0
    for subject in sorted(reference):
        before = _finite(reference[subject].get(metric))
        after = _finite(candidate[subject].get(metric))
        if before is None or after is None:
            continue
        raw_delta = after - before
        differences.append(raw_delta)
        utility_delta = raw_delta if higher_is_better else -raw_delta
        if abs(utility_delta) <= tie_tolerance:
            ties += 1
        elif utility_delta > 0:
            wins += 1
        else:
            losses += 1
    if not differences:
        return {
            "n": 0,
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    delta = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(samples, delta.size))
    bootstrap = delta[indices].mean(axis=1)
    low, high = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "n": int(delta.size),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
    }


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (6,) or second_array.shape != (6,):
        raise ValueError("box_zyx must contain six coordinates")
    lower = np.maximum(first_array[:3], second_array[:3])
    upper = np.minimum(first_array[3:], second_array[3:])
    intersection = float(np.maximum(upper - lower, 0.0).prod())
    first_volume = float(np.maximum(first_array[3:] - first_array[:3], 0.0).prod())
    second_volume = float(np.maximum(second_array[3:] - second_array[:3], 0.0).prod())
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0 else 0.0


def dataset_average_precision(
    records: Sequence[Mapping[str, Any]], *, threshold: float
) -> float:
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    target_boxes: dict[str, list[Sequence[float]]] = {}
    detections: list[tuple[float, str, int, Sequence[float]]] = []
    target_count = 0
    for record in records:
        subject = str(record["subject_id"])
        targets = [target["box_zyx"] for target in record.get("targets", [])]
        target_boxes[subject] = targets
        target_count += len(targets)
        for index, detection in enumerate(record.get("detections", [])):
            detections.append(
                (float(detection["score"]), subject, index, detection["box_zyx"])
            )
    detections.sort(key=lambda item: (-item[0], item[1], item[2]))
    if target_count == 0:
        return 1.0 if not detections else 0.0
    matched: dict[str, set[int]] = {subject: set() for subject in target_boxes}
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _, subject, _, box in detections:
        overlaps = [_box_iou(box, target) for target in target_boxes[subject]]
        best = int(np.argmax(overlaps)) if overlaps else -1
        is_true = (
            best >= 0
            and overlaps[best] >= threshold
            and best not in matched[subject]
        )
        if is_true:
            matched[subject].add(best)
        true_positive.append(float(is_true))
        false_positive.append(float(not is_true))
    if not true_positive:
        return 0.0
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / target_count
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def pf_instance_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose cardinality and singleton-box failure modes from saved records."""
    target_counts: list[float] = []
    predicted_counts: list[float] = []
    small_target_count = 0
    small_matched_count = 0.0
    box_volumes: list[float] = []
    scores: list[float] = []
    singleton_scores: list[float] = []
    for record in records:
        metrics = record.get("metrics", {})
        target_count = _finite(metrics.get("target_count"))
        predicted_count = _finite(metrics.get("predicted_count"))
        if target_count is not None:
            target_counts.append(target_count)
        if predicted_count is not None:
            predicted_counts.append(predicted_count)
        case_small_count = int(_finite(metrics.get("target_small_count_3mm")) or 0)
        case_small_recall = _finite(metrics.get("small_lesion_recall_3mm"))
        small_target_count += case_small_count
        if case_small_count and case_small_recall is not None:
            small_matched_count += case_small_count * case_small_recall
        for detection in record.get("detections", []):
            box = np.asarray(detection["box_zyx"], dtype=np.float64)
            if box.shape != (6,):
                raise ValueError("box_zyx must contain six coordinates")
            volume = float(np.maximum(box[3:] - box[:3], 0.0).prod())
            score = float(detection["score"])
            box_volumes.append(volume)
            scores.append(score)
            if volume == 1.0:
                singleton_scores.append(score)
    volumes = np.asarray(box_volumes, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    singleton_count = int(np.count_nonzero(volumes == 1.0))
    return {
        "case_count": len(records),
        "target_count_mean": float(np.mean(target_counts)) if target_counts else None,
        "predicted_count_mean": (
            float(np.mean(predicted_counts)) if predicted_counts else None
        ),
        "detection_count": int(volumes.size),
        "singleton_box_count": singleton_count,
        "singleton_box_fraction": (
            float(singleton_count / volumes.size) if volumes.size else None
        ),
        "box_volume_median_voxels": (
            float(np.median(volumes)) if volumes.size else None
        ),
        "box_volume_p90_voxels": (
            float(np.percentile(volumes, 90.0)) if volumes.size else None
        ),
        "score_median": float(np.median(score_array)) if score_array.size else None,
        "score_p90": (
            float(np.percentile(score_array, 90.0)) if score_array.size else None
        ),
        "singleton_score_mean": (
            float(np.mean(singleton_scores)) if singleton_scores else None
        ),
        "small_target_count": int(small_target_count),
        "small_matched_count": float(small_matched_count),
        "small_micro_recall_3mm": (
            float(small_matched_count / small_target_count)
            if small_target_count
            else None
        ),
    }


def _load_legacy(path: Path, cohort: Sequence[str], version: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        source_rows = list(csv.DictReader(source))
    rows = align_rows(source_rows, cohort, version)
    for row in rows:
        row["count_mae"] = abs(float(row["num_pred"]) - float(row["num_gt"]))
        for metric in LEGACY_METRICS:
            if metric not in row:
                raise ValueError(f"{version} lacks required legacy metric {metric}")
            row[metric] = _finite(row[metric])
    return rows


def _load_pf(
    directory: Path, cohort: Sequence[str], version: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    records: list[dict[str, Any]] = []
    for path in sorted((directory / "cases").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        subject = str(record.get("subject_id", "")).strip()
        metrics = dict(record.get("metrics", {}))
        metrics["subject_id"] = subject
        inflation = _finite(metrics.get("volume_inflation"))
        metrics["volume_inflation_abs_error"] = (
            None if inflation is None else abs(inflation - 1.0)
        )
        records.append(record)
    rows = align_rows(
        [dict(record.get("metrics", {}), subject_id=record["subject_id"]) for record in records],
        cohort,
        version,
    )
    for row in rows:
        inflation = _finite(row.get("volume_inflation"))
        row["volume_inflation_abs_error"] = None if inflation is None else abs(inflation - 1.0)
        for metric in PF_METRICS:
            if metric not in row:
                raise ValueError(f"{version} lacks required PF metric {metric}")
            row[metric] = _finite(row[metric])
        row.pop("tc_dice", None)
    record_rows = align_rows(records, cohort, version)
    evaluator_hashes = {
        str(record.get("evaluator_sha256", "")) for record in record_rows
        if record.get("evaluator_sha256")
    }
    checkpoint_hashes = {
        str(record.get("checkpoint_sha256", "")) for record in record_rows
        if record.get("checkpoint_sha256")
    }
    return rows, record_rows, evaluator_hashes, checkpoint_hashes


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_native(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(
            sink,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _native(row.get(field)) for field in fields})
    temporary.replace(path)


def _summary_rows(
    panel: str,
    versions: Sequence[str],
    rows_by_version: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in metrics:
        metric_rows: list[dict[str, Any]] = []
        for version in versions:
            row = {
                "panel": panel,
                "version": version,
                "metric": metric,
                "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                **summarize_metric(rows_by_version[version], metric),
            }
            metric_rows.append(row)
        eligible = [row for row in metric_rows if row["mean"] is not None]
        eligible.sort(
            key=lambda row: float(row["mean"]),
            reverse=metric not in LOWER_IS_BETTER,
        )
        for rank, row in enumerate(eligible, start=1):
            row["rank"] = rank
        for row in metric_rows:
            row.setdefault("rank", None)
        output.extend(metric_rows)
    return output


def _paired_rows(
    panel: str,
    pairs: Sequence[tuple[str, str, str]],
    rows_by_version: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for comparison, reference, candidate in pairs:
        for metric in metrics:
            result = paired_comparison(
                rows_by_version[reference],
                rows_by_version[candidate],
                metric,
                higher_is_better=metric not in LOWER_IS_BETTER,
            )
            output.append(
                {
                    "panel": panel,
                    "comparison": comparison,
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                    **result,
                }
            )
    return output


def _markdown_table(
    summary: Sequence[Mapping[str, Any]], versions: Sequence[str], metrics: Sequence[str]
) -> list[str]:
    indexed = {(row["version"], row["metric"]): row for row in summary}
    lines = [
        "| Version | " + " | ".join(metrics) + " |",
        "|---|" + "|".join("---:" for _ in metrics) + "|",
    ]
    for version in versions:
        values = []
        for metric in metrics:
            row = indexed[(version, metric)]
            mean = row["mean"]
            values.append("NA" if mean is None else f"{float(mean):.4f} (n={row['n']})")
        lines.append(f"| {version} | " + " | ".join(values) + " |")
    return lines


def run_analysis(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cohort_case_dir = Path(config["cohort_case_dir"])
    expected_count = int(config.get("expected_count", 108))
    cohort, cohort_hash = freeze_cohort(cohort_case_dir, expected_count=expected_count)
    _write_text(output / "cohort_subjects.txt", "".join(f"{subject}\n" for subject in cohort))

    availability: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    legacy_rows: dict[str, list[dict[str, Any]]] = {}
    legacy_versions: list[str] = []
    for entry in config["legacy_versions"]:
        name = str(entry["name"])
        path = Path(entry["path"])
        rows = _load_legacy(path, cohort, name)
        legacy_rows[name] = rows
        legacy_versions.append(name)
        availability.append(
            {
                "panel": "legacy",
                "version": name,
                "status": "included",
                "reason": "full-volume per-case table covers all frozen subjects",
                "source_path": str(path),
                "rows_on_cohort": len(rows),
                "evaluator_family": str(entry.get("evaluator_family", "legacy")),
            }
        )
        source_manifest.append(
            {
                "panel": "legacy",
                "version": name,
                "path": str(path),
                "sha256": _sha256_file(path),
                "source_row_count": sum(1 for _ in path.open(encoding="utf-8-sig")) - 1,
                "evaluator_family": str(entry.get("evaluator_family", "legacy")),
            }
        )

    pf_rows: dict[str, list[dict[str, Any]]] = {}
    pf_records: dict[str, list[dict[str, Any]]] = {}
    pf_versions: list[str] = []
    for entry in config["pf_versions"]:
        name = str(entry["name"])
        directory = Path(entry["path"])
        rows, records, evaluator_hashes, checkpoint_hashes = _load_pf(
            directory, cohort, name
        )
        pf_rows[name] = rows
        pf_records[name] = records
        pf_versions.append(name)
        fingerprint, file_count = _directory_fingerprint(directory)
        availability.append(
            {
                "panel": "pf",
                "version": name,
                "status": "included",
                "reason": "persisted case JSON records cover all frozen subjects",
                "source_path": str(directory),
                "rows_on_cohort": len(rows),
                "evaluator_family": str(entry.get("evaluator_family", "pf")),
            }
        )
        source_manifest.append(
            {
                "panel": "pf",
                "version": name,
                "path": str(directory),
                "cases_directory_sha256": fingerprint,
                "source_case_count": file_count,
                "cohort_row_count": len(rows),
                "evaluator_family": str(entry.get("evaluator_family", "pf")),
                "evaluator_sha256_values": sorted(evaluator_hashes),
                "checkpoint_sha256_values": sorted(checkpoint_hashes),
            }
        )

    for entry in config.get("unavailable_versions", []):
        availability.append(
            {
                "panel": str(entry["panel"]),
                "version": str(entry["name"]),
                "status": "not_ranked",
                "reason": str(entry["reason"]),
                "source_path": str(entry.get("path", "")),
                "rows_on_cohort": int(entry.get("rows_on_cohort", 0)),
                "evaluator_family": str(entry.get("evaluator_family", "")),
            }
        )

    legacy_summary = _summary_rows(
        "legacy", legacy_versions, legacy_rows, LEGACY_METRICS
    )
    pf_summary = _summary_rows("pf", pf_versions, pf_rows, PF_METRICS)
    legacy_pairs: list[tuple[str, str, str]] = []
    for index, candidate in enumerate(legacy_versions[1:], start=1):
        legacy_pairs.append((f"{candidate}_minus_{legacy_versions[0]}", legacy_versions[0], candidate))
        predecessor = legacy_versions[index - 1]
        if predecessor != legacy_versions[0]:
            legacy_pairs.append((f"{candidate}_minus_{predecessor}", predecessor, candidate))
    pf_pairs = [
        (f"{candidate}_minus_{reference}", reference, candidate)
        for reference_index, reference in enumerate(pf_versions)
        for candidate in pf_versions[reference_index + 1 :]
    ]
    legacy_paired = _paired_rows(
        "legacy", legacy_pairs, legacy_rows, LEGACY_METRICS
    )
    pf_paired = _paired_rows("pf", pf_pairs, pf_rows, PF_METRICS)

    dataset_ap = {
        version: {
            "dataset_ap50": dataset_average_precision(records, threshold=0.50),
            "dataset_ap75": dataset_average_precision(records, threshold=0.75),
            "subjects": len(records),
            "targets": sum(len(record.get("targets", [])) for record in records),
            "detections": sum(len(record.get("detections", [])) for record in records),
        }
        for version, records in pf_records.items()
    }
    instance_diagnostics = {
        version: pf_instance_diagnostics(records)
        for version, records in pf_records.items()
    }

    legacy_aligned = [
        {"panel": "legacy", "version": version, **row}
        for version in legacy_versions
        for row in legacy_rows[version]
    ]
    pf_aligned = [
        {"panel": "pf", "version": version, **row}
        for version in pf_versions
        for row in pf_rows[version]
    ]
    _write_csv(
        output / "legacy_aligned_108.csv",
        legacy_aligned,
        ("panel", "version", "subject_id", *LEGACY_METRICS),
    )
    _write_csv(
        output / "pf_aligned_108.csv",
        pf_aligned,
        ("panel", "version", "subject_id", *PF_METRICS),
    )
    summary_fields = (
        "panel", "version", "metric", "direction", "rank", "n", "mean", "median", "std"
    )
    paired_fields = (
        "panel", "comparison", "reference", "candidate", "metric", "direction",
        "n", "mean_delta", "ci95_low", "ci95_high", "wins", "ties", "losses",
    )
    _write_csv(output / "legacy_summary_108.csv", legacy_summary, summary_fields)
    _write_csv(output / "pf_summary_108.csv", pf_summary, summary_fields)
    _write_csv(output / "legacy_paired_deltas_108.csv", legacy_paired, paired_fields)
    _write_csv(output / "pf_paired_deltas_108.csv", pf_paired, paired_fields)
    _write_csv(
        output / "version_availability.csv",
        availability,
        ("panel", "version", "status", "reason", "source_path", "rows_on_cohort", "evaluator_family"),
    )
    _write_json(output / "pf_dataset_ap_108.json", dataset_ap)
    diagnostic_fields = (
        "version", "case_count", "target_count_mean", "predicted_count_mean",
        "detection_count", "singleton_box_count", "singleton_box_fraction",
        "box_volume_median_voxels", "box_volume_p90_voxels", "score_median",
        "score_p90", "singleton_score_mean", "small_target_count",
        "small_matched_count", "small_micro_recall_3mm",
    )
    _write_csv(
        output / "pf_instance_diagnostics_108.csv",
        [dict(version=version, **instance_diagnostics[version]) for version in pf_versions],
        diagnostic_fields,
    )

    manifest = {
        "analysis_kind": "common completed cohort; records-only; no inference",
        "cohort_case_dir": str(cohort_case_dir),
        "cohort_count": len(cohort),
        "cohort_subject_list_sha256": cohort_hash,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "tie_tolerance": TIE_TOLERANCE,
        "legacy_metrics": list(LEGACY_METRICS),
        "pf_metrics": list(PF_METRICS),
        "excluded_metrics": EXCLUDED_PF_METRICS,
        "source_manifest": source_manifest,
        "evaluator_equivalence_evidence": config.get("evaluator_equivalence_evidence", {}),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "limitations": [
            "The 108 subjects are a completion-defined subset of the interrupted SB-PF final run.",
            "Ranks are valid only within each evaluator panel.",
            "Legacy AP50/AP75 are macro per-case AP; PF dataset AP is reconstructed separately.",
        ],
    }
    _write_json(output / "cohort_manifest.json", manifest)

    report_lines = [
        "# MicroBT Common-108 Comparison",
        "",
        f"Cohort: **{len(cohort)} common completed subjects**; SHA256 `{cohort_hash}`.",
        "No inference or checkpoint loading was performed. Ranks are panel-local.",
        "",
        "## Legacy Clean/R panel",
        "",
        "Legacy AP values below are macro per-case AP, not dataset AP.",
        "",
        *_markdown_table(
            legacy_summary,
            legacy_versions,
            ("dice_mean", "dice_wt", "dice_tc", "dice_et", "count_mae", "ap50", "ap75"),
        ),
        "",
        "## PF/SB-PF panel",
        "",
        *_markdown_table(
            pf_summary,
            pf_versions,
            (
                "wt_dice", "et_dice", "lesion_mask_dice", "hd95_mm", "count_mae",
                "box_ap50", "box_ap75", "small_lesion_recall_3mm",
            ),
        ),
        "",
        "### PF dataset AP",
        "",
        "| Version | AP50 | AP75 | Targets | Detections |",
        "|---|---:|---:|---:|---:|",
    ]
    for version in pf_versions:
        values = dataset_ap[version]
        report_lines.append(
            f"| {version} | {values['dataset_ap50']:.4f} | {values['dataset_ap75']:.4f} | "
            f"{values['targets']} | {values['detections']} |"
        )
    report_lines.extend(
        [
            "",
            "### PF cardinality and singleton diagnostics",
            "",
            "| Version | Pred/case | Singleton boxes | Singleton fraction | Small micro recall (215 lesions) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for version in pf_versions:
        values = instance_diagnostics[version]
        report_lines.append(
            f"| {version} | {values['predicted_count_mean']:.2f} | "
            f"{values['singleton_box_count']} | {values['singleton_box_fraction']:.4f} | "
            f"{values['small_micro_recall_3mm']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Exclusions and interpretation boundary",
            "",
            f"- PF `tc_dice` excluded: {EXCLUDED_PF_METRICS['tc_dice']}.",
            "- R1/R3.5/R3.7/R3.8 remain in the availability ledger but are not ranked without a compatible 108-subject full-volume table.",
            "- These results do not replace the frozen 203-case evaluation and do not establish cross-panel causal superiority.",
            "",
        ]
    )
    _write_text(output / "comparison_108.md", "\n".join(report_lines))
    comparison = {
        "manifest": manifest,
        "availability": availability,
        "legacy_summary": legacy_summary,
        "legacy_paired_deltas": legacy_paired,
        "pf_summary": pf_summary,
        "pf_paired_deltas": pf_paired,
        "pf_dataset_ap": dataset_ap,
        "pf_instance_diagnostics": instance_diagnostics,
    }
    _write_json(output / "comparison_108.json", comparison)
    return comparison


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_analysis(arguments.config)
    print(
        json.dumps(
            {
                "cohort_count": result["manifest"]["cohort_count"],
                "cohort_sha256": result["manifest"]["cohort_subject_list_sha256"],
                "legacy_versions": len({row["version"] for row in result["legacy_summary"]}),
                "pf_versions": len({row["version"] for row in result["pf_summary"]}),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
