from __future__ import annotations

import numpy as np


def aggregate_arm_rows(rows: list[dict]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("at least one arm row is required")
    gt_count = int(sum(int(row["gt_count"]) for row in rows))
    micro_count = int(sum(int(row["micro_count"]) for row in rows))
    case_count = len(rows)
    if gt_count <= 0:
        raise ValueError("arm rows must contain at least one GT object")
    return {
        "case_count": case_count,
        "gt_count_total": gt_count,
        "micro_count_total": micro_count,
        "predicted_count_total": int(sum(int(row["predicted_count"]) for row in rows)),
        "count_mae": float(sum(float(row["absolute_count_error"]) for row in rows) / case_count),
        "proposal_recall_2mm": float(sum(int(row["proposal_hits_2mm"]) for row in rows) / gt_count),
        "proposal_recall_5mm": float(sum(int(row["proposal_hits_5mm"]) for row in rows) / gt_count),
        "micro_recall_2mm": (
            float(sum(int(row["micro_hits_2mm"]) for row in rows) / micro_count)
            if micro_count else None
        ),
        "micro_recall_5mm": (
            float(sum(int(row["micro_hits_5mm"]) for row in rows) / micro_count)
            if micro_count else None
        ),
        "centroid_inside": float(sum(int(row["centroid_inside_hits"]) for row in rows) / gt_count),
        "mean_nearest_center_mm": float(sum(float(row["nearest_center_sum_mm"]) for row in rows) / gt_count),
        "micro_mean_nearest_center_mm": (
            float(sum(float(row["micro_nearest_center_sum_mm"]) for row in rows) / micro_count)
            if micro_count else None
        ),
    }


def build_oracle_transport(
    instance_labels: np.ndarray,
    centers_zyx: tuple[tuple[int, int, int], ...],
    offset_scale_zyx: tuple[float, float, float],
) -> np.ndarray:
    labels = np.asarray(instance_labels)
    scale = np.asarray(offset_scale_zyx, dtype=np.float32)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("instance_labels must be a three-dimensional integer array")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("offset_scale_zyx must contain three positive finite values")
    result = np.zeros((3, *labels.shape), dtype=np.float32)
    for instance_id, center in enumerate(centers_zyx, start=1):
        coords = np.argwhere(labels == instance_id)
        if coords.size == 0:
            raise ValueError(f"center {instance_id} has no corresponding voxels")
        offsets = (np.asarray(center, dtype=np.float32)[None] - coords) / scale[None]
        result[:, labels == instance_id] = offsets.T
    return result


def make_support_transport_matrix(
    predicted_support: np.ndarray,
    predicted_transport: np.ndarray,
    instance_labels: np.ndarray,
    centers_zyx: tuple[tuple[int, int, int], ...],
    offset_scale_zyx: tuple[float, float, float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    support = np.asarray(predicted_support, dtype=np.float32)
    transport = np.asarray(predicted_transport, dtype=np.float32)
    labels = np.asarray(instance_labels)
    if support.shape != labels.shape:
        raise ValueError("predicted support and instance labels must have equal shape")
    if transport.shape != (3, *labels.shape):
        raise ValueError("predicted transport must have shape [3,D,H,W]")
    oracle_transport = build_oracle_transport(labels, centers_zyx, offset_scale_zyx)
    gt_support = (labels > 0).astype(np.float32)
    clipped_support = support * gt_support
    return {
        "gt_support_gt_transport": (gt_support, oracle_transport),
        "gt_support_pred_transport": (gt_support, transport),
        "pred_support_gt_transport": (clipped_support, oracle_transport),
        "pred_support_pred_transport": (support, transport),
    }


def endpoint_instance_diagnostics(
    instance_labels: np.ndarray,
    centers_zyx: tuple[tuple[int, int, int], ...],
    support: np.ndarray,
    transport: np.ndarray,
    offset_scale_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
) -> list[dict[str, float | int]]:
    labels = np.asarray(instance_labels)
    support = np.asarray(support, dtype=np.float32)
    transport = np.asarray(transport, dtype=np.float32)
    scale = np.asarray(offset_scale_zyx, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    rows: list[dict[str, float | int]] = []
    for instance_id, center in enumerate(centers_zyx, start=1):
        coords = np.argwhere(labels == instance_id)
        endpoints = coords.astype(np.float32) + transport[:, labels == instance_id].T * scale[None]
        center_array = np.asarray(center, dtype=np.float32)
        error_mm = np.linalg.norm((endpoints - center_array[None]) * spacing[None], axis=1)
        rounded = np.rint(endpoints).astype(np.int64)
        _, counts = np.unique(rounded, axis=0, return_counts=True)
        rows.append({
            "instance_id": instance_id,
            "voxel_count": int(len(coords)),
            "support_mean": float(support[labels == instance_id].mean()),
            "support_ge_0_1_rate": float((support[labels == instance_id] >= 0.1).mean()),
            "endpoint_error_mean_mm": float(error_mm.mean()),
            "endpoint_error_median_mm": float(np.median(error_mm)),
            "endpoint_within_0_5_vox_rate": float((np.linalg.norm(endpoints - center_array[None], axis=1) <= 0.5).mean()),
            "rounded_endpoint_mode_fraction": float(counts.max() / counts.sum()),
        })
    return rows
