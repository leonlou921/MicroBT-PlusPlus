from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ADMISSION_METRICS = (
    "box_ap50",
    "box_ap75",
    "small_lesion_recall_3mm",
    "small_proposal_recall_3mm",
)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compare_admission_outputs(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if float(tolerance) < 0:
        raise ValueError("admission tolerance must be nonnegative")
    segmentation_equal = np.array_equal(
        np.asarray(reference["segmentation"]), np.asarray(candidate["segmentation"])
    )
    foreground_equal = np.array_equal(
        np.asarray(reference["instance_ids"]) > 0,
        np.asarray(candidate["instance_ids"]) > 0,
    )
    count_equal = int(reference["count"]) == int(candidate["count"])
    metric_deltas: dict[str, float | None] = {}
    metric_equal: dict[str, bool] = {}
    for metric in ADMISSION_METRICS:
        before = _finite_or_none(reference["metrics"].get(metric))
        after = _finite_or_none(candidate["metrics"].get(metric))
        if before is None or after is None:
            metric_deltas[metric] = None
            metric_equal[metric] = before is None and after is None
        else:
            delta = abs(after - before)
            metric_deltas[metric] = float(delta)
            metric_equal[metric] = delta < float(tolerance) or math.isclose(
                delta, float(tolerance), rel_tol=0.0, abs_tol=1e-12
            )
    admitted = bool(
        segmentation_equal
        and foreground_equal
        and count_equal
        and all(metric_equal.values())
    )
    return {
        "admitted": admitted,
        "segmentation_equal": bool(segmentation_equal),
        "instance_foreground_equal": bool(foreground_equal),
        "count_equal": bool(count_equal),
        "metric_equal": metric_equal,
        "metric_absolute_deltas": metric_deltas,
        "tolerance": float(tolerance),
    }


def select_fastest_admitted(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    admitted = [
        dict(result)
        for result in results
        if bool(result.get("admitted"))
        and math.isfinite(float(result.get("elapsed_seconds", float("nan"))))
    ]
    if not admitted:
        raise RuntimeError("no numerically admitted inference batch is available")
    return min(
        admitted,
        key=lambda result: (
            float(result["elapsed_seconds"]),
            int(result["batch_size"]),
        ),
    )
