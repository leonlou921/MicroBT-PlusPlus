from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


LOWER_IS_BETTER = {
    "wt_hd95_mm",
    "tc_hd95_mm",
    "et_hd95_mm",
    "hd95_mm",
    "count_mae",
    "split_count",
    "merge_count",
    "false_positives_per_case",
    "volume_inflation",
}


def paired_bootstrap_ci(
    deltas: Sequence[float], *, samples: int = 10_000, seed: int = 42
) -> dict[str, float | int | None]:
    values = np.asarray(deltas, dtype=np.float64)
    values = values[np.isfinite(values)]
    if int(samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not values.size:
        return {"n": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None}
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, values.size, size=(int(samples), values.size))
    bootstrap = values[indices].mean(axis=1)
    low, high = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "n": int(values.size),
        "mean_delta": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def paired_comparison(
    best_records: Sequence[Mapping[str, Any]],
    final_records: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            subject = str(row.get("subject_id", ""))
            if not subject or subject in result:
                raise ValueError("paired rows contain blank or duplicate subject IDs")
            result[subject] = row
        return result

    best = index(best_records)
    final = index(final_records)
    if set(best) != set(final):
        raise ValueError("paired rows must have identical subject IDs")
    subjects = sorted(best)
    metric_names = sorted(
        set.intersection(
            *(set(row.get("metrics", {})) for row in list(best.values()) + list(final.values()))
        )
    ) if subjects else []
    metrics: dict[str, Any] = {}
    for metric in metric_names:
        deltas: list[float] = []
        wins = ties = losses = 0
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        for subject in subjects:
            before = _finite(best[subject]["metrics"].get(metric))
            after = _finite(final[subject]["metrics"].get(metric))
            if before is None or after is None:
                continue
            delta = after - before
            deltas.append(delta)
            beneficial = -delta if direction == "lower" else delta
            if math.isclose(beneficial, 0.0, rel_tol=0.0, abs_tol=1e-12):
                ties += 1
            elif beneficial > 0:
                wins += 1
            else:
                losses += 1
        metrics[metric] = {
            **paired_bootstrap_ci(deltas, samples=samples, seed=seed),
            "direction": direction,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
    return {
        "schema_version": 1,
        "case_count": len(subjects),
        "subject_ids": subjects,
        "delta_definition": "checkpoint_final_minus_checkpoint_best",
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "metrics": metrics,
    }
