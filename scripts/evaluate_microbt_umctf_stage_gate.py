from __future__ import annotations

import re

import numpy as np


def count_real_log_errors(text: str) -> int:
    """Count failures without treating zero-valued telemetry as an error."""
    hard = len(re.findall(r"Traceback|CUDA out of memory|RuntimeError:|FloatingPointError", text, re.I))
    nonfinite = sum(
        int(value) for value in re.findall(r'"nonfinite_steps"\s*:\s*(\d+)', text)
    )
    return hard + nonfinite


def scored_instance_boxes(
    instance_labels: np.ndarray,
    center_scores: tuple[float, ...],
) -> list[tuple[tuple[int, int, int, int, int, int], float]]:
    """Preserve sparse atom IDs when pairing derived boxes with atom scores."""
    labels = np.asarray(instance_labels)
    result = []
    for instance_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        if instance_id > len(center_scores):
            raise ValueError("instance label has no corresponding atom score")
        coords = np.argwhere(labels == instance_id)
        box = tuple(int(v) for v in np.concatenate((coords.min(0), coords.max(0) + 1)))
        result.append((box, float(center_scores[instance_id - 1])))
    return result


if __name__ == "__main__":
    from scripts.evaluate_microbt_umctf_full_gate import main
    main()
