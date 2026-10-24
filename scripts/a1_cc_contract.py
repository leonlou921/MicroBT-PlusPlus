from __future__ import annotations

import numpy as np
from scipy.ndimage import label


CC_STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)
REGION_THRESHOLD = 0.5
SCORE_TOP_FRACTION = 0.10


def connected_components_with_scores(
    wt_probability: np.ndarray,
    *,
    threshold: float = REGION_THRESHOLD,
    top_fraction: float = SCORE_TOP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(wt_probability, dtype=np.float32)
    if probability.ndim != 3 or not np.isfinite(probability).all():
        raise ValueError("WT probability must be a finite 3D array")
    if not 0.0 < threshold < 1.0 or not 0.0 < top_fraction <= 1.0:
        raise ValueError("invalid frozen A1 component contract")
    instances, count = label(probability > threshold, structure=CC_STRUCTURE_26)
    scores = []
    for instance_id in range(1, int(count) + 1):
        values = np.sort(probability[instances == instance_id].reshape(-1))
        top_k = max(1, int(np.ceil(values.size * top_fraction)))
        scores.append(float(values[-top_k:].mean()))
    return instances.astype(np.int32, copy=False), np.asarray(scores, dtype=np.float32)
