from __future__ import annotations

import numpy as np
import pytest

from nnunetv2.evaluation.microbt_pf_fast_admission import (
    compare_admission_outputs,
    select_fastest_admitted,
)


def _output(metric_shift: float = 0.0, count: int = 1) -> dict:
    return {
        "segmentation": np.array([[[0, 3]]], dtype=np.uint8),
        "instance_ids": np.array([[[0, 1]]], dtype=np.int32),
        "count": count,
        "metrics": {
            "box_ap50": 0.5 + metric_shift,
            "box_ap75": 0.25 + metric_shift,
            "small_lesion_recall_3mm": 0.75 + metric_shift,
            "small_proposal_recall_3mm": None,
        },
    }


def test_admission_uses_exact_arrays_and_inclusive_tolerance() -> None:
    accepted = compare_admission_outputs(_output(), _output(1e-6), tolerance=1e-6)
    assert accepted["admitted"] is True
    rejected = compare_admission_outputs(_output(), _output(1.0001e-6), tolerance=1e-6)
    assert rejected["admitted"] is False
    changed = _output()
    changed["segmentation"] = changed["segmentation"].copy()
    changed["segmentation"][0, 0, 1] = 2
    assert compare_admission_outputs(_output(), changed)["admitted"] is False


def test_common_null_is_allowed_but_one_sided_null_is_rejected() -> None:
    assert compare_admission_outputs(_output(), _output())["admitted"] is True
    candidate = _output()
    candidate["metrics"]["small_proposal_recall_3mm"] = 0.0
    assert compare_admission_outputs(_output(), candidate)["admitted"] is False


def test_selector_chooses_fastest_passing_candidate() -> None:
    selected = select_fastest_admitted(
        [
            {"batch_size": 1, "admitted": True, "elapsed_seconds": 10.0},
            {"batch_size": 4, "admitted": True, "elapsed_seconds": 4.0},
            {"batch_size": 8, "admitted": False, "elapsed_seconds": 3.0, "oom": True},
            {"batch_size": 6, "admitted": True, "elapsed_seconds": 5.0},
        ]
    )
    assert selected["batch_size"] == 4


def test_selector_fails_when_even_batch_one_is_not_admitted() -> None:
    with pytest.raises(RuntimeError, match="no numerically admitted"):
        select_fastest_admitted([{"batch_size": 1, "admitted": False, "elapsed_seconds": 1.0}])
