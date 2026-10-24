from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nnunetv2.evaluation.microbt_pf_full203 import (
    atomic_write_json,
    dataset_average_precision,
    summarize_full203,
    validate_case_record,
)


def _record(subject: str, score: float, pred_box, gt_box, *, small: bool = True) -> dict:
    return {
        "schema_version": 1,
        "subject_id": subject,
        "checkpoint_sha256": "checkpoint",
        "evaluator_sha256": "evaluator",
        "elapsed_seconds": 1.0,
        "metrics": {"wt_dice": 1.0, "target_count": 1, "predicted_count": 1},
        "detections": [{"box_zyx": pred_box, "score": score}],
        "targets": [{"box_zyx": gt_box, "equivalent_diameter_mm": 2.0 if small else 4.0}],
    }


def test_dataset_ap_is_globally_score_sorted() -> None:
    good = _record("b", 0.8, [0, 0, 0, 2, 2, 2], [0, 0, 0, 2, 2, 2])
    false_first = _record("a", 0.9, [5, 5, 5, 7, 7, 7], [0, 0, 0, 2, 2, 2])
    assert dataset_average_precision([false_first, good], 0.5) == pytest.approx(0.25)


def test_resume_validation_is_bound_to_subject_and_hashes() -> None:
    record = _record("case", 1.0, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    validate_case_record(record, "case", "checkpoint", "evaluator")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_case_record(record, "case", "different", "evaluator")
    with pytest.raises(ValueError, match="subject"):
        validate_case_record(record, "other", "checkpoint", "evaluator")


def test_summary_requires_exact_aligned_panel() -> None:
    records = [
        _record("a", 1.0, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
        _record("b", 1.0, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
    ]
    summary = summarize_full203(records, ("a", "b"), expected_count=2)
    assert summary["case_count"] == 2
    assert summary["dataset_ap50"] == 1.0
    with pytest.raises(ValueError, match="exactly 3"):
        summarize_full203(records, ("a", "b", "c"), expected_count=3)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_full203([records[0], records[0]], ("a", "b"), expected_count=2)


def test_atomic_json_normalizes_nonfinite(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    atomic_write_json(destination, {"finite": 1.0, "missing": float("nan")})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "missing": None,
    }
    assert not (tmp_path / "record.json.tmp").exists()


def test_registered_small_lesion_contract_uses_equivalent_diameter() -> None:
    records = [
        _record("small", 1.0, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], small=True),
        _record("large", 1.0, [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], small=False),
    ]
    summary = summarize_full203(records, ("large", "small"), expected_count=2)
    assert summary["target_small_count_3mm"] == 1
    assert summary["small_lesion_box_recall50"] == 1.0
