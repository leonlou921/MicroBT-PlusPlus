from __future__ import annotations

import json

import pytest

from scripts.evaluate_microbt_umctf_full203 import (
    load_validation_subjects,
    summarize_case_records,
    validate_case_record,
)


CHECKPOINT = "checkpoint-sha"
EVALUATOR = "evaluator-sha"
ARMS = (
    "pred_mass_pred_transport",
    "gt_mass_pred_transport",
    "pred_mass_gt_transport",
    "gt_mass_gt_transport",
)


def _case(subject_id: str, *, hit: bool, predicted_count: int) -> dict:
    target_box = [0, 0, 0, 2, 2, 2]
    detection_box = target_box if hit else [5, 5, 5, 7, 7, 7]
    nearest = 0.0 if hit else 10.0
    return {
        "schema_version": 1,
        "subject_id": subject_id,
        "checkpoint_sha256": CHECKPOINT,
        "evaluator_sha256": EVALUATOR,
        "elapsed_seconds": 2.0,
        "dice": [1.0, 0.5, 0.25] if hit else [0.5, 0.5, 0.25],
        "gt_count": 1,
        "predicted_count": predicted_count,
        "input_mass": float(predicted_count),
        "atom_mass": float(predicted_count) - (0.0 if hit else 0.5),
        "detections": [{"box_zyx": detection_box, "score": 0.9 if hit else 0.8}],
        "targets": [{"box_zyx": target_box, "equivalent_diameter_mm": 2.0}],
        "objects": [{
            "nearest_distance_mm": nearest,
            "equivalent_diameter_mm": 2.0,
            "centroid_inside": 1.0 if hit else 0.0,
            "box_tightness": 1.0 if hit else 0.0,
            "duplicate_extras": 0 if hit else 1,
        }],
        "causal": {
            arm: {
                "predicted_count": 1,
                "objects": [{
                    "nearest_distance_mm": 0.0 if arm != "pred_mass_pred_transport" else nearest,
                    "equivalent_diameter_mm": 2.0,
                }],
            }
            for arm in ARMS
        },
    }


def test_load_validation_subjects_requires_exact_unique_fold(tmp_path):
    split = tmp_path / "splits_final.json"
    split.write_text(json.dumps([{"train": ["train"], "val": ["b", "a"]}]))
    assert load_validation_subjects(split, fold=0, expected_count=2) == ("a", "b")

    split.write_text(json.dumps([{"train": [], "val": ["a", "a"]}]))
    with pytest.raises(ValueError, match="unique"):
        load_validation_subjects(split, fold=0, expected_count=2)


def test_validate_case_record_rejects_checkpoint_drift():
    record = _case("a", hit=True, predicted_count=1)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_case_record(record, "a", "different", EVALUATOR)


def test_summarize_case_records_computes_global_metrics_and_causal_matrix():
    records = [
        _case("a", hit=True, predicted_count=1),
        _case("b", hit=False, predicted_count=2),
    ]
    summary = summarize_case_records(
        records,
        expected_subjects=("a", "b"),
        expected_count=2,
        checkpoint_sha256=CHECKPOINT,
        evaluator_sha256=EVALUATOR,
    )

    metrics = summary["metrics"]
    assert metrics["case_count"] == 2
    assert metrics["object_count"] == 2
    assert metrics["predicted_count_total"] == 3
    assert metrics["dice"] == pytest.approx([0.75, 0.5, 0.25])
    assert metrics["proposal_recall_5mm"] == pytest.approx(0.5)
    assert metrics["micro_recall_5mm_le3mm"] == pytest.approx(0.5)
    assert metrics["count_mae"] == pytest.approx(0.5)
    assert metrics["ap50"] == pytest.approx(0.5)
    assert metrics["ap75"] == pytest.approx(0.5)
    assert metrics["centroid_inside"] == pytest.approx(0.5)
    assert metrics["box_tightness_mean_best_iou"] == pytest.approx(0.5)
    assert metrics["duplicate_extras_per_gt_5mm"] == pytest.approx(0.5)
    assert metrics["mass_conservation_abs_error_mean"] == pytest.approx(0.25)

    predicted = summary["causal_matrix"]["pred_mass_pred_transport"]
    oracle_transport = summary["causal_matrix"]["pred_mass_gt_transport"]
    assert predicted["proposal_recall_5mm"] == pytest.approx(0.5)
    assert predicted["micro_recall_5mm_le3mm"] == pytest.approx(0.5)
    assert oracle_transport["proposal_recall_5mm"] == pytest.approx(1.0)
    assert oracle_transport["micro_recall_5mm_le3mm"] == pytest.approx(1.0)

