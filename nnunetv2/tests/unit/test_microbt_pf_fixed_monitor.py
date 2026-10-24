from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

import nnunetv2.evaluation.microbt_pf_fixed_monitor as fixed_monitor_module
from nnunetv2.evaluation.microbt_pf_fixed_monitor import (
    build_target_geometry,
    evaluate_fixed_case,
    label_instances_26,
    proposal_recall_at_radius,
    summarize_fixed_cases,
    validate_fixed_case_manifest,
)


def _legacy_target_centers(
    target_instances: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> tuple[list[int], np.ndarray]:
    identifiers: list[int] = []
    centers: list[np.ndarray] = []
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    for identifier in sorted(int(value) for value in np.unique(target_instances) if value > 0):
        mask = target_instances == identifier
        distance = distance_transform_edt(mask, sampling=spacing)
        maximum = float(distance.max())
        center = np.argwhere(mask & (distance == maximum))[0]
        identifiers.append(identifier)
        centers.append(center.astype(np.float64))
    return identifiers, np.stack(centers) if centers else np.empty((0, 3), dtype=np.float64)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "dataset_name": "Dataset001_Example",
        "fold": 0,
        "split_sha256": "a" * 64,
        "connectivity": 26,
        "small_diameter_mm": 3.0,
        "decoder": {"field_min": 0.25, "foreground_level": 0.0},
        "milestone_every_epochs": 25,
        "cases": [
            {
                "subject_id": "case-a",
                "stratum": "one/small",
                "expected_count": 1,
                "expected_small_count": 1,
            },
            {
                "subject_id": "case-b",
                "stratum": "many/no-small",
                "expected_count": 12,
                "expected_small_count": 0,
            },
        ],
    }


def test_manifest_preflight_binds_split_cases_and_metric_contract() -> None:
    observed = validate_fixed_case_manifest(
        _manifest(),
        split_sha256="a" * 64,
        validation_subjects={"case-a", "case-b", "case-c"},
    )
    assert observed["case_ids"] == ["case-a", "case-b"]
    assert observed["milestone_every_epochs"] == 25

    wrong = copy.deepcopy(_manifest())
    wrong["connectivity"] = 6
    with pytest.raises(ValueError, match="26-connectivity"):
        validate_fixed_case_manifest(
            wrong,
            split_sha256="a" * 64,
            validation_subjects={"case-a", "case-b"},
        )


def test_label_instances_uses_26_connectivity_for_diagonal_voxels() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True
    instances = label_instances_26(mask)
    assert int(instances.max()) == 1
    assert instances[1, 1, 1] == instances[2, 2, 2] == 1


def test_proposal_recall_is_one_to_one_and_spacing_aware() -> None:
    target = np.zeros((9, 9, 9), dtype=np.int32)
    target[1, 1, 1] = 1
    target[7, 7, 7] = 2
    proposals = ((1, 1, 1), (1, 1, 2))

    assert proposal_recall_at_radius(
        target, proposals, spacing_zyx=(1.0, 1.0, 2.0), radius_mm=3.0
    ) == pytest.approx(0.5)
    assert proposal_recall_at_radius(
        target, ((1, 1, 1), (7, 7, 7)), spacing_zyx=(1, 1, 1), radius_mm=3.0
    ) == pytest.approx(1.0)


def test_cropped_target_geometry_exactly_matches_legacy_full_volume_centers() -> None:
    target = np.zeros((15, 17, 19), dtype=np.int32)
    target[0:3, 0:4, 0:5] = 1
    target[6:12, 8:15, 10:18] = 3
    target[9, 11, 13] = 0
    spacing = (0.5, 0.75, 1.25)

    expected_ids, expected_centers = _legacy_target_centers(target, spacing)
    geometry = build_target_geometry(target, spacing)

    assert list(geometry.identifiers) == expected_ids
    np.testing.assert_array_equal(geometry.centers_zyx, expected_centers)


def test_fixed_case_computes_each_target_distance_transform_once(monkeypatch) -> None:
    semantic = np.zeros((11, 11, 11), dtype=np.uint8)
    semantic[1:4, 1:4, 1:4] = 3
    semantic[7:10, 7:10, 7:10] = 3
    instances = label_instances_26(semantic > 0)
    field = np.full(semantic.shape, -1.0, dtype=np.float32)
    field[2, 2, 2] = 0.8
    field[8, 8, 8] = 0.9
    prediction = {
        "segmentation": semantic.copy(),
        "instance_ids": instances.copy(),
        "field": field,
        "objects": (
            {"centroid_zyx": [2, 2, 2], "birth": 0.8, "death": -1.0},
            {"centroid_zyx": [8, 8, 8], "birth": 0.9, "death": -1.0},
        ),
        "count": 2,
    }
    calls: list[tuple[int, ...]] = []
    original = fixed_monitor_module.distance_transform_edt

    def counted(array, *args, **kwargs):
        calls.append(tuple(array.shape))
        return original(array, *args, **kwargs)

    monkeypatch.setattr(fixed_monitor_module, "distance_transform_edt", counted)
    evaluate_fixed_case("case", prediction, semantic, spacing_zyx=(1.0, 1.0, 1.0))

    assert len(calls) == 2
    assert all(shape != semantic.shape for shape in calls)


def test_fixed_case_row_covers_field_detection_count_and_small_lesion_learning() -> None:
    semantic = np.zeros((7, 7, 7), dtype=np.uint8)
    semantic[1, 1, 1] = 3
    semantic[5, 5, 5] = 3
    target_instances = label_instances_26(semantic > 0)
    field = np.full(semantic.shape, -1.0, dtype=np.float32)
    field[1, 1, 1] = 0.8
    field[5, 5, 5] = 0.9
    prediction = {
        "segmentation": semantic.copy(),
        "instance_ids": target_instances.copy(),
        "field": field,
        "objects": (
            {"centroid_zyx": [1, 1, 1], "birth": 0.8, "death": -1.0},
            {"centroid_zyx": [5, 5, 5], "birth": 0.9, "death": -1.0},
        ),
        "count": 2,
    }

    row = evaluate_fixed_case(
        "case",
        prediction,
        semantic,
        spacing_zyx=(1.0, 1.0, 1.0),
        expected_count=2,
        expected_small_count=2,
    )

    assert row["target_count"] == row["predicted_count"] == 2
    assert row["signed_count_error"] == 0
    assert row["branch_recall"] == pytest.approx(1.0)
    assert row["proposal_recall_3mm"] == pytest.approx(1.0)
    assert row["proposal_recall_5mm"] == pytest.approx(1.0)
    assert row["small_proposal_recall_3mm"] == pytest.approx(1.0)
    assert row["small_lesion_recall_3mm"] == pytest.approx(1.0)
    assert row["field_gt_center_mean"] == pytest.approx(0.85)
    assert row["field_background_p95"] == pytest.approx(-1.0)
    assert row["field_peak_center_distance_mm"] == pytest.approx(0.0)
    assert row["decoded_persistence_mean"] == pytest.approx(1.85)
    assert row["box_ap50"] == pytest.approx(1.0)
    assert row["box_ap75"] == pytest.approx(1.0)


def test_fixed_monitor_counts_ids_without_materializing_full_volume_masks(
    monkeypatch,
) -> None:
    semantic = np.zeros((7, 7, 7), dtype=np.uint8)
    semantic[1, 1, 1] = 3
    semantic[5, 5, 5] = 3
    instances = label_instances_26(semantic > 0)
    field = np.full(semantic.shape, -1.0, dtype=np.float32)
    prediction = {
        "segmentation": semantic,
        "instance_ids": instances,
        "field": field,
        "objects": (
            {"centroid_zyx": [1, 1, 1], "birth": 0.8, "death": -1.0},
            {"centroid_zyx": [5, 5, 5], "birth": 0.9, "death": -1.0},
        ),
        "count": 2,
    }
    monkeypatch.setattr(
        fixed_monitor_module,
        "_instance_masks",
        lambda _array: (_ for _ in ()).throw(
            AssertionError("fixed monitor must count instance IDs directly")
        ),
    )

    row = evaluate_fixed_case(
        "case", prediction, semantic, spacing_zyx=(1.0, 1.0, 1.0)
    )
    assert row["predicted_count"] == 2


def test_fixed_case_summary_reports_macro_and_count_totals() -> None:
    rows = [
        {
            "subject_id": "a",
            "target_count": 2,
            "predicted_count": 3,
            "signed_count_error": 1,
            "branch_recall": 0.5,
        },
        {
            "subject_id": "b",
            "target_count": 1,
            "predicted_count": 1,
            "signed_count_error": 0,
            "branch_recall": 1.0,
        },
    ]
    summary = summarize_fixed_cases(rows, epoch=25)
    assert summary["epoch"] == 25
    assert summary["case_count"] == 2
    assert summary["totals"] == {"target_count": 3, "predicted_count": 4}
    assert summary["macro"]["branch_recall"] == pytest.approx(0.75)
    assert summary["macro"]["signed_count_error"] == pytest.approx(0.5)
