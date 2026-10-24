from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_erosion, distance_transform_edt

import scripts.evaluate_microbt_pf_causal_matrix as causal_module
from scripts.evaluate_microbt_pf_causal_matrix import (
    REQUIRED_METRICS,
    align_variant_rows,
    build_causal_manifest,
    evaluate_case,
    paired_bootstrap_delta,
    resolve_fixed_checkpoints,
    summarize_metric,
)


def _legacy_hd95(prediction: np.ndarray, target: np.ndarray, spacing) -> float:
    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate(
        (distance_to_target[pred_surface], distance_to_prediction[target_surface])
    )
    return float(np.percentile(distances, 95.0))


def _rows(subjects: list[str], offset: float = 0.0) -> list[dict]:
    return [
        {
            "subject_id": subject,
            **{metric: float(index) + offset for metric in REQUIRED_METRICS},
        }
        for index, subject in enumerate(subjects)
    ]


def test_subject_alignment_is_sorted_and_independent_of_input_order() -> None:
    variants = {
        "A0": _rows(["case_c", "case_a", "case_b"]),
        "A0-D": _rows(["case_b", "case_c", "case_a"]),
        "A1": _rows(["case_a", "case_b", "case_c"]),
        "A2": _rows(["case_c", "case_b", "case_a"]),
        "A3": _rows(["case_b", "case_a", "case_c"]),
    }
    aligned = align_variant_rows(variants, expected_count=3)
    assert list(aligned) == ["A0", "A0-D", "A1", "A2", "A3"]
    assert [row["subject_id"] for row in aligned["A2"]] == [
        "case_a",
        "case_b",
        "case_c",
    ]


def test_alignment_rejects_an_incomplete_203_case_variant() -> None:
    subjects = [f"case_{index:03d}" for index in range(203)]
    variants = {name: _rows(subjects) for name in ("A0", "A0-D", "A1", "A2", "A3")}
    variants["A2"] = variants["A2"][:-1]
    with pytest.raises(ValueError, match="A2.*203"):
        align_variant_rows(variants, expected_count=203)


def test_all_nan_rows_are_excluded_from_metric_summary() -> None:
    rows = [
        {"subject_id": "a", "wt_dice": np.nan},
        {"subject_id": "b", "wt_dice": None},
        {"subject_id": "c", "wt_dice": 0.75},
    ]
    assert summarize_metric(rows, "wt_dice") == {
        "n": 1,
        "mean": 0.75,
        "median": 0.75,
        "std": 0.0,
    }


def test_paired_bootstrap_uses_only_jointly_finite_subjects() -> None:
    reference = [
        {"subject_id": "a", "wt_dice": 0.2},
        {"subject_id": "b", "wt_dice": np.nan},
        {"subject_id": "c", "wt_dice": 0.6},
    ]
    candidate = [
        {"subject_id": "a", "wt_dice": 0.3},
        {"subject_id": "b", "wt_dice": 0.9},
        {"subject_id": "c", "wt_dice": 0.8},
    ]
    observed = paired_bootstrap_delta(
        reference, candidate, "wt_dice", seed=7, samples=1000
    )
    assert observed["n"] == 2
    assert observed["mean_delta"] == pytest.approx(0.15)
    assert observed["ci95_low"] <= observed["mean_delta"] <= observed["ci95_high"]


def test_fixed_checkpoint_selection_and_a0d_identity(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    checkpoints: dict[str, str] = {}
    for variant in ("A0", "A1", "A2", "A3"):
        fold = result_root / variant / "fold_0"
        fold.mkdir(parents=True)
        final = fold / "checkpoint_final.pth"
        final.write_bytes(b"a0" if variant == "A0" else variant.encode())
        (fold / "checkpoint_best.pth").write_bytes(b"forbidden")
        checkpoints[variant] = str(final)
    checkpoints["A0-D"] = checkpoints["A0"]

    resolved = resolve_fixed_checkpoints(checkpoints)
    assert resolved["A0"]["name"] == "checkpoint_final.pth"
    assert resolved["A0-D"]["sha256"] == resolved["A0"]["sha256"]
    assert resolved["A0-D"]["path"] == resolved["A0"]["path"]

    wrong = dict(checkpoints)
    wrong["A0-D"] = checkpoints["A1"]
    with pytest.raises(ValueError, match="A0-D.*A0"):
        resolve_fixed_checkpoints(wrong)


def test_manifest_keeps_raw_rows_before_summaries(tmp_path: Path) -> None:
    subjects = ["case_a", "case_b"]
    variants = {name: _rows(subjects, index * 0.1) for index, name in enumerate(("A0", "A0-D", "A1", "A2", "A3"))}
    checkpoints = {}
    for variant in ("A0", "A1", "A2", "A3"):
        path = tmp_path / variant / "fold_0" / "checkpoint_final.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"a0" if variant == "A0" else variant.encode())
        checkpoints[variant] = str(path)
    checkpoints["A0-D"] = checkpoints["A0"]

    manifest = build_causal_manifest(
        variants,
        checkpoints,
        expected_count=2,
        bootstrap_seed=42,
        bootstrap_samples=100,
    )

    assert list(manifest)[:2] == ["schema_version", "raw_per_case"]
    assert len(manifest["raw_per_case"]["A3"]) == 2
    assert manifest["comparisons"]["A3_minus_A2"]["wt_dice"]["n"] == 2
    serialized = json.dumps(manifest, allow_nan=False)
    assert hashlib.sha256(serialized.encode()).hexdigest()


def test_registered_case_metrics_cover_semantic_instance_and_box_contract() -> None:
    semantic = np.zeros((5, 5, 5), dtype=np.uint8)
    semantic[2, 2, 2] = 3
    instances = np.zeros_like(semantic, dtype=np.int32)
    instances[2, 2, 2] = 1

    row = evaluate_case(
        "case",
        semantic,
        instances,
        semantic,
        instances,
        spacing_zyx=(1.0, 1.0, 1.0),
        object_scores=(0.9,),
    )

    assert set(row) == {"subject_id", *REQUIRED_METRICS}
    assert row["subject_id"] == "case"
    for metric in REQUIRED_METRICS:
        assert row[metric] == pytest.approx(1.0 if metric not in {
            "hd95_mm", "count_mae", "split_count", "merge_count", "false_positives_per_case"
        } else 0.0)


def test_hd95_crops_sparse_union_and_matches_legacy_at_image_boundary(monkeypatch) -> None:
    prediction = np.zeros((31, 33, 35), dtype=bool)
    target = np.zeros_like(prediction)
    prediction[0:4, 2:7, 3:8] = True
    target[1:5, 3:8, 4:9] = True
    spacing = (0.5, 0.75, 1.25)
    expected = _legacy_hd95(prediction, target, spacing)
    shapes: list[tuple[int, ...]] = []
    original = causal_module.binary_erosion

    def counted(array, *args, **kwargs):
        shapes.append(tuple(array.shape))
        return original(array, *args, **kwargs)

    monkeypatch.setattr(causal_module, "binary_erosion", counted)
    observed = causal_module._hd95(prediction, target, spacing)

    assert observed == pytest.approx(expected, abs=0.0, rel=0.0)
    assert shapes and all(np.prod(shape) < prediction.size for shape in shapes)


def test_registered_metrics_keep_legacy_values_without_full_volume_instance_masks(
    monkeypatch,
) -> None:
    target_instances = np.zeros((8, 8, 8), dtype=np.int32)
    target_instances[1:3, 1:3, 1:3] = 2
    target_instances[5:7, 5:7, 5:7] = 7
    predicted_instances = np.zeros_like(target_instances)
    predicted_instances[1:3, 1:3, 1:3] = 3
    predicted_instances[4:7, 5:7, 5:7] = 5
    predicted_instances[0, 7, 0] = 9
    target_semantic = np.zeros_like(target_instances, dtype=np.uint8)
    target_semantic[target_instances > 0] = 3
    predicted_semantic = np.zeros_like(target_semantic)
    predicted_semantic[predicted_instances > 0] = 3
    predicted_semantic[0, 7, 0] = 1

    def forbidden(_instance_ids):
        raise AssertionError("full-volume per-instance masks are forbidden")

    monkeypatch.setattr(causal_module, "_instance_masks", forbidden)
    observed = evaluate_case(
        "golden",
        predicted_semantic,
        predicted_instances,
        target_semantic,
        target_instances,
        spacing_zyx=(0.8, 1.2, 1.5),
        object_scores=(0.9, 0.8, 0.1),
    )

    expected = {
        "subject_id": "golden",
        "wt_dice": 0.8648648648648649,
        "tc_dice": 0.8888888888888888,
        "et_dice": 0.8888888888888888,
        "lesion_mask_dice": 0.8648648648648649,
        "hd95_mm": 0.8,
        "branch_precision": 2.0 / 3.0,
        "branch_recall": 1.0,
        "branch_f1": 0.8,
        "count_mae": 1.0,
        "exact_count": 0.0,
        "split_count": 0.0,
        "merge_count": 0.0,
        "false_positives_per_case": 1.0,
        "small_lesion_recall_3mm": 1.0,
        "small_lesion_f1_3mm": 2.0 / 3.0,
        "centroid_inside": 1.0,
        "box_ap50": 1.0,
        "box_ap75": 0.5,
        "box_tightness": 1.0,
        "volume_inflation": 1.25,
    }
    assert observed.keys() == expected.keys()
    for key, expected_value in expected.items():
        if key == "subject_id":
            assert observed[key] == expected_value
        else:
            assert observed[key] == pytest.approx(expected_value, abs=1e-12, rel=1e-12)


def test_high_instance_count_evaluation_does_not_materialize_instance_masks(
    monkeypatch,
) -> None:
    shape = (48, 48, 48)
    instances = np.zeros(shape, dtype=np.int32)
    coordinates = np.asarray(np.unravel_index(np.arange(192) * 503, shape)).T
    for identifier, coordinate in enumerate(coordinates, start=1):
        instances[tuple(coordinate)] = identifier
    semantic = (instances > 0).astype(np.uint8) * 3

    monkeypatch.setattr(
        causal_module,
        "_instance_masks",
        lambda _array: (_ for _ in ()).throw(
            AssertionError("full-volume per-instance masks are forbidden")
        ),
    )
    row = evaluate_case(
        "many",
        semantic,
        instances,
        semantic,
        instances,
        object_scores=np.linspace(1.0, 0.0, 192),
    )
    assert row["count_mae"] == 0.0
    assert row["branch_f1"] == pytest.approx(1.0)
    assert row["box_ap75"] == pytest.approx(1.0)
