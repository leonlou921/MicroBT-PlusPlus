from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.export_umctf_test251_native import (
    conserve_resampled_instance_ids,
    regions_to_semantic,
    undo_crop_and_transpose,
    validate_native_export_records,
)


def test_regions_to_semantic_uses_frozen_region_class_order() -> None:
    regions = np.zeros((3, 2, 2, 2), dtype=np.uint8)
    regions[0, 0, 0, 0] = 1
    regions[0, 0, 0, 1] = 1
    regions[1, 0, 0, 1] = 1
    regions[2, 0, 0, 1] = 1

    semantic = regions_to_semantic(regions, (1, 2, 3))

    expected = np.zeros((2, 2, 2), dtype=np.uint16)
    expected[0, 0, 0] = 1
    expected[0, 0, 1] = 3
    np.testing.assert_array_equal(semantic, expected)


def test_undo_crop_and_transpose_restores_native_shape_and_location() -> None:
    cropped = np.zeros((2, 3, 3), dtype=np.int32)
    cropped[1, 2, 0] = 7
    properties = {
        "shape_before_cropping": (4, 5, 6),
        "bbox_used_for_cropping": ((1, 3), (1, 4), (2, 5)),
    }

    restored = undo_crop_and_transpose(cropped, properties, (2, 0, 1))

    assert restored.shape == (6, 4, 5)
    assert restored[2, 2, 3] == 7
    assert int(np.count_nonzero(restored)) == 1


def test_conserve_resampled_instance_ids_restores_dropped_atom_without_dropping_another() -> None:
    source = np.zeros((1, 4, 4), dtype=np.int32)
    source[0, 1, 1] = 1
    source[0, 1, 2] = 2
    resampled = np.zeros((1, 3, 3), dtype=np.int32)
    resampled[0, 1, 1] = 1

    repaired = conserve_resampled_instance_ids(source, resampled)

    assert set(np.unique(repaired)) == {0, 1, 2}
    assert int(np.count_nonzero(repaired == 1)) == 1
    assert int(np.count_nonzero(repaired == 2)) == 1
    assert repaired.shape == resampled.shape


def test_native_export_audit_rejects_missing_subject_or_instance_loss(tmp_path: Path) -> None:
    expected = ("case-a", "case-b")
    good = [
        {
            "subject_id": "case-a",
            "semantic_path": str(tmp_path / "a.nii.gz"),
            "instances_path": str(tmp_path / "a_instances.nii.gz"),
            "preprocessed_instance_count": 2,
            "native_instance_count": 2,
            "label_geometry_exact": True,
        },
        {
            "subject_id": "case-b",
            "semantic_path": str(tmp_path / "b.nii.gz"),
            "instances_path": str(tmp_path / "b_instances.nii.gz"),
            "preprocessed_instance_count": 0,
            "native_instance_count": 0,
            "label_geometry_exact": True,
        },
    ]
    for record in good:
        Path(record["semantic_path"]).touch()
        Path(record["instances_path"]).touch()

    audit = validate_native_export_records(good, expected)
    assert audit["status"] == "passed"
    assert audit["semantic_file_count"] == 2
    assert audit["instance_file_count"] == 2

    with pytest.raises(ValueError, match="subjects"):
        validate_native_export_records(good[:1], expected)
    broken = [dict(record) for record in good]
    broken[0]["native_instance_count"] = 1
    with pytest.raises(ValueError, match="instance count"):
        validate_native_export_records(broken, expected)
    wrong_geometry = [dict(record) for record in good]
    wrong_geometry[1]["label_geometry_exact"] = False
    with pytest.raises(ValueError, match="label geometry"):
        validate_native_export_records(wrong_geometry, expected)
