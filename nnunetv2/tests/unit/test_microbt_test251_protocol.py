from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evaluate_microbt_umctf_test251 import (
    dataset_ap_1vox,
    load_test_subjects,
    resolve_met_case_files,
    select_test_subjects,
)


def test_load_test_subjects_requires_exact_unique_251(tmp_path: Path) -> None:
    labels = tmp_path / "labelsTs"
    labels.mkdir()
    for index in range(251):
        (labels / f"EXAMPLE__ExampleCase-{index:05d}-000.nii.gz").touch()
    subjects = load_test_subjects(labels, expected_count=251)
    assert len(subjects) == 251
    assert len(set(subjects)) == 251

    (labels / "unexpected.txt").touch()
    assert load_test_subjects(labels, expected_count=251) == subjects


def test_load_test_subjects_rejects_incomplete_test_set(tmp_path: Path) -> None:
    labels = tmp_path / "labelsTs"
    labels.mkdir()
    (labels / "EXAMPLE__ExampleCase-00001-000.nii.gz").touch()
    with pytest.raises(ValueError, match="exactly 251"):
        load_test_subjects(labels, expected_count=251)


def test_resolve_met_case_files_uses_frozen_modality_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    case = source / "ExampleCase-00002-000"
    case.mkdir(parents=True)
    for suffix in ("t1c", "t1n", "t2f", "t2w"):
        (case / f"ExampleCase-00002-000-{suffix}.nii.gz").touch()
    paths = resolve_met_case_files(source, "EXAMPLE__ExampleCase-00002-000")
    assert [path.name.rsplit("-", 1)[-1] for path in paths] == [
        "t1c.nii.gz",
        "t1n.nii.gz",
        "t2f.nii.gz",
        "t2w.nii.gz",
    ]


def test_resolve_met_case_files_accepts_aligned_flat_nnunet_test_folder(tmp_path: Path) -> None:
    source = tmp_path / "imagesTs"
    source.mkdir()
    for index in range(4):
        (source / f"EXAMPLE__ExampleCase-00002-000_{index:04d}.nii.gz").touch()
    paths = resolve_met_case_files(source, "EXAMPLE__ExampleCase-00002-000")
    assert [path.name for path in paths] == [
        f"EXAMPLE__ExampleCase-00002-000_{index:04d}.nii.gz" for index in range(4)
    ]


def test_select_test_subjects_accepts_only_a_frozen_unique_subset(tmp_path: Path) -> None:
    labels = tmp_path / "labelsTs"
    labels.mkdir()
    subjects = [f"EXAMPLE__ExampleCase-{index:05d}-000" for index in range(251)]
    for subject in subjects:
        (labels / f"{subject}.nii.gz").touch()
    subset_file = tmp_path / "subjects.json"
    subset_file.write_text(__import__("json").dumps(subjects[::3]), encoding="utf-8")
    selected = select_test_subjects(
        labels,
        expected_count=len(subjects[::3]),
        total_expected_count=251,
        subject_ids_json=subset_file,
    )
    assert selected == tuple(subjects[::3])


def test_test251_runner_defaults_all_future_inference_to_seven_workers() -> None:
    root = Path(__file__).resolve().parents[3]
    text = (root / "scripts/run_microbt_umctf_test251.sh").read_text(encoding="utf-8")
    assert 'export nnUNet_results=$(dirname "$(dirname "${model_folder}")")' in text
    assert 'workers=${MICROBT_TEST251_WORKERS:-7}' in text
    assert "test251_shard_accel.py" in text
    assert "--subject-ids-json" in text


def test_ap_1vox_uses_global_score_order_and_one_to_one_matching() -> None:
    records = [
        {
            "targets": [{"box_zyx": [0, 0, 0, 2, 2, 2]}],
            "detections": [
                {"box_zyx": [5, 5, 5, 6, 6, 6], "score": 0.9},
                {"box_zyx": [1, 1, 1, 3, 3, 3], "score": 0.8},
                {"box_zyx": [0, 0, 0, 1, 1, 1], "score": 0.7},
            ],
        }
    ]
    assert dataset_ap_1vox(records) == pytest.approx(0.5)
