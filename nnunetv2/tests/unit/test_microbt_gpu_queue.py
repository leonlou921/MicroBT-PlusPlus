from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.microbt_gpu_experiment_queue import (
    validate_exit_file,
    validate_test_summary,
)


def test_validate_exit_file_accepts_only_zero(tmp_path: Path) -> None:
    path = tmp_path / "stage.exit"
    path.write_text("0\n", encoding="utf-8")
    validate_exit_file(path)
    path.write_text("4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exit=4"):
        validate_exit_file(path)


def test_validate_test_summary_requires_complete_unique_251(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    payload = {
        "status": "complete",
        "subject_ids": [f"case-{index:03d}" for index in range(251)],
        "metrics": {"case_count": 251, "nonfinite_case_count": 0},
        "prediction_file_count": 251,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_test_summary(path, expected_count=251)
    payload["subject_ids"][-1] = payload["subject_ids"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unique"):
        validate_test_summary(path, expected_count=251)
