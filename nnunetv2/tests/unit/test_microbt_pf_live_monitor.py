from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from nnunetv2.evaluation.microbt_pf_live_monitor import FixedCaseMonitor


def _write_contract(tmp_path: Path, *, wrong_sha: bool = False) -> Path:
    split = tmp_path / "preprocessed" / "splits_final.json"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps([{"train": ["train-a"], "val": ["case-a", "case-b"]}]),
        encoding="utf-8",
    )
    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset_name": "DatasetTest",
        "fold": 0,
        "split_file": "splits_final.json",
        "split_sha256": "0" * 64 if wrong_sha else digest,
        "connectivity": 26,
        "small_diameter_mm": 3.0,
        "decoder": {"field_min": 0.25, "foreground_level": 0.0},
        "inference": {
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "use_mirroring": True,
        },
        "milestone_every_epochs": 25,
        "cases": [
            {
                "subject_id": "case-a",
                "stratum": "small",
                "expected_count": 1,
                "expected_small_count": 1,
            },
            {
                "subject_id": "case-b",
                "stratum": "large",
                "expected_count": 2,
                "expected_small_count": 0,
            },
        ],
    }
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _Trainer:
    def __init__(self, tmp_path: Path) -> None:
        self.preprocessed_dataset_folder_base = str(tmp_path / "preprocessed")
        self.output_folder = str(tmp_path / "results" / "fold_0")
        self.fold = 0
        self.num_epochs = 200
        self.network = torch.nn.Linear(1, 1)


def _row(case: dict) -> dict:
    expected = int(case["expected_count"])
    return {
        "subject_id": case["subject_id"],
        "target_count": expected,
        "predicted_count": expected,
        "signed_count_error": 0,
        "branch_recall": 1.0,
    }


def test_from_trainer_preflight_rejects_split_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split SHA256"):
        FixedCaseMonitor.from_trainer(
            _Trainer(tmp_path),
            _write_contract(tmp_path, wrong_sha=True),
            tmp_path / "monitor",
            case_runner_factory=lambda *_: _row,
        )


def test_milestone_run_writes_atomic_artifacts_and_restores_train_mode(
    tmp_path: Path,
) -> None:
    trainer = _Trainer(tmp_path)
    trainer.network.train()
    monitor = FixedCaseMonitor.from_trainer(
        trainer,
        _write_contract(tmp_path),
        tmp_path / "monitor",
        case_runner_factory=lambda *_: _row,
        git_head="abc123",
    )

    assert monitor.run_if_due(24) is None
    assert not (tmp_path / "monitor" / "epoch_0024").exists()
    result = monitor.run_if_due(25)

    assert result is not None and result["status"] == "success"
    assert trainer.network.training is True
    assert (tmp_path / "monitor" / "fixed_cases_manifest.json").is_file()
    cases = json.loads(
        (tmp_path / "monitor" / "epoch_0025" / "cases.json").read_text()
    )
    summary = json.loads(
        (tmp_path / "monitor" / "epoch_0025" / "summary.json").read_text()
    )
    assert [row["subject_id"] for row in cases["cases"]] == ["case-a", "case-b"]
    assert summary["metrics"]["totals"] == {
        "target_count": 3,
        "predicted_count": 3,
    }
    history = (tmp_path / "monitor" / "fixed_case_history.jsonl").read_text()
    assert len(history.splitlines()) == 1


def test_runtime_failure_is_recorded_and_does_not_leak_eval_mode(
    tmp_path: Path,
) -> None:
    trainer = _Trainer(tmp_path)
    trainer.network.train()

    def failing(_case: dict) -> dict:
        raise RuntimeError("sentinel boom")

    monitor = FixedCaseMonitor.from_trainer(
        trainer,
        _write_contract(tmp_path),
        tmp_path / "monitor",
        case_runner_factory=lambda *_: failing,
    )
    result = monitor.run_if_due(25)

    assert result is not None and result["status"] == "failed"
    assert "sentinel boom" in result["error"]
    assert trainer.network.training is True
    on_disk = json.loads(
        (tmp_path / "monitor" / "epoch_0025" / "summary.json").read_text()
    )
    assert on_disk["status"] == "failed"


def test_final_epoch_runs_only_from_final_hook(tmp_path: Path) -> None:
    monitor = FixedCaseMonitor.from_trainer(
        _Trainer(tmp_path),
        _write_contract(tmp_path),
        tmp_path / "monitor",
        case_runner_factory=lambda *_: _row,
    )
    assert monitor.is_due(175)
    assert not monitor.is_due(200)
    assert monitor.is_due(200, final=True)
