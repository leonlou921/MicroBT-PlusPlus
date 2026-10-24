from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nnunetv2.scripts.run_microbt_pf_full203 as runner_module
from nnunetv2.scripts.run_microbt_pf_full203 import (
    InferenceSelection,
    load_validation_subjects,
    run_admission,
    run_checkpoint,
    reusable_case_record,
)


def test_split_requires_exactly_203_unique_sorted_subjects(tmp_path: Path) -> None:
    subjects = [f"case_{index:03d}" for index in range(203)]
    split = tmp_path / "splits_final.json"
    split.write_text(json.dumps([{"train": ["train"], "val": list(reversed(subjects))}]))
    assert load_validation_subjects(split, 0) == tuple(subjects)
    split.write_text(json.dumps([{"train": [], "val": subjects[:-1]}]))
    with pytest.raises(ValueError, match="exactly 203"):
        load_validation_subjects(split, 0)


def test_resume_accepts_only_matching_record(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    payload = {
        "schema_version": 1,
        "subject_id": "case",
        "checkpoint_sha256": "checkpoint",
        "evaluator_sha256": "evaluator",
        "elapsed_seconds": 1.0,
        "metrics": {},
        "detections": [],
        "targets": [],
    }
    path.write_text(json.dumps(payload))
    assert reusable_case_record(path, "case", "checkpoint", "evaluator") == payload
    assert reusable_case_record(path, "case", "changed", "evaluator") is None
    assert reusable_case_record(
        path,
        "case",
        "checkpoint",
        "new-evaluator",
        compatible_evaluator_sha256=("evaluator",),
    ) == payload


def test_runner_source_has_no_final_fixed_monitor_dependency() -> None:
    source = Path(__file__).parents[2] / "scripts" / "run_microbt_pf_full203.py"
    text = source.read_text(encoding="utf-8")
    assert "run_microbt_pf_fixed_monitor" not in text
    assert "fixed_case_history" not in text


def test_admission_falls_back_to_stock_when_no_fast_candidate_is_exact(
    tmp_path: Path, monkeypatch
) -> None:
    def make_predictor(_folder, checkpoint, _fold, *, fast, batch_size, device):
        return SimpleNamespace(
            checkpoint=checkpoint,
            fast=fast,
            forward_microbatch_size=batch_size,
            network=SimpleNamespace(to=lambda _device: None),
        )

    def infer_case(predictor, _dataset, _subject):
        value = 1 if predictor.fast else 0
        restored = {
            "segmentation": np.asarray([[[value]]], dtype=np.uint8),
            "instance_ids": np.asarray([[[value]]], dtype=np.int32),
            "count": value,
            "metrics": {
                "box_ap50": None,
                "box_ap75": None,
                "small_lesion_recall_3mm": None,
                "small_proposal_recall_3mm": None,
            },
        }
        return restored, np.zeros((1, 1, 1), dtype=np.uint8), (1.0, 1.0, 1.0), 1.0, 0

    monkeypatch.setattr(runner_module, "_make_predictor", make_predictor)
    monkeypatch.setattr(runner_module, "_make_dataset", lambda *_args: None)
    monkeypatch.setattr(runner_module, "_infer_case", infer_case)
    monkeypatch.setattr(runner_module, "_admission_view", lambda restored, *_args: restored)
    monkeypatch.setattr(runner_module, "_release", lambda *_args: None)

    selection = run_admission(
        tmp_path,
        {"best": "checkpoint_best.pth", "final": "checkpoint_final.pth"},
        0,
        "case",
        (1, 2),
        tmp_path / "admission",
        runner_module.torch.device("cpu"),
    )

    assert selection == InferenceSelection(mode="stock", batch_size=None)
    payload = json.loads((tmp_path / "admission" / "admission.json").read_text())
    assert payload["selected_mode"] == "stock"
    assert payload["selected_batch_size"] is None
    assert len(payload["checkpoints"]["best"]["trials"]) == 2
    assert payload["checkpoints"]["final"]["trials"] == []


def test_checkpoint_runner_constructs_stock_predictor_for_stock_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []

    def make_predictor(*_args, **kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(network=SimpleNamespace(to=lambda _device: None))

    monkeypatch.setattr(runner_module, "_make_predictor", make_predictor)
    monkeypatch.setattr(runner_module, "_make_dataset", lambda *_args: None)
    monkeypatch.setattr(
        runner_module,
        "_infer_case",
        lambda *_args: (
            {"segmentation": np.zeros((1, 1, 1), dtype=np.uint8)},
            np.zeros((1, 1, 1), dtype=np.uint8),
            (1.0, 1.0, 1.0),
            0.5,
            0,
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "build_case_record",
        lambda subject, *_args, checkpoint_sha256, evaluator_sha256, elapsed_seconds, **_kwargs: {
            "schema_version": 1,
            "subject_id": subject,
            "checkpoint_sha256": checkpoint_sha256,
            "evaluator_sha256": evaluator_sha256,
            "elapsed_seconds": elapsed_seconds,
            "metrics": {},
            "detections": [],
            "targets": [],
        },
    )
    monkeypatch.setattr(
        runner_module,
        "summarize_full203",
        lambda *_args, **_kwargs: {"macro": {}},
    )
    monkeypatch.setattr(runner_module, "_write_per_case_csv", lambda *_args: None)
    monkeypatch.setattr(runner_module, "_release", lambda *_args: None)

    _records, summary = run_checkpoint(
        "best",
        tmp_path,
        "checkpoint_best.pth",
        "checkpoint",
        0,
        ("case",),
        InferenceSelection(mode="stock", batch_size=None),
        tmp_path / "output",
        "evaluator",
        tmp_path / "status.json",
        runner_module.torch.device("cpu"),
    )

    assert calls == [{"fast": False, "batch_size": 1, "device": runner_module.torch.device("cpu")}]
    assert summary["inference_mode"] == "stock"
    assert summary["forward_microbatch_size"] is None
