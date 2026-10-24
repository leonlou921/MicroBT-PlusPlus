from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nnunetv2.evaluation.microbt_pf_training_monitor import (
    summarize_component_outputs,
    upsert_jsonl,
)


def _step(
    *,
    loss: float,
    region: float,
    field: float,
    weighted_betti: float,
    raw_betti: float,
    active: bool,
) -> dict[str, np.ndarray]:
    return {
        "loss": np.asarray(loss, dtype=np.float32),
        "region": np.asarray(region, dtype=np.float32),
        "field": np.asarray(field, dtype=np.float32),
        "betti_0": np.asarray(weighted_betti, dtype=np.float32),
        "persistent_field": np.asarray(field + weighted_betti, dtype=np.float32),
        "betti_0_raw": np.asarray(raw_betti, dtype=np.float32),
        "h0_active": np.asarray(int(active), dtype=np.int64),
    }


def test_component_summary_uses_active_steps_for_raw_betti() -> None:
    # Break caught: averaging raw Betti over inactive zero-valued steps would
    # under-report topology loss by the sparse interval.
    outputs = [
        _step(loss=2.6, region=0.4, field=0.2, weighted_betti=2.0, raw_betti=0.2, active=True),
        _step(loss=0.8, region=0.5, field=0.3, weighted_betti=0.0, raw_betti=0.0, active=False),
        _step(loss=4.8, region=0.6, field=0.2, weighted_betti=4.0, raw_betti=0.4, active=True),
    ]

    summary = summarize_component_outputs(
        outputs,
        phase="train",
        epoch=7,
        schedule={"mode": "sparse_exact", "effective_h0_weight": 10.0},
    )

    assert summary == {
        "schema_version": 1,
        "phase": "train",
        "epoch": 7,
        "steps": 3,
        "h0_active_steps": 2,
        "h0_inactive_steps": 1,
        "loss_mean": pytest.approx(8.2 / 3.0),
        "region_mean": pytest.approx(0.5),
        "field_mean": pytest.approx(0.7 / 3.0),
        "betti_0_raw_active_mean": pytest.approx(0.3),
        "betti_0_weighted_all_mean": pytest.approx(2.0),
        "persistent_field_mean": pytest.approx(6.7 / 3.0),
        "schedule": {"effective_h0_weight": 10.0, "mode": "sparse_exact"},
    }


def test_component_summary_rejects_nonfinite_values() -> None:
    # Break caught: JSON nulls in optimization losses would conceal a corrupt
    # training epoch.
    outputs = [
        _step(
            loss=float("nan"),
            region=0.4,
            field=0.2,
            weighted_betti=2.0,
            raw_betti=0.2,
            active=True,
        )
    ]
    with pytest.raises(ValueError, match="nonfinite loss"):
        summarize_component_outputs(outputs, phase="train", epoch=0, schedule={})


def test_component_summary_reports_sbpf_birth_completeness_curve() -> None:
    first = _step(
        loss=5.0,
        region=1.0,
        field=1.0,
        weighted_betti=1.0,
        raw_betti=0.1,
        active=True,
    )
    first.update(
        birth_completeness=np.asarray(2.0),
        birth_completeness_raw=np.asarray(0.2),
        object_morse=np.asarray(4.0),
    )
    second = _step(
        loss=2.0,
        region=1.0,
        field=1.0,
        weighted_betti=0.0,
        raw_betti=0.0,
        active=False,
    )
    second.update(
        birth_completeness=np.asarray(0.0),
        birth_completeness_raw=np.asarray(0.0),
        object_morse=np.asarray(1.0),
    )

    summary = summarize_component_outputs(
        [first, second], phase="train", epoch=2, schedule={}
    )

    assert summary["birth_completeness_raw_active_mean"] == pytest.approx(0.2)
    assert summary["birth_completeness_weighted_all_mean"] == pytest.approx(1.0)
    assert summary["object_morse_mean"] == pytest.approx(2.5)


def test_component_summary_reports_spp_false_persistence_curve() -> None:
    first = _step(
        loss=5.0,
        region=1.0,
        field=1.0,
        weighted_betti=2.0,
        raw_betti=0.2,
        active=True,
    )
    first["false_persistence"] = np.asarray(1.5)
    second = _step(
        loss=2.0,
        region=1.0,
        field=1.0,
        weighted_betti=0.0,
        raw_betti=0.0,
        active=False,
    )
    second["false_persistence"] = np.asarray(0.0)

    summary = summarize_component_outputs(
        [first, second], phase="train", epoch=2, schedule={}
    )

    assert summary["false_persistence_weighted_all_mean"] == pytest.approx(0.75)


def test_component_summary_reports_every_step_obr_curves() -> None:
    first = _step(
        loss=4.0,
        region=1.0,
        field=0.5,
        weighted_betti=1.0,
        raw_betti=0.1,
        active=True,
    )
    first.update(
        birth_reservation=np.asarray(1.0),
        birth_reservation_raw=np.asarray(2.0),
        birth_existence=np.asarray(0.75),
        birth_existence_raw=np.asarray(1.5),
        birth_prominence=np.asarray(0.25),
        birth_prominence_raw=np.asarray(0.5),
        object_morse=np.asarray(2.5),
    )
    second = _step(
        loss=2.0,
        region=1.0,
        field=0.5,
        weighted_betti=0.0,
        raw_betti=0.0,
        active=False,
    )
    second.update(
        birth_reservation=np.asarray(0.5),
        birth_reservation_raw=np.asarray(1.0),
        birth_existence=np.asarray(0.4),
        birth_existence_raw=np.asarray(0.8),
        birth_prominence=np.asarray(0.1),
        birth_prominence_raw=np.asarray(0.2),
        object_morse=np.asarray(1.0),
    )

    summary = summarize_component_outputs(
        [first, second], phase="train", epoch=4, schedule={}
    )

    assert summary["birth_reservation_weighted_all_mean"] == pytest.approx(0.75)
    assert summary["birth_reservation_raw_all_mean"] == pytest.approx(1.5)
    assert summary["birth_existence_raw_all_mean"] == pytest.approx(1.15)
    assert summary["birth_prominence_raw_all_mean"] == pytest.approx(0.35)
    assert summary["object_morse_mean"] == pytest.approx(1.75)


def test_upsert_jsonl_replaces_matching_phase_epoch_atomically(tmp_path: Path) -> None:
    # Break caught: a resumed epoch must replace, not duplicate, the same
    # phase/epoch learning-curve point.
    path = tmp_path / "learning_components.jsonl"
    upsert_jsonl(path, {"phase": "train", "epoch": 3, "loss_mean": 1.0}, ("phase", "epoch"))
    upsert_jsonl(path, {"phase": "val", "epoch": 3, "loss_mean": 2.0}, ("phase", "epoch"))
    upsert_jsonl(path, {"phase": "train", "epoch": 3, "loss_mean": 0.75}, ("phase", "epoch"))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"epoch": 3, "loss_mean": 0.75, "phase": "train"},
        {"epoch": 3, "loss_mean": 2.0, "phase": "val"},
    ]
    assert not list(tmp_path.glob("*.tmp"))
