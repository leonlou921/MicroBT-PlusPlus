from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from nnunetv2.topology.persistent_field_0d import (
    induced_matching_0d,
    profile_induced_matching_0d,
)
from scripts.profile_microbt_pf_h0 import (
    atomic_write_json,
    profile_field_pair,
    select_cached_cases,
)


def _literal_fields() -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(
        [[[0.9, -0.2, 0.7], [-0.4, 0.8, -0.5]]], dtype=np.float32
    )
    target = np.asarray(
        [[[1.0, -0.1, 0.6], [-0.3, 0.75, -0.6]]], dtype=np.float32
    )
    return prediction, target


def _assert_matching_equal(profiled, reference) -> None:
    np.testing.assert_array_equal(
        profiled.prediction.birth_flat, reference.prediction.birth_flat
    )
    np.testing.assert_array_equal(
        profiled.prediction.death_flat, reference.prediction.death_flat
    )
    np.testing.assert_array_equal(
        profiled.prediction.essential, reference.prediction.essential
    )
    np.testing.assert_array_equal(profiled.target.birth_flat, reference.target.birth_flat)
    np.testing.assert_array_equal(profiled.target.death_flat, reference.target.death_flat)
    np.testing.assert_array_equal(profiled.target.essential, reference.target.essential)
    np.testing.assert_array_equal(
        profiled.matched_prediction, reference.matched_prediction
    )
    np.testing.assert_array_equal(profiled.matched_target, reference.matched_target)
    np.testing.assert_array_equal(
        profiled.unmatched_prediction, reference.unmatched_prediction
    )
    np.testing.assert_array_equal(profiled.unmatched_target, reference.unmatched_target)


def test_profile_wrapper_preserves_validated_matching_and_native_timings() -> None:
    prediction, target = _literal_fields()

    reference = induced_matching_0d(prediction, target)
    profiled, profile_seconds = profile_induced_matching_0d(prediction, target)

    _assert_matching_equal(profiled, reference)
    assert set(profile_seconds) == {
        "prediction_pairs",
        "target_pairs",
        "comparison_matching",
        "total",
    }
    assert all(type(value) is float and value >= 0 for value in profile_seconds.values())


def test_profile_field_pair_reports_shape_counts_and_exact_stages() -> None:
    prediction, target = _literal_fields()

    result = profile_field_pair("literal", prediction, target, source="unit-test")

    assert result["name"] == "literal"
    assert result["source"] == "unit-test"
    assert result["shape"] == [1, 2, 3]
    assert result["voxel_count"] == 6
    assert result["prediction_pair_count"] >= 1
    assert result["target_pair_count"] >= 1
    assert result["matched_pair_count"] >= 1
    assert set(result["profile_seconds"]) == {
        "prediction_pairs",
        "target_pairs",
        "comparison_matching",
        "total",
    }
    assert result["precomputed_target_exact"] is True
    assert type(result["precomputed_target_total_seconds"]) is float
    assert result["precomputed_target_total_seconds"] >= 0
    assert result["ordering_exact"] is True
    assert set(result["ordering_seconds"]) == {"radix", "reference"}


def test_cached_case_selection_is_sorted_bounded_and_requires_complete_pairs(
    tmp_path: Path,
) -> None:
    for name in ("case-c", "case-a", "case-b"):
        case = tmp_path / "nested" / name
        case.mkdir(parents=True)
        (case / "instances.npz").touch()
        (case / "metadata.json").write_text("{}", encoding="utf-8")
    incomplete = tmp_path / "nested" / "case-0"
    incomplete.mkdir()
    (incomplete / "instances.npz").touch()

    selected = select_cached_cases(tmp_path, limit=2)

    assert [path.name for path in selected] == ["case-a", "case-b"]


def test_atomic_write_json_replaces_destination_without_temp_residue(tmp_path: Path) -> None:
    output = tmp_path / "profiles" / "baseline.json"
    output.parent.mkdir()
    output.write_text("stale", encoding="utf-8")

    atomic_write_json(output, {"schema_version": 1, "profiles": []})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "profiles": [],
    }
    assert list(output.parent.iterdir()) == [output]


def test_profile_cli_can_import_repository_when_executed_by_path() -> None:
    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(repository / "scripts" / "profile_microbt_pf_h0.py"), "--help"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--cache-root" in completed.stdout


def test_validation_profiler_excludes_warmup_and_preserves_audit_count() -> None:
    module = importlib.import_module("nnunetv2.scripts.profile_microbt_pf_validation")

    class Target:
        topology = None

    class Trainer:
        device = torch.device("cpu")
        validation_h0_audit_iterations = 2

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.exact_fast_telemetry = {"validation_exact_h0_calls": 0}
            self._exact_fast_validation_index = 0

        def on_validation_epoch_start(self) -> None:
            self._exact_fast_validation_index = 0
            self.exact_fast_telemetry["validation_exact_h0_calls"] = 0

        def validation_step(self, batch: dict) -> dict:
            audit = (
                self._exact_fast_validation_index
                < self.validation_h0_audit_iterations
            )
            self._exact_fast_validation_index += 1
            self.calls.append("audit" if audit else "monitor")
            if audit:
                self.exact_fast_telemetry["validation_exact_h0_calls"] += 1
            return {"loss": np.asarray(1.0, dtype=np.float32)}

        def on_validation_epoch_end(self, outputs: list[dict]) -> None:
            assert len(outputs) == 3

    trainer = Trainer()
    loader = iter(
        [{"persistent_targets": (Target(),)} for _ in range(4)]
    )

    result = module.run_validation_profile(trainer, loader, steps=3, warmup=True)

    assert trainer.calls == ["monitor", "audit", "audit", "monitor"]
    assert result["steps"] == 3
    assert result["exact_h0_calls"] == 2
    assert result["topology_present_batches"] == 0
    assert result["loss_first"] == 1.0
    assert result["loss_last"] == 1.0
    assert result["wall_seconds"] >= 0
    assert result["median_batch_wait_seconds"] >= 0
    assert result["median_step_seconds"] >= 0
    assert result["p95_step_seconds"] >= result["median_step_seconds"]
