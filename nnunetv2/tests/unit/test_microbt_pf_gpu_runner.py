from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import nnunetv2.scripts.microbt_pf_gpu_feasibility as gpu_runner

from nnunetv2.scripts.microbt_pf_gpu_feasibility import (
    build_nnunet_dataset,
    feasibility_gate,
    select_cached_cases,
    write_gpu_artifact,
)


def test_feasibility_gate_uses_preregistered_inclusive_limits() -> None:
    a0 = {
        "median_step_time_seconds": 1.0,
        "max_memory_allocated_bytes": 10_000,
        "nonfinite_steps": 0,
        "skipped_updates": 0,
    }
    a3 = {
        "median_step_time_seconds": 1.5,
        "max_memory_allocated_bytes": 11_000,
        "nonfinite_steps": 0,
        "skipped_updates": 0,
    }
    observed = feasibility_gate(a0, a3)
    assert observed == {
        "memory_ratio_a3_over_a0": 1.1,
        "step_time_ratio_a3_over_a0": 1.5,
        "passed": True,
        "reasons": [],
    }


def test_feasibility_gate_reports_each_failed_condition() -> None:
    a0 = {
        "median_step_time_seconds": 1.0,
        "max_memory_allocated_bytes": 10_000,
        "nonfinite_steps": 0,
        "skipped_updates": 0,
    }
    a3 = {
        "median_step_time_seconds": 1.6,
        "max_memory_allocated_bytes": 11_001,
        "nonfinite_steps": 1,
        "skipped_updates": 2,
    }
    observed = feasibility_gate(a0, a3)
    assert observed["passed"] is False
    assert observed["reasons"] == [
        "a3_nonfinite_steps",
        "a3_skipped_updates",
        "memory_ratio_exceeds_1.10",
        "step_time_ratio_exceeds_1.50",
    ]


def test_cached_case_selection_is_sorted_and_exact(tmp_path: Path) -> None:
    for name in ("case_c", "case_a", "case_b"):
        (tmp_path / name).mkdir()
    assert select_cached_cases(tmp_path, 2) == ("case_a", "case_b")


def test_gpu_artifact_is_atomic_native_json(tmp_path: Path) -> None:
    output = tmp_path / "gpu.json"
    write_gpu_artifact(output, {"ratio": 1.25, "count": 5, "passed": False})
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "ratio": 1.25,
        "count": 5,
        "passed": False,
    }


def test_build_loader_gives_nnunet_dataset_mutable_case_identifiers(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    class SortingDataset:
        def __init__(self, _folder: str, identifiers: list[str]) -> None:
            observed["identifier_type"] = type(identifiers)
            identifiers.sort()
            observed["identifiers"] = identifiers

    class CapturingLoader:
        def __init__(self, dataset, **kwargs) -> None:
            observed["dataset"] = dataset
            observed["loader_kwargs"] = kwargs

    monkeypatch.setattr(gpu_runner, "infer_dataset_class", lambda _folder: SortingDataset)
    monkeypatch.setattr(gpu_runner, "MicroBTPFDataLoader", CapturingLoader)

    gpu_runner._build_loader(
        source=tmp_path / "source",
        cache=tmp_path / "cache",
        cases=("case_b", "case_a"),
        configuration_manager=SimpleNamespace(patch_size=(8, 8, 8)),
        label_manager=object(),
        variant="A3",
    )

    assert observed["identifier_type"] is list
    assert observed["identifiers"] == ["case_a", "case_b"]


def test_full_volume_dataset_also_receives_mutable_case_identifiers(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    class SortingDataset:
        def __init__(self, _folder: str, identifiers: list[str]) -> None:
            observed["identifier_type"] = type(identifiers)
            identifiers.sort()
            observed["identifiers"] = identifiers

    monkeypatch.setattr(gpu_runner, "infer_dataset_class", lambda _folder: SortingDataset)
    build_nnunet_dataset(tmp_path, ("case_b", "case_a"))
    assert observed == {
        "identifier_type": list,
        "identifiers": ["case_a", "case_b"],
    }
