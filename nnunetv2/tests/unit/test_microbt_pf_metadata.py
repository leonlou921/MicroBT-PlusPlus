from __future__ import annotations

import hashlib
import json
import pickle
import stat
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from nnunetv2.training.dataloading.microbt_pf_metadata import build_metadata_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_case(source: Path, case_id: str) -> None:
    data = np.zeros((1, 6, 6, 6), dtype=np.float32)
    segmentation = np.zeros((1, 6, 6, 6), dtype=np.int16)
    segmentation[0, 1:3, 1:3, 1:3] = 1
    segmentation[0, 4, 4, 4] = 3
    np.savez_compressed(source / f"{case_id}.npz", data=data, seg=segmentation)
    properties = {
        "spacing": [2.0, 1.0, 1.0],
        "shape_before_cropping": (8, 8, 8),
        "bbox_used_for_cropping": [[1, 7], [1, 7], [1, 7]],
        "shape_after_cropping_and_before_resampling": (6, 6, 6),
        "sitk_stuff": {"origin": (0.0, 0.0, 0.0)},
    }
    with (source / f"{case_id}.pkl").open("wb") as handle:
        pickle.dump(properties, handle)


def _install_fake_dataset_module(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"infer": 0, "load": 0}

    class FakeDataset:
        def __init__(self, folder: str, identifiers: list[str] | None = None) -> None:
            self.source_folder = folder
            self.identifiers = ["case_001"] if identifiers is None else sorted(identifiers)

        def load_case(self, identifier: str):
            calls["load"] += 1
            archive = np.load(Path(self.source_folder) / f"{identifier}.npz")
            with (Path(self.source_folder) / f"{identifier}.pkl").open("rb") as handle:
                properties = pickle.load(handle)
            return archive["data"], archive["seg"], None, properties

    fake_module = types.ModuleType("nnunetv2.training.dataloading.nnunet_dataset")

    def infer_dataset_class(folder: str):
        calls["infer"] += 1
        assert Path(folder).name == "source"
        return FakeDataset

    fake_module.infer_dataset_class = infer_dataset_class
    monkeypatch.setitem(
        sys.modules, "nnunetv2.training.dataloading.nnunet_dataset", fake_module
    )
    return calls


def test_cache_build_is_read_only_atomic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    _write_case(source, "case_001")
    calls = _install_fake_dataset_module(monkeypatch)
    source_state = {
        path.name: (_sha256(path), path.stat().st_mtime_ns)
        for path in sorted(source.iterdir())
    }
    for path in source.iterdir():
        path.chmod(stat.S_IREAD)
    source.chmod(stat.S_IREAD | stat.S_IEXEC)

    try:
        outputs = build_metadata_cache(source, cache)
        case_directory = outputs[0]
        first_output_state = {
            path.name: (_sha256(path), path.stat().st_mtime_ns)
            for path in sorted(case_directory.iterdir())
        }
        resumed = build_metadata_cache(source, cache)
    finally:
        source.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        for path in source.iterdir():
            path.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert outputs == [cache / "case_001"]
    assert resumed == outputs
    assert calls == {"infer": 2, "load": 2}
    assert source_state == {
        path.name: (_sha256(path), path.stat().st_mtime_ns)
        for path in sorted(source.iterdir())
    }
    assert first_output_state == {
        path.name: (_sha256(path), path.stat().st_mtime_ns)
        for path in sorted(case_directory.iterdir())
    }
    with np.load(case_directory / "instances.npz") as archive:
        instances = archive["instance_ids"]
    assert instances.dtype == np.int32
    assert instances.max() == 2

    metadata = json.loads((case_directory / "metadata.json").read_text())
    assert metadata["spacing_zyx"] == [2.0, 1.0, 1.0]
    assert metadata["geometry"]["shape_before_cropping"] == [8, 8, 8]
    assert metadata["geometry"]["bbox_used_for_cropping"] == [[1, 7], [1, 7], [1, 7]]
    assert [item["id"] for item in metadata["instances"]] == [1, 2]
    assert all("center_zyx" in item and "bbox_zyx" in item for item in metadata["instances"])
    assert {item["relative_path"] for item in metadata["source_manifest"]} == {
        "case_001.npz",
        "case_001.pkl",
    }
    assert not list(cache.glob(".case_001.tmp-*"))


def test_source_hash_mismatch_fails_without_overwriting_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    _write_case(source, "case_001")
    _install_fake_dataset_module(monkeypatch)
    case_directory = build_metadata_cache(source, cache)[0]
    cached_before = {
        path.name: path.read_bytes() for path in sorted(case_directory.iterdir())
    }

    with (source / "case_001.pkl").open("rb") as handle:
        properties = pickle.load(handle)
    properties["spacing"] = [3.0, 1.0, 1.0]
    with (source / "case_001.pkl").open("wb") as handle:
        pickle.dump(properties, handle)

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        build_metadata_cache(source, cache)

    assert cached_before == {
        path.name: path.read_bytes() for path in sorted(case_directory.iterdir())
    }


def test_parallel_cache_build_preserves_sorted_atomic_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    identifiers = ("case_003", "case_001", "case_002")
    for identifier in identifiers:
        _write_case(source, identifier)
    _install_fake_dataset_module(monkeypatch)

    outputs = build_metadata_cache(
        source,
        cache,
        identifiers=identifiers,
        workers=2,
    )

    assert outputs == [cache / name for name in sorted(identifiers)]
    assert all((path / "metadata.json").is_file() for path in outputs)
    assert all((path / "instances.npz").is_file() for path in outputs)
    assert not list(cache.glob(".*.tmp-*"))


def test_cache_must_not_be_inside_read_only_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_case(source, "case_001")
    _install_fake_dataset_module(monkeypatch)

    with pytest.raises(ValueError, match="separate from the preprocessed source"):
        build_metadata_cache(source, source / "cache")


def test_cli_location_resolution_uses_plan_data_identifier(tmp_path: Path) -> None:
    from nnunetv2.scripts.build_microbt_pf_metadata import resolve_metadata_locations

    dataset_directory = tmp_path / "Dataset001_Test"
    cache_root = tmp_path / "cache"
    source = dataset_directory / "nnUNetPlans_3d_fullres"
    source.mkdir(parents=True)
    plans = {
        "configurations": {
            "3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres"}
        }
    }
    (dataset_directory / "nnUNetResEncUNetMPlans.json").write_text(
        json.dumps(plans), encoding="utf-8"
    )

    resolved_source, resolved_cache = resolve_metadata_locations(
        dataset_directory=dataset_directory,
        cache_root=cache_root,
        configuration="3d_fullres",
        plans_identifier="nnUNetResEncUNetMPlans",
    )

    assert resolved_source == source.resolve()
    assert resolved_cache == (
        cache_root
        / "Dataset001_Test"
        / "nnUNetResEncUNetMPlans"
        / "3d_fullres"
    ).resolve()
