from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nnunetv2.training.dataloading.microbt_pf_targets import (
    deterministic_interior_center,
    label_instances_26,
)


_GEOMETRY_KEYS = (
    "shape_before_cropping",
    "bbox_used_for_cropping",
    "shape_after_cropping_and_before_resampling",
    "sitk_stuff",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _hash_file(path: Path, digest: "hashlib._Hash") -> int:
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return size


def _manifest_entry(path: Path, source_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative_child = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative_child).to_bytes(8, "little"))
            digest.update(relative_child)
            size += _hash_file(child, digest)
    else:
        size = _hash_file(path, digest)
    return {
        "relative_path": path.relative_to(source_root).as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


def _source_manifest(paths: Iterable[Path], source_root: Path) -> list[dict[str, Any]]:
    return [_manifest_entry(path, source_root) for path in sorted(paths)]


def _content_signature(manifest: list[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (entry["relative_path"], entry["sha256"], int(entry["size_bytes"]))
        for entry in manifest
    )


def _case_source_paths(source: Path, identifier: str) -> list[Path]:
    candidates = (
        f"{identifier}.npz",
        f"{identifier}.npy",
        f"{identifier}_seg.npy",
        f"{identifier}.b2nd",
        f"{identifier}_seg.b2nd",
        f"{identifier}.pkl",
    )
    paths = [source / name for name in candidates if (source / name).exists()]
    if not paths:
        raise FileNotFoundError(f"No source artifacts found for {identifier}")
    return paths


def _segmentation_volume(segmentation: np.ndarray) -> np.ndarray:
    volume = np.asarray(segmentation)
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError("Segmentation must have shape [D,H,W] or [1,D,H,W]")
    return volume


def _instance_records(
    instance_ids: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for instance_id in range(1, int(instance_ids.max()) + 1):
        component = instance_ids == instance_id
        coordinates = np.argwhere(component)
        if not coordinates.size:
            continue
        center = deterministic_interior_center(component, spacing_zyx)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        records.append(
            {
                "id": instance_id,
                "center_zyx": list(center),
                "bbox_zyx": [
                    [int(lower[axis]), int(upper[axis])] for axis in range(3)
                ],
                "voxel_count": int(component.sum()),
            }
        )
    return records


def _existing_cache_matches(
    case_directory: Path, current_manifest: list[dict[str, Any]]
) -> bool:
    metadata_path = case_directory / "metadata.json"
    instances_path = case_directory / "instances.npz"
    if not metadata_path.is_file() or not instances_path.is_file():
        raise RuntimeError(f"Incomplete metadata cache at {case_directory}")
    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _content_signature(existing["source_manifest"]) != _content_signature(
        current_manifest
    ):
        raise RuntimeError(
            f"source hash mismatch for existing metadata cache {case_directory}"
        )
    return True


def build_case_metadata(
    identifier: str,
    segmentation: np.ndarray,
    properties: dict[str, Any],
    source_paths: Iterable[Path],
    source_root: Path,
    case_directory: Path,
) -> Path:
    source_root = Path(source_root).resolve()
    case_directory = Path(case_directory).resolve()
    source_paths = [Path(path).resolve() for path in source_paths]
    manifest = _source_manifest(source_paths, source_root)
    if case_directory.exists():
        _existing_cache_matches(case_directory, manifest)
        return case_directory

    spacing = tuple(float(value) for value in properties["spacing"])
    volume = _segmentation_volume(segmentation)
    instance_ids = label_instances_26(volume > 0)
    metadata = {
        "schema_version": 1,
        "case_identifier": identifier,
        "spacing_zyx": list(spacing),
        "geometry": {
            key: _jsonable(properties[key])
            for key in _GEOMETRY_KEYS
            if key in properties
        },
        "instances": _instance_records(instance_ids, spacing),
        "source_manifest": manifest,
    }

    case_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{identifier}.tmp-", dir=case_directory.parent)
    )
    try:
        np.savez_compressed(
            temporary / "instances.npz",
            instance_ids=instance_ids.astype(np.int32, copy=False),
        )
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, case_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return case_directory


def build_metadata_cache(
    preprocessed_case_directory: str | Path,
    cache_directory: str | Path,
    identifiers: Iterable[str] | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> list[Path]:
    source = Path(preprocessed_case_directory).resolve()
    cache = Path(cache_directory).resolve()
    if cache == source or source in cache.parents:
        raise ValueError("Metadata cache must be separate from the preprocessed source")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if workers < 1:
        raise ValueError("workers must be positive")

    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(str(source))
    requested = None if identifiers is None else sorted(str(item) for item in identifiers)
    dataset = dataset_class(str(source), requested)
    selected = list(dataset.identifiers)
    if limit is not None:
        selected = selected[:limit]

    def build_one(identifier: str) -> Path:
        case_dataset = (
            dataset
            if workers == 1
            else dataset_class(str(source), [identifier])
        )
        _, segmentation, _, properties = case_dataset.load_case(identifier)
        return build_case_metadata(
            identifier=identifier,
            segmentation=np.asarray(segmentation),
            properties=properties,
            source_paths=_case_source_paths(source, identifier),
            source_root=source,
            case_directory=cache / identifier,
        )

    if workers == 1:
        return [build_one(identifier) for identifier in selected]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(build_one, selected))
