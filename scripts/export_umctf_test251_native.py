#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from scripts.evaluate_microbt_umctf_test251 import resolve_met_case_files


_WORKER: dict[str, Any] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def regions_to_semantic(
    region_mask: np.ndarray, regions_class_order: Sequence[int]
) -> np.ndarray:
    regions = np.asarray(region_mask)
    order = tuple(int(value) for value in regions_class_order)
    if regions.ndim != 4 or regions.shape[0] != len(order):
        raise ValueError("region mask and regions_class_order are incompatible")
    semantic = np.zeros(regions.shape[1:], dtype=np.uint16)
    for index, label_value in enumerate(order):
        semantic[regions[index] > 0] = label_value
    return semantic


def undo_crop_and_transpose(
    cropped: np.ndarray,
    properties: Mapping[str, Any],
    transpose_backward: Sequence[int],
) -> np.ndarray:
    array = np.asarray(cropped)
    shape = tuple(int(value) for value in properties["shape_before_cropping"])
    output = np.zeros(shape, dtype=array.dtype)
    bbox = properties["bbox_used_for_cropping"]
    slicer = tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)
    if output[slicer].shape != array.shape:
        raise ValueError("cropped native prediction does not match preprocessing bbox")
    output[slicer] = array
    return output.transpose(tuple(int(value) for value in transpose_backward))


def validate_native_export_records(
    records: Sequence[Mapping[str, Any]], expected_subjects: Sequence[str]
) -> dict[str, Any]:
    expected = tuple(sorted(str(value) for value in expected_subjects))
    actual = tuple(sorted(str(record["subject_id"]) for record in records))
    if len(expected) != len(set(expected)) or actual != expected or len(actual) != len(set(actual)):
        raise ValueError("native export subjects are not exact and unique")
    for record in records:
        if int(record["preprocessed_instance_count"]) != int(record["native_instance_count"]):
            raise ValueError(f"native instance count loss for {record['subject_id']}")
        if not bool(record.get("label_geometry_exact", False)):
            raise ValueError(f"native label geometry mismatch for {record['subject_id']}")
        if not Path(record["semantic_path"]).is_file():
            raise ValueError(f"missing native semantic file for {record['subject_id']}")
        if not Path(record["instances_path"]).is_file():
            raise ValueError(f"missing native instances file for {record['subject_id']}")
    return {
        "status": "passed",
        "subject_count": len(actual),
        "semantic_file_count": len(actual),
        "instance_file_count": len(actual),
        "instance_count_conserved": True,
    }


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def conserve_resampled_instance_ids(
    source_instances: np.ndarray,
    resampled_instances: np.ndarray,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Restore instance IDs that nearest-neighbour downsampling did not sample."""
    source = np.asarray(source_instances)
    repaired = np.asarray(resampled_instances).copy()
    if source.ndim != repaired.ndim or source.ndim != 3:
        raise ValueError("instance conservation requires aligned 3D grids")
    if support_mask is not None and np.asarray(support_mask).shape != repaired.shape:
        raise ValueError("instance conservation support mask has the wrong shape")

    expected = set(int(value) for value in np.unique(source)) - {0}
    present = set(int(value) for value in np.unique(repaired)) - {0}
    if not present.issubset(expected):
        raise ValueError("resampling introduced an unknown instance ID")

    source_shape = np.asarray(source.shape, dtype=np.float64)
    target_shape = np.asarray(repaired.shape, dtype=np.float64)
    support = None if support_mask is None else np.asarray(support_mask, dtype=bool)
    for instance_id in sorted(expected - present):
        coordinates = np.argwhere(source == instance_id)
        mapped_centroid = (
            (coordinates.mean(axis=0) + 0.5) * target_shape / source_shape - 0.5
        )
        available = repaired == 0
        if support is not None and np.any(available & support):
            available &= support
        candidates = np.argwhere(available)
        if candidates.size == 0:
            raise ValueError("native grid has no free voxel for a dropped instance ID")
        distances = np.sum((candidates - mapped_centroid[None]) ** 2, axis=1)
        repaired[tuple(candidates[int(np.argmin(distances))])] = instance_id

    actual = set(int(value) for value in np.unique(repaired)) - {0}
    if actual != expected:
        raise ValueError("failed to conserve resampled instance IDs")
    return repaired


def _restore_prediction(
    region_mask: np.ndarray,
    instance_labels: np.ndarray,
    properties: Mapping[str, Any],
    configuration_manager: Any,
    plans_manager: PlansManager,
    regions_class_order: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    regions = np.asarray(region_mask, dtype=np.uint8)
    instances = np.asarray(instance_labels, dtype=np.int32)
    if regions.ndim != 4 or regions.shape[0] != len(tuple(regions_class_order)):
        raise ValueError("saved region mask has an invalid shape")
    if instances.shape != regions.shape[1:]:
        raise ValueError("saved semantic and instance grids do not align")

    target_shape = tuple(
        int(value) for value in properties["shape_after_cropping_and_before_resampling"]
    )
    original_spacing = [
        float(properties["spacing"][index]) for index in plans_manager.transpose_forward
    ]
    current_spacing = list(float(value) for value in configuration_manager.spacing)
    if len(current_spacing) < len(target_shape):
        current_spacing = [original_spacing[0], *current_spacing]
    regions_cropped = _numpy(
        configuration_manager.resampling_fn_seg(
            regions, target_shape, current_spacing, original_spacing
        )
    ).astype(np.uint8, copy=False)
    instances_cropped = _numpy(
        configuration_manager.resampling_fn_seg(
            instances[None], target_shape, current_spacing, original_spacing
        )
    )[0].astype(np.int32, copy=False)
    instances_cropped = conserve_resampled_instance_ids(
        instances,
        instances_cropped,
        support_mask=np.any(regions_cropped > 0, axis=0),
    )
    semantic_cropped = regions_to_semantic(regions_cropped, regions_class_order)
    semantic_native = undo_crop_and_transpose(
        semantic_cropped, properties, plans_manager.transpose_backward
    ).astype(np.uint16, copy=False)
    instances_native = undo_crop_and_transpose(
        instances_cropped, properties, plans_manager.transpose_backward
    ).astype(np.int32, copy=False)
    return semantic_native, instances_native


def _geometry_signature(properties: Mapping[str, Any]) -> dict[str, list[float]]:
    sitk = properties.get("sitk_stuff", {})
    return {
        "spacing": [float(value) for value in properties["spacing"]],
        "origin": [float(value) for value in sitk.get("origin", ())],
        "direction": [float(value) for value in sitk.get("direction", ())],
    }


def _geometry_matches(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    left = _geometry_signature(first)
    right = _geometry_signature(second)
    return all(
        len(left[key]) == len(right[key])
        and np.allclose(left[key], right[key], rtol=0.0, atol=1e-6)
        for key in left
    )


def _write_seg_atomic(writer: Any, array: np.ndarray, output: Path, properties: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.nii.gz")
    writer.write_seg(array, str(temporary), properties)
    temporary.replace(output)


def _initialize_worker(
    model_folder: str,
    evaluation_output: str,
    source_root: str,
    labels_ts: str,
    checkpoint_sha256: str,
    evaluator_sha256: str,
    configuration: str,
) -> None:
    folder = Path(model_folder)
    plans = PlansManager(folder / "plans.json")
    dataset_json = load_json(folder / "dataset.json")
    configuration_manager = plans.get_configuration(configuration)
    label_manager = plans.get_label_manager(dataset_json)
    if not label_manager.has_regions or label_manager.regions_class_order is None:
        raise RuntimeError("native UMCTF export requires the frozen region label contract")
    _WORKER.clear()
    _WORKER.update(
        {
            "plans": plans,
            "configuration": configuration_manager,
            "dataset_json": dataset_json,
            "preprocessor": configuration_manager.preprocessor_class(verbose=False),
            "writer": plans.image_reader_writer_class(),
            "regions_class_order": tuple(label_manager.regions_class_order),
            "allowed_labels": set(int(value) for value in label_manager.all_labels),
            "evaluation_output": Path(evaluation_output),
            "source_root": Path(source_root),
            "labels_ts": Path(labels_ts),
            "checkpoint_sha256": checkpoint_sha256,
            "evaluator_sha256": evaluator_sha256,
        }
    )


def _export_subject(subject_id: str) -> dict[str, Any]:
    context = _WORKER
    evaluation_output: Path = context["evaluation_output"]
    prediction_path = evaluation_output / "predictions" / f"{subject_id}.npz"
    case_path = evaluation_output / "cases" / f"{subject_id}.json"
    native_root = evaluation_output / "native_nifti"
    semantic_path = native_root / "semantic" / f"{subject_id}.nii.gz"
    instances_path = native_root / "instances" / f"{subject_id}.nii.gz"
    record_path = native_root / "cases" / f"{subject_id}.json"
    prediction_sha256 = sha256(prediction_path)

    if record_path.is_file() and semantic_path.is_file() and instances_path.is_file():
        candidate = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            candidate.get("prediction_sha256") == prediction_sha256
            and candidate.get("semantic_sha256") == sha256(semantic_path)
            and candidate.get("instances_sha256") == sha256(instances_path)
            and candidate.get("checkpoint_sha256") == context["checkpoint_sha256"]
            and candidate.get("evaluator_sha256") == context["evaluator_sha256"]
            and candidate.get("label_geometry_exact") is True
        ):
            candidate["reused"] = True
            return candidate

    image_files = resolve_met_case_files(context["source_root"], subject_id)
    data, _, properties = context["preprocessor"].run_case(
        [str(path) for path in image_files],
        None,
        context["plans"],
        context["configuration"],
        context["dataset_json"],
    )
    with np.load(prediction_path) as prediction:
        region_mask = np.asarray(prediction["region_mask"])
        instance_labels = np.asarray(prediction["instance_labels"])
    if tuple(data.shape[1:]) != tuple(instance_labels.shape):
        raise ValueError(f"saved prediction grid drift for {subject_id}")
    del data
    semantic_native, instances_native = _restore_prediction(
        region_mask,
        instance_labels,
        properties,
        context["configuration"],
        context["plans"],
        context["regions_class_order"],
    )
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if case.get("checkpoint_sha256") != context["checkpoint_sha256"]:
        raise ValueError(f"case checkpoint drift for {subject_id}")
    if case.get("evaluator_sha256") != context["evaluator_sha256"]:
        raise ValueError(f"case evaluator drift for {subject_id}")
    preprocessed_count = len(np.unique(instance_labels)) - int(np.any(instance_labels == 0))
    native_count = len(np.unique(instances_native)) - int(np.any(instances_native == 0))
    if int(case["predicted_count"]) != preprocessed_count:
        raise ValueError(f"case/prediction count drift for {subject_id}")
    if native_count != preprocessed_count:
        raise ValueError(f"native instance count loss for {subject_id}")
    if not set(int(value) for value in np.unique(semantic_native)).issubset(
        context["allowed_labels"]
    ):
        raise ValueError(f"invalid semantic label after native export for {subject_id}")

    _write_seg_atomic(context["writer"], semantic_native, semantic_path, properties)
    _write_seg_atomic(context["writer"], instances_native, instances_path, properties)
    semantic_loaded, semantic_properties = context["writer"].read_seg(str(semantic_path))
    instances_loaded, instances_properties = context["writer"].read_seg(str(instances_path))
    label_loaded, label_properties = context["writer"].read_seg(
        str(context["labels_ts"] / f"{subject_id}.nii.gz")
    )
    if tuple(semantic_loaded.shape[1:]) != tuple(semantic_native.shape):
        raise ValueError(f"semantic NIfTI shape drift for {subject_id}")
    if tuple(instances_loaded.shape[1:]) != tuple(instances_native.shape):
        raise ValueError(f"instance NIfTI shape drift for {subject_id}")
    label_geometry_exact = (
        tuple(label_loaded.shape[1:]) == tuple(semantic_native.shape)
        and _geometry_matches(label_properties, semantic_properties)
        and _geometry_matches(label_properties, instances_properties)
    )
    if not label_geometry_exact:
        raise ValueError(f"native NIfTI geometry drift for {subject_id}")

    record = {
        "schema_version": 1,
        "subject_id": subject_id,
        "checkpoint_sha256": context["checkpoint_sha256"],
        "evaluator_sha256": context["evaluator_sha256"],
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "semantic_path": str(semantic_path),
        "semantic_sha256": sha256(semantic_path),
        "instances_path": str(instances_path),
        "instances_sha256": sha256(instances_path),
        "native_shape": list(semantic_native.shape),
        "geometry": _geometry_signature(properties),
        "preprocessed_instance_count": preprocessed_count,
        "native_instance_count": native_count,
        "label_geometry_exact": label_geometry_exact,
        "reused": False,
    }
    atomic_write_json(record_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export frozen UMCTF test251 outputs to native-space NIfTI")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_output = args.evaluation_output.resolve()
    contract = json.loads((evaluation_output / "contract.json").read_text(encoding="utf-8"))
    summary = json.loads((evaluation_output / "summary.json").read_text(encoding="utf-8"))
    subjects = tuple(str(value) for value in contract["subject_ids"])
    if summary.get("status") != "complete" or summary.get("subject_ids") != sorted(subjects):
        raise ValueError("native export requires a complete aligned test251 summary")
    if int(summary.get("prediction_file_count", -1)) != len(subjects):
        raise ValueError("native export requires all saved prediction NPZ files")
    workers = max(1, min(int(args.workers), len(subjects)))
    initializer_args = (
        str(args.model_folder.resolve()),
        str(evaluation_output),
        str(Path(contract["source_root"]).resolve()),
        str(Path(contract["labels_ts"]).resolve()),
        str(contract["checkpoint_sha256"]),
        str(contract["evaluator_sha256"]),
        args.configuration,
    )
    if workers == 1:
        _initialize_worker(*initializer_args)
        records = [_export_subject(subject) for subject in subjects]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker, initargs=initializer_args
        ) as pool:
            records = list(pool.map(_export_subject, subjects))
    audit = validate_native_export_records(records, subjects)
    audit.update(
        {
            "schema_version": 1,
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "evaluator_sha256": contract["evaluator_sha256"],
            "source_root": contract["source_root"],
            "labels_ts": contract["labels_ts"],
            "workers": workers,
            "reused_count": sum(bool(record.get("reused")) for record in records),
            "semantic_directory": str(evaluation_output / "native_nifti" / "semantic"),
            "instances_directory": str(evaluation_output / "native_nifti" / "instances"),
            "records": records,
        }
    )
    atomic_write_json(evaluation_output / "native_nifti_export_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "records"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
