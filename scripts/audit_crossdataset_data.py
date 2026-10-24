from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.crossdataset_case_files import resolve_case_files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subjects(folder: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name.removesuffix(".nii.gz") for path in folder.glob("*.nii.gz")))


def _resolve_preprocessed_dir(preprocessed_root: Path, plan_path: Path) -> Path:
    if not plan_path.is_file():
        return preprocessed_root / "nnUNetResEncUNetMPlans_3d_fullres"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    configuration = plan.get("configurations", {}).get("3d_fullres", {})
    data_identifier = configuration.get("data_identifier")
    if not isinstance(data_identifier, str) or not data_identifier:
        raise ValueError("ResEnc-M 3d_fullres plan is missing data_identifier")
    return preprocessed_root / data_identifier


def _count_preprocessed_cases(processed: Path) -> tuple[int, str | None]:
    npz_cases = {path.name.removesuffix(".npz") for path in processed.glob("*.npz")}
    if npz_cases:
        return len(npz_cases), "npz"
    data_cases = {
        path.name.removesuffix(".b2nd")
        for path in processed.glob("*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    }
    if not data_cases:
        return 0, None
    seg_cases = {
        path.name.removesuffix("_seg.b2nd")
        for path in processed.glob("*_seg.b2nd")
    }
    property_cases = {path.name.removesuffix(".pkl") for path in processed.glob("*.pkl")}
    if data_cases != seg_cases or data_cases != property_cases:
        raise ValueError("b2nd preprocessing data/seg/properties coverage is inconsistent")
    return len(data_cases), "b2nd"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-preprocessed", action="store_true")
    args = parser.parse_args()

    raw = args.data_root / "nnUNet_raw" / args.dataset_name
    pre = args.data_root / "nnUNet_preprocessed" / args.dataset_name
    dataset_json_path = raw / "dataset.json"
    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    channel_count = len(dataset_json["channel_names"])
    train = _subjects(raw / "labelsTr")
    test = _subjects(raw / "labelsTs")
    if not train or not test or set(train) & set(test):
        raise ValueError("train/test labels must be nonempty, unique, and disjoint")
    if int(dataset_json["numTraining"]) != len(train):
        raise ValueError("dataset.json numTraining does not match labelsTr")
    for split, subjects in (("imagesTr", train), ("imagesTs", test)):
        for subject in subjects:
            files = resolve_case_files(raw / split, subject)
            if len(files) != channel_count:
                raise ValueError(f"channel count drift for {split}/{subject}")

    plan = pre / "nnUNetResEncUNetMPlans.json"
    processed = _resolve_preprocessed_dir(pre, plan)
    preprocessed_case_count, preprocessed_storage_format = _count_preprocessed_cases(processed)
    if args.require_preprocessed:
        if not plan.is_file() or not processed.is_dir():
            raise FileNotFoundError("ResEnc-M plans/preprocessing are incomplete")
        if preprocessed_case_count != len(train):
            raise ValueError("preprocessed case coverage does not match labelsTr")

    case_manifest = {
        "train": list(train),
        "test": list(test),
        "channels": channel_count,
    }
    payload = {
        "schema_version": 1,
        "status": "passed",
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "channel_count": channel_count,
        "training_case_count": len(train),
        "test_case_count": len(test),
        "train_test_overlap": 0,
        "dataset_json_sha256": _sha256(dataset_json_path),
        "case_manifest_sha256": hashlib.sha256(
            json.dumps(case_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "resencm_plan_sha256": _sha256(plan) if plan.is_file() else None,
        "preprocessed_case_count": preprocessed_case_count,
        "preprocessed_storage_format": preprocessed_storage_format,
        "preprocessed_npz_count": len(tuple(processed.glob("*.npz"))) if processed.is_dir() else 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
