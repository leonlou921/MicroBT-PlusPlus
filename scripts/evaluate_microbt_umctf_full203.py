from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import label

from nnunetv2.evaluation.microbt_pf_full203 import (
    _box_iou,
    atomic_write_json,
    dataset_average_precision,
)
from nnunetv2.inference.microbt_umctf_decode import decode_umctf_instances
from nnunetv2.inference.microbt_umctf_predictor import (
    MicroBTUnitMassPredictor,
    raw_logits_to_umctf,
)
from nnunetv2.training.dataloading.microbt_pf_targets import (
    deterministic_interior_center,
)
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from scripts.evaluate_microbt_umctf_stage_gate import scored_instance_boxes


ARM_NAMES = (
    "pred_mass_pred_transport",
    "gt_mass_pred_transport",
    "pred_mass_gt_transport",
    "gt_mass_gt_transport",
)
DECODER_CONTRACT = {
    "tile_step_size": 0.5,
    "use_gaussian": True,
    "use_mirroring": False,
    "region_threshold": 0.5,
    "support_threshold": 0.10,
    "density_threshold": 1e-6,
    "min_atom_mass": 0.25,
    "nms_radius": 2,
    "max_centers": 64,
    "offset_scale_zyx": [56.0, 80.0, 64.0],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluator_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    relative = (
        "scripts/evaluate_microbt_umctf_full203.py",
        "scripts/evaluate_microbt_umctf_stage_gate.py",
        "nnunetv2/evaluation/microbt_pf_full203.py",
        "nnunetv2/inference/microbt_umctf_decode.py",
        "nnunetv2/inference/microbt_umctf_predictor.py",
        "nnunetv2/nets/microbt_umctf/model.py",
        "nnunetv2/training/dataloading/microbt_pf_targets.py",
    )
    digest = hashlib.sha256()
    for item in relative:
        path = root / item
        digest.update(item.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_validation_subjects(
    split_path: str | Path,
    fold: int,
    *,
    expected_count: int = 203,
) -> tuple[str, ...]:
    splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if not isinstance(splits, list) or fold < 0 or fold >= len(splits):
        raise ValueError("requested fold is absent from split file")
    subjects = tuple(str(value).strip() for value in splits[fold].get("val", ()))
    if len(subjects) != expected_count or any(not value for value in subjects):
        raise ValueError(
            f"fold validation must contain exactly {expected_count} nonblank subjects"
        )
    if len(set(subjects)) != len(subjects):
        raise ValueError("fold validation subject IDs must be unique")
    return tuple(sorted(subjects))


def _distance(value: Any) -> float:
    if value is None:
        return float("inf")
    result = float(value)
    return result if math.isfinite(result) else float("inf")


def validate_case_record(
    record: Mapping[str, Any],
    subject_id: str,
    checkpoint_sha256: str,
    evaluator_sha256: str,
) -> None:
    if int(record.get("schema_version", -1)) != 1:
        raise ValueError("case record schema mismatch")
    if str(record.get("subject_id")) != str(subject_id):
        raise ValueError("case record subject mismatch")
    if str(record.get("checkpoint_sha256")) != str(checkpoint_sha256):
        raise ValueError("case record checkpoint mismatch")
    if str(record.get("evaluator_sha256")) != str(evaluator_sha256):
        raise ValueError("case record evaluator mismatch")
    elapsed = float(record.get("elapsed_seconds", -1))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("case record elapsed time is invalid")
    dice = record.get("dice")
    if not isinstance(dice, list) or len(dice) != 3:
        raise ValueError("case record dice payload is invalid")
    if not all(math.isfinite(float(value)) for value in dice):
        raise ValueError("case record dice contains nonfinite values")
    gt_count = int(record.get("gt_count", -1))
    predicted_count = int(record.get("predicted_count", -1))
    if gt_count < 0 or predicted_count < 0:
        raise ValueError("case record count payload is invalid")
    targets = record.get("targets")
    objects = record.get("objects")
    detections = record.get("detections")
    if not isinstance(targets, list) or len(targets) != gt_count:
        raise ValueError("case record targets do not match gt_count")
    if not isinstance(objects, list) or len(objects) != gt_count:
        raise ValueError("case record objects do not match gt_count")
    if not isinstance(detections, list):
        raise ValueError("case record detections are invalid")
    causal = record.get("causal")
    if not isinstance(causal, Mapping) or set(causal) != set(ARM_NAMES):
        raise ValueError("case record causal arms are invalid")
    for arm in ARM_NAMES:
        arm_payload = causal[arm]
        if int(arm_payload.get("predicted_count", -1)) < 0:
            raise ValueError(f"case record causal count is invalid: {arm}")
        if len(arm_payload.get("objects", ())) != gt_count:
            raise ValueError(f"case record causal objects are invalid: {arm}")


def summarize_case_records(
    records: Sequence[Mapping[str, Any]],
    expected_subjects: Sequence[str],
    *,
    expected_count: int,
    checkpoint_sha256: str,
    evaluator_sha256: str,
) -> dict[str, Any]:
    if len(records) != expected_count:
        raise ValueError(f"exactly {expected_count} case records are required")
    expected = tuple(sorted(str(value) for value in expected_subjects))
    if len(expected) != expected_count or len(set(expected)) != expected_count:
        raise ValueError("expected subjects are not exact and unique")
    by_subject = {str(record.get("subject_id")): record for record in records}
    if len(by_subject) != expected_count or tuple(sorted(by_subject)) != expected:
        raise ValueError("case records do not align with expected subjects")
    ordered = [by_subject[subject] for subject in expected]
    for subject, record in zip(expected, ordered, strict=True):
        validate_case_record(record, subject, checkpoint_sha256, evaluator_sha256)

    objects = [item for record in ordered for item in record["objects"]]
    distances = np.asarray(
        [_distance(item["nearest_distance_mm"]) for item in objects], dtype=np.float64
    )
    diameters = np.asarray(
        [float(item["equivalent_diameter_mm"]) for item in objects], dtype=np.float64
    )
    micro = diameters <= 3.0
    dice = np.asarray([record["dice"] for record in ordered], dtype=np.float64)
    gt_counts = np.asarray([record["gt_count"] for record in ordered], dtype=np.int64)
    predicted_counts = np.asarray(
        [record["predicted_count"] for record in ordered], dtype=np.int64
    )
    input_masses = np.asarray([record["input_mass"] for record in ordered], dtype=np.float64)
    atom_masses = np.asarray([record["atom_mass"] for record in ordered], dtype=np.float64)

    def _recall(mask: np.ndarray) -> float | None:
        return float(np.mean(distances[mask] <= 5.0)) if np.any(mask) else None

    metrics = {
        "case_count": len(ordered),
        "object_count": len(objects),
        "micro_object_count_le3mm": int(np.sum(micro)),
        "predicted_count_total": int(np.sum(predicted_counts)),
        "dice": np.mean(dice, axis=0).tolist(),
        "count_mae": float(np.mean(np.abs(predicted_counts - gt_counts))),
        "ap50": dataset_average_precision(ordered, 0.50),
        "ap75": dataset_average_precision(ordered, 0.75),
        "proposal_recall_5mm": float(np.mean(distances <= 5.0)) if len(distances) else None,
        "micro_recall_5mm_le3mm": _recall(micro),
        "recall_5mm_3to5mm": _recall((diameters > 3.0) & (diameters <= 5.0)),
        "recall_5mm_gt5mm": _recall(diameters > 5.0),
        "centroid_inside": float(np.mean([item["centroid_inside"] for item in objects]))
        if objects
        else None,
        "box_tightness_mean_best_iou": float(
            np.mean([item["box_tightness"] for item in objects])
        )
        if objects
        else None,
        "duplicate_extras_per_gt_5mm": float(
            np.mean([item["duplicate_extras"] for item in objects])
        )
        if objects
        else None,
        "predicted_input_mass_mean_per_case": float(np.mean(input_masses)),
        "atom_mass_mean_per_case": float(np.mean(atom_masses)),
        "mass_conservation_abs_error_mean": float(
            np.mean(np.abs(input_masses - atom_masses))
        ),
        "empty_case_count": int(np.sum(predicted_counts == 0)),
        "nonfinite_case_count": int(
            sum(not bool(record.get("numeric_finite", True)) for record in ordered)
        ),
        "runtime_seconds": float(sum(float(record["elapsed_seconds"]) for record in ordered)),
    }

    causal_matrix: dict[str, Any] = {}
    for arm in ARM_NAMES:
        arm_counts = np.asarray(
            [record["causal"][arm]["predicted_count"] for record in ordered],
            dtype=np.int64,
        )
        arm_objects = [
            item for record in ordered for item in record["causal"][arm]["objects"]
        ]
        arm_distances = np.asarray(
            [_distance(item["nearest_distance_mm"]) for item in arm_objects],
            dtype=np.float64,
        )
        arm_diameters = np.asarray(
            [float(item["equivalent_diameter_mm"]) for item in arm_objects],
            dtype=np.float64,
        )
        arm_micro = arm_diameters <= 3.0
        causal_matrix[arm] = {
            "proposal_recall_5mm": float(np.mean(arm_distances <= 5.0))
            if len(arm_distances)
            else None,
            "micro_recall_5mm_le3mm": float(np.mean(arm_distances[arm_micro] <= 5.0))
            if np.any(arm_micro)
            else None,
            "count_mae": float(np.mean(np.abs(arm_counts - gt_counts))),
            "predicted_count_total": int(np.sum(arm_counts)),
        }

    return {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "metrics": metrics,
        "causal_matrix": causal_matrix,
        "subject_ids": list(expected),
    }


def _region_target(semantic: np.ndarray, regions: Sequence[Any]) -> np.ndarray:
    return np.stack(
        [
            np.isin(
                semantic,
                region if isinstance(region, (tuple, list)) else (region,),
            )
            for region in regions
        ]
    )


def _dice(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    values = []
    for channel in range(target.shape[0]):
        p, t = predicted[channel], target[channel]
        denominator = int(p.sum()) + int(t.sum())
        values.append(
            1.0 if denominator == 0 else 2.0 * float((p & t).sum()) / denominator
        )
    return values


def _equivalent_diameter(mask: np.ndarray, spacing: Sequence[float]) -> float:
    volume = float(mask.sum()) * float(np.prod(spacing))
    return (6.0 * volume / math.pi) ** (1.0 / 3.0)


def _finite_distance(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def evaluate_case(
    predictor: MicroBTUnitMassPredictor,
    dataset: Any,
    subject_id: str,
    *,
    spacing: Sequence[float],
    regions: Sequence[Any],
    checkpoint_sha256: str,
    evaluator_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    data, segmentation, previous_stage, _ = dataset.load_case(subject_id)
    if previous_stage is not None:
        raise RuntimeError("full203 evaluation does not support cascaded data")
    semantic = np.asarray(segmentation[0])
    raw = predictor.predict_logits_from_preprocessed_data(
        torch.from_numpy(np.asarray(data, dtype=np.float32))
    ).cpu()
    output = raw_logits_to_umctf(raw)
    probabilities = output["region_probabilities"].numpy()
    predicted_density = output["object_density"].numpy()[0]
    predicted_transport = output["transport"].numpy()
    numeric_finite = bool(
        np.isfinite(probabilities).all()
        and np.isfinite(predicted_density).all()
        and np.isfinite(predicted_transport).all()
    )
    if not numeric_finite:
        raise RuntimeError(f"nonfinite UMCTF output for {subject_id}")

    decoded = decode_umctf_instances(
        probabilities[0],
        predicted_density,
        predicted_transport,
        offset_scale_zyx=tuple(DECODER_CONTRACT["offset_scale_zyx"]),
        support_threshold=DECODER_CONTRACT["support_threshold"],
        density_threshold=DECODER_CONTRACT["density_threshold"],
        min_atom_mass=DECODER_CONTRACT["min_atom_mass"],
        nms_radius=DECODER_CONTRACT["nms_radius"],
        max_centers=DECODER_CONTRACT["max_centers"],
    )
    gt_labels, gt_count = label(
        semantic > 0, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    scored_instances = scored_instance_boxes(
        decoded.instance_labels, decoded.center_scores
    )
    pred_boxes = tuple(item[0] for item in scored_instances)
    pred_centers = np.asarray(decoded.centers_zyx, dtype=np.float64).reshape(-1, 3)
    gt_density = np.zeros_like(predicted_density, dtype=np.float32)
    gt_transport = np.zeros_like(predicted_transport, dtype=np.float32)
    targets: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    gt_centers: list[np.ndarray] = []
    gt_diameters: list[float] = []

    for instance_id in range(1, int(gt_count) + 1):
        mask = gt_labels == instance_id
        center = np.asarray(
            deterministic_interior_center(mask, spacing), dtype=np.float64
        )
        diameter = _equivalent_diameter(mask, spacing)
        gt_centers.append(center)
        gt_diameters.append(diameter)
        gt_density[mask] = 1.0 / float(mask.sum())
        coordinates = np.argwhere(mask)
        gt_transport[:, mask] = (
            (center[None] - coordinates)
            / np.asarray(DECODER_CONTRACT["offset_scale_zyx"])[None]
        ).T
        if len(pred_centers):
            center_distances = np.linalg.norm(
                (pred_centers - center[None]) * np.asarray(spacing), axis=1
            )
            nearest = float(center_distances.min())
            duplicate_extras = max(0, int(np.sum(center_distances <= 5.0)) - 1)
            centroid_inside = float(
                any(
                    gt_labels[tuple(int(value) for value in candidate)] == instance_id
                    for candidate in pred_centers
                )
            )
        else:
            nearest = float("inf")
            duplicate_extras = 0
            centroid_inside = 0.0
        box = [
            int(value)
            for value in np.concatenate((coordinates.min(0), coordinates.max(0) + 1))
        ]
        targets.append({"box_zyx": box, "equivalent_diameter_mm": diameter})
        objects.append(
            {
                "nearest_distance_mm": _finite_distance(nearest),
                "equivalent_diameter_mm": diameter,
                "centroid_inside": centroid_inside,
                "box_tightness": max(
                    (_box_iou(box, predicted_box) for predicted_box in pred_boxes),
                    default=0.0,
                ),
                "duplicate_extras": duplicate_extras,
            }
        )

    causal: dict[str, Any] = {}
    for arm_name, arm_density, arm_transport in (
        ("pred_mass_pred_transport", predicted_density, predicted_transport),
        ("gt_mass_pred_transport", gt_density, predicted_transport),
        ("pred_mass_gt_transport", predicted_density, gt_transport),
        ("gt_mass_gt_transport", gt_density, gt_transport),
    ):
        arm = decode_umctf_instances(
            probabilities[0],
            arm_density,
            arm_transport,
            offset_scale_zyx=tuple(DECODER_CONTRACT["offset_scale_zyx"]),
            support_threshold=DECODER_CONTRACT["support_threshold"],
            density_threshold=DECODER_CONTRACT["density_threshold"],
            min_atom_mass=DECODER_CONTRACT["min_atom_mass"],
            nms_radius=DECODER_CONTRACT["nms_radius"],
            max_centers=DECODER_CONTRACT["max_centers"],
        )
        arm_centers = np.asarray(arm.centers_zyx, dtype=np.float64).reshape(-1, 3)
        arm_objects = []
        for center, diameter in zip(gt_centers, gt_diameters, strict=True):
            nearest = (
                float(
                    np.linalg.norm(
                        (arm_centers - center[None]) * np.asarray(spacing), axis=1
                    ).min()
                )
                if len(arm_centers)
                else float("inf")
            )
            arm_objects.append(
                {
                    "nearest_distance_mm": _finite_distance(nearest),
                    "equivalent_diameter_mm": diameter,
                }
            )
        causal[arm_name] = {
            "predicted_count": int(arm.count),
            "objects": arm_objects,
        }

    target_regions = _region_target(semantic, regions)
    record = {
        "schema_version": 1,
        "subject_id": subject_id,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "numeric_finite": numeric_finite,
        "dice": _dice(probabilities > DECODER_CONTRACT["region_threshold"], target_regions),
        "gt_count": int(gt_count),
        "predicted_count": int(decoded.count),
        "input_mass": float(decoded.total_input_mass),
        "atom_mass": float(decoded.total_atom_mass),
        "detections": [
            {"box_zyx": [int(value) for value in box], "score": float(score)}
            for box, score in scored_instances
        ],
        "targets": targets,
        "objects": objects,
        "causal": causal,
    }
    validate_case_record(record, subject_id, checkpoint_sha256, evaluator_sha256)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable frozen full-volume UMCTF evaluator"
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=203)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = args.model_folder / f"fold_{args.fold}" / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_checkpoint_sha256 = _sha256(checkpoint_path)
    if actual_checkpoint_sha256 != args.checkpoint_sha256:
        raise RuntimeError("checkpoint hash drift")
    evaluator_sha256 = _evaluator_sha256()
    subjects = load_validation_subjects(
        args.split, args.fold, expected_count=args.expected_count
    )
    if any("/" in subject or "\\" in subject for subject in subjects):
        raise ValueError("subject IDs must be safe filenames")

    output = args.output_dir.resolve()
    case_folder = output / "cases"
    case_folder.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "split_sha256": _sha256(args.split),
        "fold": args.fold,
        "expected_count": args.expected_count,
        "subject_ids": list(subjects),
        "decoder_contract": DECODER_CONTRACT,
    }
    contract_path = output / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("existing full203 contract drift")
    else:
        atomic_write_json(contract_path, contract)

    predictor = MicroBTUnitMassPredictor(
        tile_step_size=DECODER_CONTRACT["tile_step_size"],
        use_gaussian=DECODER_CONTRACT["use_gaussian"],
        use_mirroring=DECODER_CONTRACT["use_mirroring"],
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_folder),
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint,
    )
    predictor.network.to("cuda").eval()
    from nnunetv2.paths import nnUNet_preprocessed

    preprocessed = Path(nnUNet_preprocessed) / predictor.plans_manager.dataset_name
    preprocessed /= predictor.configuration_manager.data_identifier
    dataset_class = infer_dataset_class(str(preprocessed))
    dataset = dataset_class(
        str(preprocessed), list(subjects), folder_with_segs_from_previous_stage=None
    )
    spacing = tuple(float(value) for value in predictor.configuration_manager.spacing)
    regions = predictor.label_manager.foreground_regions

    records = []
    reused = 0
    for index, subject_id in enumerate(subjects, start=1):
        case_path = case_folder / f"{subject_id}.json"
        record = None
        if case_path.is_file():
            try:
                candidate = json.loads(case_path.read_text(encoding="utf-8"))
                validate_case_record(
                    candidate,
                    subject_id,
                    actual_checkpoint_sha256,
                    evaluator_sha256,
                )
                record = candidate
                reused += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                record = None
        if record is None:
            record = evaluate_case(
                predictor,
                dataset,
                subject_id,
                spacing=spacing,
                regions=regions,
                checkpoint_sha256=actual_checkpoint_sha256,
                evaluator_sha256=evaluator_sha256,
            )
            atomic_write_json(case_path, record)
        records.append(record)
        atomic_write_json(
            output / "progress.json",
            {
                "schema_version": 1,
                "status": "running" if index < len(subjects) else "aggregating",
                "completed": index,
                "expected": len(subjects),
                "reused": reused,
                "last_subject_id": subject_id,
                "checkpoint_sha256": actual_checkpoint_sha256,
                "evaluator_sha256": evaluator_sha256,
            },
        )
        print(
            json.dumps(
                {
                    "completed": index,
                    "expected": len(subjects),
                    "subject_id": subject_id,
                    "elapsed_seconds": record["elapsed_seconds"],
                    "reused": record is not None and case_path.is_file(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.cuda.empty_cache()

    summary = summarize_case_records(
        records,
        subjects,
        expected_count=args.expected_count,
        checkpoint_sha256=actual_checkpoint_sha256,
        evaluator_sha256=evaluator_sha256,
    )
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        output / "progress.json",
        {
            "schema_version": 1,
            "status": "complete",
            "completed": len(subjects),
            "expected": len(subjects),
            "reused": reused,
            "checkpoint_sha256": actual_checkpoint_sha256,
            "evaluator_sha256": evaluator_sha256,
        },
    )
    print(json.dumps({"status": "complete", **summary["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

