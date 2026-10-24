from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label

from nnunetv2.evaluation.microbt_pf_full203 import (
    _box_iou,
    atomic_write_json,
    dataset_average_precision,
)
from nnunetv2.inference.microbt_ctf_decode import decode_ctf_instances, instance_boxes
from nnunetv2.inference.microbt_ctf_predictor import MicroBTCTFPredictor, raw_logits_to_ctf
from nnunetv2.training.dataloading.microbt_pf_targets import deterministic_interior_center
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _region_target(semantic: np.ndarray, regions) -> np.ndarray:
    return np.stack(
        [np.isin(semantic, region if isinstance(region, (tuple, list)) else (region,)) for region in regions]
    )


def _dice(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    values = []
    for channel in range(target.shape[0]):
        p, t = predicted[channel], target[channel]
        denominator = int(p.sum()) + int(t.sum())
        values.append(1.0 if denominator == 0 else 2.0 * float((p & t).sum()) / denominator)
    return values


def _equivalent_diameter(mask: np.ndarray, spacing) -> float:
    volume = float(mask.sum()) * float(np.prod(spacing))
    return (6.0 * volume / math.pi) ** (1.0 / 3.0)


def _training_retention(model_folder: Path) -> dict:
    logs = sorted((model_folder / "fold_0").glob("training_log_*.txt"))
    if not logs:
        raise FileNotFoundError("training log is missing")
    text = "\n".join(path.read_text(errors="ignore") for path in logs)
    epoch = None
    dice = {}
    for line_text in text.splitlines():
        match = re.search(r"Epoch (\d+)\s*$", line_text)
        if match:
            epoch = int(match.group(1))
        if epoch is not None and "Pseudo dice [" in line_text:
            raw = line_text.split("Pseudo dice [", 1)[1].split("]", 1)[0]
            dice[epoch] = [float(value.strip()) for value in raw.split(",")]
    def median(lo, hi):
        return [float(np.median([dice[e][c] for e in range(lo, hi + 1)])) for c in range(3)]
    base, late = median(25, 29), median(55, 59)
    return {
        "base_25_29": base,
        "late_55_59": late,
        "retention": [late[i] / base[i] if base[i] else None for i in range(3)],
        "log_error_hits": len(re.findall(r"Traceback|RuntimeError|out of memory|nonfinite", text, flags=re.I)),
        "exact_h0_calls": sum(int(value) for value in re.findall(r'"exact_h0_train_calls":\s*(\d+)', text)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-mae-lt", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_path = args.model_folder / "fold_0" / args.checkpoint
    manifest = json.loads(args.manifest.read_text())
    case_ids = [str(case["subject_id"]) for case in manifest["cases"]]
    if len(case_ids) != 8:
        raise ValueError("the fixed CTF gate requires exactly eight cases")
    predictor = MicroBTCTFPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=True, device=torch.device("cuda"),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model_folder), use_folds=(0,), checkpoint_name=args.checkpoint
    )
    predictor.network.to("cuda").eval()
    from nnunetv2.paths import nnUNet_preprocessed
    preprocessed = Path(nnUNet_preprocessed) / predictor.plans_manager.dataset_name
    preprocessed /= predictor.configuration_manager.data_identifier
    dataset_class = infer_dataset_class(str(preprocessed))
    dataset = dataset_class(str(preprocessed), case_ids, folder_with_segs_from_previous_stage=None)
    spacing = tuple(float(v) for v in predictor.configuration_manager.spacing)
    regions = predictor.label_manager.foreground_regions
    records = []
    all_gt_distances = []
    all_micro_distances = []
    all_centroid_inside = []
    all_tightness = []
    dice_rows = []
    count_errors = []
    started = time.perf_counter()
    for subject_id in case_ids:
        case_started = time.perf_counter()
        data, segmentation, previous_stage, _ = dataset.load_case(subject_id)
        if previous_stage is not None:
            raise RuntimeError("CTF gate does not support cascaded data")
        semantic = np.asarray(segmentation[0])
        raw = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(np.asarray(data, dtype=np.float32))
        ).cpu()
        output = raw_logits_to_ctf(raw)
        probabilities = output["region_probabilities"].numpy()
        target_regions = _region_target(semantic, regions)
        dice_rows.append(_dice(probabilities > 0.5, target_regions))
        decoded = decode_ctf_instances(
            probabilities[0], output["transport"].numpy(),
            offset_scale_zyx=(56.0, 80.0, 64.0), support_threshold=0.10,
            min_vote_mass=0.25, nms_radius=2, max_centers=64,
        )
        gt_labels, gt_count = label(semantic > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        count_errors.append(abs(decoded.count - int(gt_count)))
        pred_boxes = instance_boxes(decoded.instance_labels)
        gt_boxes = []
        targets = []
        pred_centers = np.asarray(decoded.centers_zyx, dtype=np.float64).reshape(-1, 3)
        for instance_id in range(1, int(gt_count) + 1):
            mask = gt_labels == instance_id
            center = np.asarray(deterministic_interior_center(mask, spacing), dtype=np.float64)
            diameter = _equivalent_diameter(mask, spacing)
            if len(pred_centers):
                distances = np.linalg.norm((pred_centers - center[None]) * np.asarray(spacing), axis=1)
                nearest = float(distances.min())
                inside = float(any(gt_labels[tuple(int(v) for v in c)] == instance_id for c in pred_centers))
            else:
                nearest, inside = float("inf"), 0.0
            all_gt_distances.append(nearest)
            if diameter <= 3.0:
                all_micro_distances.append(nearest)
            all_centroid_inside.append(inside)
            coords = np.argwhere(mask)
            box = tuple(int(v) for v in np.concatenate((coords.min(0), coords.max(0) + 1)))
            gt_boxes.append(box)
            targets.append({"box_zyx": box, "equivalent_diameter_mm": diameter})
            all_tightness.append(max((_box_iou(box, pb) for pb in pred_boxes), default=0.0))
        detections = [
            {"box_zyx": box, "score": float(decoded.center_scores[index])}
            for index, box in enumerate(pred_boxes)
        ]
        records.append({
            "subject_id": subject_id, "detections": detections, "targets": targets,
            "elapsed_seconds": time.perf_counter() - case_started, "metrics": {},
        })
        torch.cuda.empty_cache()

    training = _training_retention(args.model_folder)
    metrics = {
        "case_count": len(records),
        "object_count": int(sum(len(record["targets"]) for record in records)),
        "dice": np.mean(np.asarray(dice_rows), axis=0).tolist(),
        "count_mae": float(np.mean(count_errors)),
        "ap50": dataset_average_precision(records, 0.50),
        "ap75": dataset_average_precision(records, 0.75),
        "proposal_recall_5mm": float(np.mean(np.asarray(all_gt_distances) <= 5.0)),
        "micro_recall_5mm_le3mm": float(np.mean(np.asarray(all_micro_distances) <= 5.0)) if all_micro_distances else None,
        "centroid_inside": float(np.mean(all_centroid_inside)),
        "box_tightness_mean_best_iou": float(np.mean(all_tightness)),
        "runtime_seconds": time.perf_counter() - started,
    }
    gates = {
        "dice_retention_all_ge_0_70": all(value is not None and value >= 0.70 for value in training["retention"]),
        "no_log_errors": training["log_error_hits"] == 0,
        "h0_calls_zero": training["exact_h0_calls"] == 0,
        "fixed_objects_52": metrics["object_count"] == 52,
        "proposal_recall_5mm_ge_0_60": metrics["proposal_recall_5mm"] >= 0.60,
        "micro_recall_5mm_ge_0_40": metrics["micro_recall_5mm_le3mm"] is not None and metrics["micro_recall_5mm_le3mm"] >= 0.40,
        "ap50_ge_0_05": metrics["ap50"] >= 0.05,
        "centroid_inside_ge_0_30": metrics["centroid_inside"] >= 0.30,
    }
    if args.count_mae_lt is not None:
        gates["count_mae_lt_threshold"] = metrics["count_mae"] < args.count_mae_lt
    payload = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "full203_unlocked": bool(all(gates.values())),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training": training,
        "metrics": metrics,
        "gates": gates,
        "records": records,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "full203_unlocked", "metrics", "gates")}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
