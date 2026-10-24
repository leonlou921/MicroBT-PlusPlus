from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label

from nnunetv2.evaluation.microbt_ctf_decomposition import (
    aggregate_arm_rows,
    endpoint_instance_diagnostics,
    make_support_transport_matrix,
)
from nnunetv2.evaluation.microbt_pf_full203 import atomic_write_json
from nnunetv2.inference.microbt_ctf_decode import decode_ctf_instances
from nnunetv2.inference.microbt_ctf_predictor import MicroBTCTFPredictor, raw_logits_to_ctf
from nnunetv2.training.dataloading.microbt_pf_targets import deterministic_interior_center
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


OFFSET_SCALE = (56.0, 80.0, 64.0)


def _diameter(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    volume = float(mask.sum()) * float(np.prod(spacing))
    return (6.0 * volume / math.pi) ** (1.0 / 3.0)


def _summarize_distances(distances: list[float], micro: list[bool]) -> dict[str, float | None]:
    values = np.asarray(distances, dtype=np.float64)
    small = values[np.asarray(micro, dtype=bool)]
    return {
        "proposal_recall_2mm": float(np.mean(values <= 2.0)),
        "proposal_recall_5mm": float(np.mean(values <= 5.0)),
        "micro_recall_2mm": float(np.mean(small <= 2.0)) if len(small) else None,
        "micro_recall_5mm": float(np.mean(small <= 5.0)) if len(small) else None,
        "median_nearest_center_mm": float(np.median(values)),
        "micro_median_nearest_center_mm": float(np.median(small)) if len(small) else None,
    }


def _evaluate_arm(
    support: np.ndarray,
    transport: np.ndarray,
    gt_labels: np.ndarray,
    centers: tuple[tuple[int, int, int], ...],
    micro_flags: list[bool],
    spacing: tuple[float, float, float],
) -> dict:
    decoded = decode_ctf_instances(
        support,
        transport,
        offset_scale_zyx=OFFSET_SCALE,
        support_threshold=0.10,
        min_vote_mass=0.25,
        nms_radius=2,
        max_centers=64,
    )
    predicted = np.asarray(decoded.centers_zyx, dtype=np.float64).reshape(-1, 3)
    distances: list[float] = []
    inside: list[float] = []
    for instance_id, center in enumerate(centers, start=1):
        if len(predicted):
            physical = (predicted - np.asarray(center)[None]) * np.asarray(spacing)[None]
            distances.append(float(np.linalg.norm(physical, axis=1).min()))
            inside.append(float(any(gt_labels[tuple(int(v) for v in point)] == instance_id for point in predicted)))
        else:
            distances.append(float("inf"))
            inside.append(0.0)
    return {
        "predicted_count": decoded.count,
        "absolute_count_error": abs(decoded.count - len(centers)),
        "centroid_inside_hits": int(sum(inside)),
        "proposal_hits_2mm": int(sum(value <= 2.0 for value in distances)),
        "proposal_hits_5mm": int(sum(value <= 5.0 for value in distances)),
        "micro_hits_2mm": int(sum(value <= 2.0 for value, is_micro in zip(distances, micro_flags, strict=True) if is_micro)),
        "micro_hits_5mm": int(sum(value <= 5.0 for value, is_micro in zip(distances, micro_flags, strict=True) if is_micro)),
        "nearest_center_sum_mm": float(sum(distances)),
        "micro_nearest_center_sum_mm": float(sum(value for value, is_micro in zip(distances, micro_flags, strict=True) if is_micro)),
        "centroid_inside": float(np.mean(inside)),
        **_summarize_distances(distances, micro_flags),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    case_ids = [str(case["subject_id"]) for case in manifest["cases"]]
    if len(case_ids) != 8:
        raise ValueError("CTF decomposition requires the frozen eight-case manifest")

    predictor = MicroBTCTFPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
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
    arm_rows: dict[str, list[dict]] = {}
    endpoint_rows: list[dict] = []
    case_rows: list[dict] = []
    started = time.perf_counter()
    for subject_id in case_ids:
        data, segmentation, previous_stage, _ = dataset.load_case(subject_id)
        if previous_stage is not None:
            raise RuntimeError("CTF decomposition does not support cascades")
        semantic = np.asarray(segmentation[0])
        raw = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(np.asarray(data, dtype=np.float32))
        ).cpu()
        output = raw_logits_to_ctf(raw)
        predicted_support = output["region_probabilities"][0].numpy()
        predicted_transport = output["transport"].numpy()
        gt_labels, gt_count = label(semantic > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
        centers = tuple(
            deterministic_interior_center(gt_labels == instance_id, spacing)
            for instance_id in range(1, int(gt_count) + 1)
        )
        diameters = [
            _diameter(gt_labels == instance_id, spacing)
            for instance_id in range(1, int(gt_count) + 1)
        ]
        micro_flags = [diameter <= 3.0 for diameter in diameters]
        matrix = make_support_transport_matrix(
            predicted_support, predicted_transport, gt_labels, centers, OFFSET_SCALE
        )
        per_case = {"subject_id": subject_id, "gt_count": int(gt_count), "micro_count": int(sum(micro_flags)), "arms": {}}
        for arm, (support, transport) in matrix.items():
            metrics = _evaluate_arm(support, transport, gt_labels, centers, micro_flags, spacing)
            metrics["gt_count"] = int(gt_count)
            metrics["micro_count"] = int(sum(micro_flags))
            arm_rows.setdefault(arm, []).append(metrics)
            per_case["arms"][arm] = metrics
        rows = endpoint_instance_diagnostics(
            gt_labels, centers, predicted_support, predicted_transport, OFFSET_SCALE, spacing
        )
        for row, diameter in zip(rows, diameters, strict=True):
            row.update({"subject_id": subject_id, "diameter_mm": diameter, "is_micro": diameter <= 3.0})
        endpoint_rows.extend(rows)
        case_rows.append(per_case)
        torch.cuda.empty_cache()

    summary = {}
    for arm, rows in arm_rows.items():
        summary[arm] = aggregate_arm_rows(rows)
    endpoint_summary = {}
    for name, selected in (
        ("all", endpoint_rows),
        ("micro_le3mm", [row for row in endpoint_rows if row["is_micro"]]),
        ("larger_gt3mm", [row for row in endpoint_rows if not row["is_micro"]]),
    ):
        endpoint_summary[name] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "support_mean", "support_ge_0_1_rate", "endpoint_error_mean_mm",
                "endpoint_error_median_mm", "endpoint_within_0_5_vox_rate",
                "rounded_endpoint_mode_fraction",
            )
        }
        endpoint_summary[name]["object_count"] = len(selected)
    payload = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "case_count": len(case_ids),
        "summary": summary,
        "endpoint_summary": endpoint_summary,
        "cases": case_rows,
        "objects": endpoint_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"summary": summary, "endpoint_summary": endpoint_summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
