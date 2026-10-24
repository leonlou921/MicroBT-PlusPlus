from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nnunetv2.evaluation.microbt_pf_full203 import atomic_write_json
from nnunetv2.inference.microbt_umctf_decode import decode_umctf_instances
from nnunetv2.inference.microbt_umctf_predictor import (
    MicroBTUnitMassPredictor,
    raw_logits_to_umctf,
)
from scripts.evaluate_microbt_umctf_full203 import (
    DECODER_CONTRACT,
    _evaluator_sha256,
    _sha256,
    evaluate_case,
    summarize_case_records,
    validate_case_record,
)


MODALITY_SUFFIXES = ("t1c", "t1n", "t2f", "t2w")


def load_test_subjects(labels_ts: str | Path, *, expected_count: int = 251) -> tuple[str, ...]:
    subjects = tuple(
        sorted(path.name.removesuffix(".nii.gz") for path in Path(labels_ts).glob("*.nii.gz"))
    )
    if len(subjects) != expected_count or len(set(subjects)) != expected_count:
        raise ValueError(f"test set must contain exactly {expected_count} unique labels")
    if any(not value or "/" in value or "\\" in value for value in subjects):
        raise ValueError("test subject IDs must be nonblank safe filenames")
    return subjects


def select_test_subjects(
    labels_ts: str | Path,
    *,
    expected_count: int,
    total_expected_count: int = 251,
    subject_ids_json: str | Path | None = None,
) -> tuple[str, ...]:
    all_subjects = load_test_subjects(labels_ts, expected_count=total_expected_count)
    if subject_ids_json is None:
        if expected_count != total_expected_count:
            raise ValueError("unsharded evaluation must cover the complete test set")
        return all_subjects
    values = json.loads(Path(subject_ids_json).read_text(encoding="utf-8"))
    subjects = tuple(str(value) for value in values)
    if len(subjects) != expected_count or len(subjects) != len(set(subjects)):
        raise ValueError("test shard must contain the expected unique subjects")
    if not set(subjects).issubset(all_subjects):
        raise ValueError("test shard contains subjects outside labelsTs")
    return subjects


def resolve_met_case_files(source_root: str | Path, subject_id: str) -> tuple[Path, ...]:
    if not subject_id.startswith("EXAMPLE__"):
        raise ValueError(f"unexpected MET subject id: {subject_id}")
    source_root = Path(source_root)
    flat = tuple(source_root / f"{subject_id}_{index:04d}.nii.gz" for index in range(4))
    if all(path.is_file() for path in flat):
        return flat
    native = subject_id.removeprefix("EXAMPLE__")
    folder = source_root / native
    files = tuple(folder / f"{native}-{suffix}.nii.gz" for suffix in MODALITY_SUFFIXES)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen test modalities: {missing}")
    return files


def _test_evaluator_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(_evaluator_sha256().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _box_overlap_one_voxel(left: Sequence[int], right: Sequence[int]) -> bool:
    return all(min(left[i + 3], right[i + 3]) > max(left[i], right[i]) for i in range(3))


def dataset_ap_1vox(records: Sequence[dict[str, Any]]) -> float:
    total_gt = sum(len(record["targets"]) for record in records)
    if total_gt == 0:
        return 0.0
    detections = sorted(
        (
            (float(item["score"]), case_index, item["box_zyx"])
            for case_index, record in enumerate(records)
            for item in record["detections"]
        ),
        reverse=True,
    )
    matched = [set() for _ in records]
    tp: list[float] = []
    fp: list[float] = []
    for _, case_index, box in detections:
        candidate = None
        for target_index, target in enumerate(records[case_index]["targets"]):
            if target_index not in matched[case_index] and _box_overlap_one_voxel(box, target["box_zyx"]):
                candidate = target_index
                break
        if candidate is None:
            tp.append(0.0)
            fp.append(1.0)
        else:
            matched[case_index].add(candidate)
            tp.append(1.0)
            fp.append(0.0)
    if not tp:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(total_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


class RawTestDatasetAdapter:
    def __init__(self, predictor: MicroBTUnitMassPredictor, source_root: Path, labels_ts: Path):
        self.predictor = predictor
        self.source_root = source_root
        self.labels_ts = labels_ts
        self.preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

    def load_case(self, subject_id: str):
        image_files = [str(path) for path in resolve_met_case_files(self.source_root, subject_id)]
        label_file = self.labels_ts / f"{subject_id}.nii.gz"
        if not label_file.is_file():
            raise FileNotFoundError(label_file)
        data, segmentation, properties = self.preprocessor.run_case(
            image_files,
            str(label_file),
            self.predictor.plans_manager,
            self.predictor.configuration_manager,
            self.predictor.dataset_json,
        )
        return data, segmentation, None, properties


def _save_prediction(path: Path, raw: torch.Tensor) -> None:
    output = raw_logits_to_umctf(raw)
    probabilities = output["region_probabilities"].numpy()
    decoded = decode_umctf_instances(
        probabilities[0],
        output["object_density"].numpy()[0],
        output["transport"].numpy(),
        offset_scale_zyx=tuple(DECODER_CONTRACT["offset_scale_zyx"]),
        support_threshold=DECODER_CONTRACT["support_threshold"],
        density_threshold=DECODER_CONTRACT["density_threshold"],
        min_atom_mass=DECODER_CONTRACT["min_atom_mass"],
        nms_radius=DECODER_CONTRACT["nms_radius"],
        max_centers=DECODER_CONTRACT["max_centers"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            region_mask=(probabilities > DECODER_CONTRACT["region_threshold"]).astype(np.uint8),
            instance_labels=np.asarray(decoded.instance_labels, dtype=np.int16),
        )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen raw-volume UMCTF test251 evaluator")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--labels-ts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=251)
    parser.add_argument("--total-expected-count", type=int, default=251)
    parser.add_argument("--subject-ids-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = args.model_folder / "fold_all" / args.checkpoint
    if _sha256(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint hash drift")
    subjects = select_test_subjects(
        args.labels_ts,
        expected_count=args.expected_count,
        total_expected_count=args.total_expected_count,
        subject_ids_json=args.subject_ids_json,
    )
    for subject in subjects:
        resolve_met_case_files(args.source_root, subject)
    evaluator_sha256 = _test_evaluator_sha256()
    output = args.output_dir.resolve()
    case_folder = output / "cases"
    prediction_folder = output / "predictions"
    case_folder.mkdir(parents=True, exist_ok=True)
    prediction_folder.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "expected_count": args.expected_count,
        "total_expected_count": args.total_expected_count,
        "subject_ids": list(subjects),
        "decoder_contract": DECODER_CONTRACT,
        "source_root": str(args.source_root.resolve()),
        "labels_ts": str(args.labels_ts.resolve()),
        "saved_predictions": ["region_mask", "instance_labels"],
    }
    contract_path = output / "contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("existing test251 contract drift")
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
        str(args.model_folder), use_folds=("all",), checkpoint_name=args.checkpoint
    )
    predictor.network.to("cuda").eval()
    dataset = RawTestDatasetAdapter(predictor, args.source_root, args.labels_ts)
    spacing = tuple(float(value) for value in predictor.configuration_manager.spacing)
    regions = predictor.label_manager.foreground_regions
    original_predict = predictor.predict_logits_from_preprocessed_data
    captured: dict[str, torch.Tensor] = {}

    def capture(data: torch.Tensor) -> torch.Tensor:
        result = original_predict(data)
        captured["raw"] = result.detach().cpu()
        return result

    predictor.predict_logits_from_preprocessed_data = capture
    records = []
    reused = 0
    for index, subject in enumerate(subjects, start=1):
        case_path = case_folder / f"{subject}.json"
        prediction_path = prediction_folder / f"{subject}.npz"
        record = None
        if case_path.is_file() and prediction_path.is_file():
            try:
                candidate = json.loads(case_path.read_text())
                validate_case_record(candidate, subject, args.checkpoint_sha256, evaluator_sha256)
                record = candidate
                reused += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                record = None
        if record is None:
            record = evaluate_case(
                predictor,
                dataset,
                subject,
                spacing=spacing,
                regions=regions,
                checkpoint_sha256=args.checkpoint_sha256,
                evaluator_sha256=evaluator_sha256,
            )
            _save_prediction(prediction_path, captured.pop("raw"))
            atomic_write_json(case_path, record)
        records.append(record)
        atomic_write_json(output / "progress.json", {
            "schema_version": 1,
            "status": "running" if index < len(subjects) else "aggregating",
            "completed": index,
            "expected": len(subjects),
            "reused": reused,
            "last_subject_id": subject,
        })
        print(json.dumps({"completed": index, "expected": len(subjects), "subject_id": subject}), flush=True)
        torch.cuda.empty_cache()

    summary = summarize_case_records(
        records,
        subjects,
        expected_count=args.expected_count,
        checkpoint_sha256=args.checkpoint_sha256,
        evaluator_sha256=evaluator_sha256,
    )
    summary["metrics"]["ap_1vox"] = dataset_ap_1vox(records)
    summary["prediction_file_count"] = len(list(prediction_folder.glob("*.npz")))
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "progress.json", {
        "schema_version": 1,
        "status": "complete",
        "completed": len(subjects),
        "expected": len(subjects),
        "reused": reused,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
