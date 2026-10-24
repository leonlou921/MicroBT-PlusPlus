from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nnunetv2.inference.microbt_umctf_predictor import MicroBTUnitMassPredictor, raw_logits_to_umctf
from scripts.a1_cc_contract import (
    REGION_THRESHOLD,
    SCORE_TOP_FRACTION,
    connected_components_with_scores,
)
from scripts.evaluate_microbt_umctf_full203 import DECODER_CONTRACT, atomic_write_json
from scripts.evaluate_microbt_umctf_test251 import RawTestDatasetAdapter, select_test_subjects


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen A1: B0 encoding plus 26CC, without atomic transport decoding")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument(
        "--ablation",
        choices=("A0_official_resencm", "A1_measure_encoding_without_atomic_transport_decoder"),
        default="A1_measure_encoding_without_atomic_transport_decoder",
    )
    parser.add_argument("--checkpoint", default="checkpoint_phase_200.pth")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--labels-ts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--total-expected-count", type=int, default=251)
    parser.add_argument("--subject-ids-json", type=Path)
    args = parser.parse_args()

    checkpoint = args.model_folder / "fold_all" / args.checkpoint
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError("checkpoint hash drift")
    subjects = select_test_subjects(
        args.labels_ts,
        expected_count=args.expected_count,
        total_expected_count=args.total_expected_count,
        subject_ids_json=args.subject_ids_json,
    )
    output = args.output_dir.resolve()
    predictions = output / "predictions"
    cases = output / "cases"
    predictions.mkdir(parents=True, exist_ok=True)
    cases.mkdir(parents=True, exist_ok=True)
    evaluator_hash = sha256(Path(__file__))
    contract = {
        "schema_version": 1,
        "ablation": args.ablation,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "evaluator_sha256": evaluator_hash,
        "source_root": str(args.source_root.resolve()),
        "labels_ts": str(args.labels_ts.resolve()),
        "model_folder": str(args.model_folder.resolve()),
        "subject_ids": list(subjects),
        "expected_count": len(subjects),
        "total_expected_count": args.total_expected_count,
        "instance_contract": {
            "method": "WT_26_connected_components",
            "region_threshold": REGION_THRESHOLD,
            "minimum_component_voxels": 1,
            "score": "mean_top_10_percent_WT_probability",
            "score_top_fraction": SCORE_TOP_FRACTION,
        },
        "saved_predictions": ["region_probabilities_float16", "region_mask", "instance_labels", "component_scores"],
    }
    contract_path = output / "contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("existing A1 contract drift")
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
    reused = 0
    for index, subject in enumerate(subjects, 1):
        target = predictions / f"{subject}.npz"
        case_path = cases / f"{subject}.json"
        component_count = None
        if target.is_file():
            with np.load(target) as existing:
                if set(existing.files) == {"region_probabilities", "region_mask", "instance_labels", "component_scores"}:
                    reused += 1
                    component_count = int(len(existing["component_scores"]))
        if component_count is None:
            data, _, previous_stage, _ = dataset.load_case(subject)
            if previous_stage is not None:
                raise RuntimeError("A1 does not support cascaded data")
            raw = predictor.predict_logits_from_preprocessed_data(
                torch.from_numpy(np.asarray(data, dtype=np.float32))
            ).cpu()
            if raw.shape[0] == 3:
                probabilities = torch.sigmoid(raw.float()).numpy()
            elif raw.shape[0] >= 7:
                probabilities = raw_logits_to_umctf(raw)["region_probabilities"].numpy()
            else:
                raise RuntimeError(f"unsupported region-logit contract with {raw.shape[0]} channels")
            if not np.isfinite(probabilities).all():
                raise RuntimeError(f"nonfinite A1 probabilities: {subject}")
            instances, scores = connected_components_with_scores(probabilities[0])
            component_count = int(len(scores))
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    region_probabilities=probabilities.astype(np.float16),
                    region_mask=(probabilities > REGION_THRESHOLD).astype(np.uint8),
                    instance_labels=instances,
                    component_scores=scores,
                )
            temporary.replace(target)
        atomic_write_json(case_path, {
            "schema_version": 1,
            "subject_id": subject,
            "checkpoint_sha256": args.checkpoint_sha256,
            "evaluator_sha256": evaluator_hash,
            "predicted_count": component_count,
            "instance_contract": contract["instance_contract"],
        })
        atomic_write_json(output / "progress.json", {
            "status": "running" if index < len(subjects) else "complete",
            "completed": index,
            "expected": len(subjects),
            "reused": reused,
        })
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
