from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nnunetv2.evaluation.microbt_pf_fast_admission import (
    compare_admission_outputs,
    select_fastest_admitted,
)
from nnunetv2.evaluation.microbt_pf_full203 import (
    atomic_write_json,
    build_case_record,
    summarize_full203,
    validate_case_record,
)
from nnunetv2.evaluation.microbt_pf_paired import paired_comparison


@dataclass(frozen=True)
class InferenceSelection:
    mode: str
    batch_size: int | None

    def __post_init__(self) -> None:
        if self.mode not in {"stock", "fast"}:
            raise ValueError(f"unsupported inference mode: {self.mode}")
        if self.mode == "stock" and self.batch_size is not None:
            raise ValueError("stock inference must not define a microbatch size")
        if self.mode == "fast" and (self.batch_size is None or self.batch_size <= 0):
            raise ValueError("fast inference requires a positive microbatch size")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validation_subjects(
    split_path: str | Path, fold: int, *, expected_count: int = 203
) -> tuple[str, ...]:
    splits = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if not isinstance(splits, list) or int(fold) < 0 or int(fold) >= len(splits):
        raise ValueError("requested fold is absent from split file")
    values = [str(value).strip() for value in splits[int(fold)].get("val", ())]
    if len(values) != int(expected_count) or any(not value for value in values):
        raise ValueError(f"fold validation must contain exactly {expected_count} nonblank subjects")
    if len(set(values)) != len(values):
        raise ValueError("fold validation subject IDs must be unique")
    return tuple(sorted(values))


def reusable_case_record(
    path: str | Path,
    subject_id: str,
    checkpoint_sha256: str,
    evaluator_sha256: str,
    *,
    compatible_evaluator_sha256: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        recorded_evaluator = str(payload.get("evaluator_sha256", ""))
        if recorded_evaluator != str(evaluator_sha256) and recorded_evaluator not in set(
            str(value) for value in compatible_evaluator_sha256
        ):
            return None
        validate_case_record(
            payload,
            subject_id,
            checkpoint_sha256,
            recorded_evaluator,
        )
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _evaluator_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    relative = (
        "nnunetv2/nets/microbt_pf/model.py",
        "nnunetv2/topology/persistent_field_0d.py",
        "nnunetv2/inference/microbt_pf_predictor.py",
        "nnunetv2/inference/microbt_pf_export.py",
        "nnunetv2/inference/microbt_pf_fast_predictor.py",
        "nnunetv2/evaluation/microbt_pf_fast_admission.py",
        "nnunetv2/evaluation/microbt_pf_full203.py",
        "nnunetv2/evaluation/microbt_pf_paired.py",
        "nnunetv2/evaluation/microbt_pf_fixed_monitor.py",
        "nnunetv2/scripts/run_microbt_pf_full203.py",
        "scripts/evaluate_microbt_pf_causal_matrix.py",
    )
    digest = hashlib.sha256()
    for item in relative:
        path = root / item
        digest.update(item.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_predictor(
    model_folder: Path,
    checkpoint_name: str,
    fold: int,
    *,
    fast: bool,
    batch_size: int,
    device: torch.device,
):
    if fast:
        from nnunetv2.inference.microbt_pf_fast_predictor import MicroBTPFFastPredictor

        predictor_class = MicroBTPFFastPredictor
        extra = {"forward_microbatch_size": int(batch_size)}
    else:
        from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor

        predictor_class = MicroBTPFPredictor
        extra = {}
    predictor = predictor_class(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        **extra,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=(int(fold),), checkpoint_name=checkpoint_name
    )
    return predictor


def _make_dataset(predictor, subjects: tuple[str, ...]):
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    base = Path(nnUNet_preprocessed) / predictor.plans_manager.dataset_name
    folder = base / predictor.configuration_manager.data_identifier
    dataset_class = infer_dataset_class(str(folder))
    return dataset_class(str(folder), list(subjects), folder_with_segs_from_previous_stage=None)


def _read_target(subject_id: str, file_ending: str) -> tuple[np.ndarray, tuple[float, ...]]:
    import SimpleITK as sitk
    from nnunetv2.paths import nnUNet_raw

    label = Path(nnUNet_raw) / "Dataset001_Example" / "labelsTr" / f"{subject_id}{file_ending}"
    if not label.is_file():
        raise FileNotFoundError(label)
    image = sitk.ReadImage(str(label))
    return sitk.GetArrayFromImage(image), tuple(float(value) for value in image.GetSpacing()[::-1])


def _infer_case(predictor, dataset, subject_id: str) -> tuple[dict[str, Any], np.ndarray, tuple[float, ...], float, int]:
    from nnunetv2.inference.microbt_pf_export import restore_microbt_pf_prediction

    data, _, previous_stage, properties = dataset.load_case(subject_id)
    if previous_stage is not None:
        raise RuntimeError("full203 evaluation does not support a cascaded previous stage")
    tensor = torch.from_numpy(np.asarray(data[:]))
    if predictor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(predictor.device)
        torch.cuda.synchronize(predictor.device)
    started = time.perf_counter()
    raw_logits = predictor.predict_sliding_window_return_logits(tensor)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize(predictor.device)
        peak = int(torch.cuda.max_memory_allocated(predictor.device))
    else:
        peak = 0
    elapsed = float(time.perf_counter() - started)
    restored = restore_microbt_pf_prediction(
        raw_logits.cpu(),
        properties,
        predictor.configuration_manager,
        predictor.plans_manager,
        a0=False,
        spp=predictor.uses_support_preserving_persistence(),
    )
    target, spacing = _read_target(subject_id, str(predictor.dataset_json["file_ending"]))
    del tensor, raw_logits
    return restored, target, spacing, elapsed, peak


def _admission_view(
    restored: dict[str, Any], target: np.ndarray, spacing: tuple[float, ...]
) -> dict[str, Any]:
    from nnunetv2.evaluation.microbt_pf_fixed_monitor import evaluate_fixed_case

    metrics = evaluate_fixed_case("admission", restored, target, spacing_zyx=spacing)
    return {
        "segmentation": restored["segmentation"],
        "instance_ids": restored["instance_ids"],
        "count": int(restored["count"]),
        "metrics": metrics,
    }


def _release(predictor: Any | None = None) -> None:
    if predictor is not None:
        try:
            predictor.network.to("cpu")
        except Exception:
            pass
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_admission(
    model_folder: Path,
    checkpoint_names: dict[str, str],
    fold: int,
    subject_id: str,
    candidates: tuple[int, ...],
    output: Path,
    device: torch.device,
) -> InferenceSelection:
    output.mkdir(parents=True, exist_ok=True)
    selected = InferenceSelection(mode="stock", batch_size=None)
    all_results: dict[str, Any] = {}
    for checkpoint_label in ("best", "final"):
        name = checkpoint_names[checkpoint_label]
        stock = _make_predictor(model_folder, name, fold, fast=False, batch_size=1, device=device)
        stock_dataset = _make_dataset(stock, (subject_id,))
        reference_restored, target, spacing, stock_time, stock_peak = _infer_case(
            stock, stock_dataset, subject_id
        )
        reference = _admission_view(reference_restored, target, spacing)
        _release(stock)

        trial_batches = (
            candidates
            if checkpoint_label == "best"
            else ((int(selected.batch_size),) if selected.mode == "fast" else ())
        )
        trials: list[dict[str, Any]] = []
        fast = None
        if trial_batches:
            fast = _make_predictor(
                model_folder, name, fold, fast=True, batch_size=1, device=device
            )
            fast_dataset = _make_dataset(fast, (subject_id,))
            for batch in trial_batches:
                fast.forward_microbatch_size = int(batch)
                try:
                    restored, candidate_target, candidate_spacing, elapsed, peak = _infer_case(
                        fast, fast_dataset, subject_id
                    )
                    comparison = compare_admission_outputs(
                        reference,
                        _admission_view(restored, candidate_target, candidate_spacing),
                        tolerance=1e-6,
                    )
                    trials.append(
                        {
                            "batch_size": int(batch),
                            "elapsed_seconds": elapsed,
                            "peak_memory_bytes": peak,
                            "oom": False,
                            **comparison,
                        }
                    )
                    del restored
                except torch.cuda.OutOfMemoryError as error:
                    trials.append(
                        {
                            "batch_size": int(batch),
                            "elapsed_seconds": None,
                            "peak_memory_bytes": None,
                            "oom": True,
                            "admitted": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    torch.cuda.empty_cache()
                except RuntimeError as error:
                    if "out of memory" not in str(error).lower():
                        raise
                    trials.append(
                        {
                            "batch_size": int(batch),
                            "elapsed_seconds": None,
                            "peak_memory_bytes": None,
                            "oom": True,
                            "admitted": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    torch.cuda.empty_cache()
        if checkpoint_label == "best":
            if any(bool(trial.get("admitted")) for trial in trials):
                selected_batch = int(select_fastest_admitted(trials)["batch_size"])
                selected = InferenceSelection(mode="fast", batch_size=selected_batch)
        elif trials and not trials[0].get("admitted"):
            selected = InferenceSelection(mode="stock", batch_size=None)
        all_results[checkpoint_label] = {
            "checkpoint_name": name,
            "stock_elapsed_seconds": stock_time,
            "stock_peak_memory_bytes": stock_peak,
            "trials": trials,
        }
        _release(fast)
        del reference_restored, reference, target
    payload = {
        "schema_version": 2,
        "subject_id": subject_id,
        "candidates": list(candidates),
        "selected_mode": selected.mode,
        "selected_batch_size": selected.batch_size,
        "checkpoints": all_results,
    }
    atomic_write_json(output / "admission.json", payload)
    return selected


def _write_per_case_csv(path: Path, records: list[dict[str, Any]]) -> None:
    metric_names = sorted(set().union(*(record["metrics"].keys() for record in records)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "elapsed_seconds", *metric_names])
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["subject_id"]):
            writer.writerow(
                {
                    "subject_id": record["subject_id"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    **record["metrics"],
                }
            )
    temporary.replace(path)


def run_checkpoint(
    label: str,
    model_folder: Path,
    checkpoint_name: str,
    checkpoint_sha256: str,
    fold: int,
    subjects: tuple[str, ...],
    selection: InferenceSelection,
    output_root: Path,
    evaluator_sha256: str,
    status_path: Path,
    device: torch.device,
    compatible_evaluator_sha256: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = output_root / label
    case_root = root / "cases"
    case_root.mkdir(parents=True, exist_ok=True)
    predictor = _make_predictor(
        model_folder,
        checkpoint_name,
        fold,
        fast=selection.mode == "fast",
        batch_size=selection.batch_size or 1,
        device=device,
    )
    dataset = _make_dataset(predictor, subjects)
    records: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects):
        record_path = case_root / f"{subject}.json"
        reusable = reusable_case_record(
            record_path,
            subject,
            checkpoint_sha256,
            evaluator_sha256,
            compatible_evaluator_sha256=compatible_evaluator_sha256,
        )
        if reusable is not None:
            records.append(reusable)
            continue
        atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "state": label,
                "subject_id": subject,
                "completed_cases": len(records),
                "expected_cases": len(subjects),
                "checkpoint_sha256": checkpoint_sha256,
                "inference_mode": selection.mode,
                "batch_size": selection.batch_size,
                "updated_at": time.time(),
            },
        )
        restored, target, spacing, elapsed, peak = _infer_case(predictor, dataset, subject)
        record = build_case_record(
            subject,
            restored,
            target,
            spacing_zyx=spacing,
            checkpoint_sha256=checkpoint_sha256,
            evaluator_sha256=evaluator_sha256,
            elapsed_seconds=elapsed,
        )
        record["peak_memory_bytes"] = int(peak)
        atomic_write_json(record_path, record)
        records.append(record)
        del restored, target
    summary = summarize_full203(records, subjects, expected_count=203)
    summary.update(
        {
            "checkpoint_label": label,
            "checkpoint_name": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha256,
            "evaluator_sha256": evaluator_sha256,
            "inference_mode": selection.mode,
            "forward_microbatch_size": selection.batch_size,
        }
    )
    atomic_write_json(root / "summary.json", summary)
    _write_per_case_csv(root / "per_case.csv", records)
    _release(predictor)
    return records, summary


def _comparison_markdown(payload: dict[str, Any], summaries: dict[str, Any]) -> str:
    lines = [
        "# MicroBT-PF checkpoint_final vs checkpoint_best",
        "",
        "All deltas are final minus best on the same 203 fold-0 subjects.",
        "",
        "| Metric | Best | Final | Delta | 95% paired CI | W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    headline = ("wt_dice", "tc_dice", "et_dice", "box_ap50", "box_ap75", "count_mae", "small_lesion_recall_3mm")
    for metric in headline:
        comparison = payload["metrics"].get(metric)
        if comparison is None:
            continue
        best = summaries["best"]["macro"].get(metric, {}).get("mean")
        final = summaries["final"]["macro"].get(metric, {}).get("mean")
        ci = f"[{comparison['ci95_low']:.6g}, {comparison['ci95_high']:.6g}]" if comparison["ci95_low"] is not None else "n/a"
        lines.append(
            f"| {metric} | {best if best is not None else 'n/a'} | {final if final is not None else 'n/a'} | "
            f"{comparison['mean_delta'] if comparison['mean_delta'] is not None else 'n/a'} | {ci} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} |"
        )
    lines.extend(
        [
            "",
            f"Dataset AP50: best={summaries['best']['dataset_ap50']}, final={summaries['final']['dataset_ap50']}.",
            f"Dataset AP75: best={summaries['best']['dataset_ap75']}, final={summaries['final']['dataset_ap75']}.",
            f"Registered <=3 mm box recall50: best={summaries['best']['small_lesion_box_recall50']}, final={summaries['final']['small_lesion_box_recall50']}.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired MicroBT-PF full-203 evaluation")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-best-name", default="checkpoint_best.pth")
    parser.add_argument("--checkpoint-final-name", default="checkpoint_final.pth")
    parser.add_argument("--checkpoint-best-sha256", required=True)
    parser.add_argument("--checkpoint-final-sha256", required=True)
    parser.add_argument("--admission-subject")
    parser.add_argument("--batch-candidates", default="1,2,4,6,8")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--compatible-evaluator-sha256",
        action="append",
        default=[],
        help="Previously verified evaluator hash whose existing case records may be resumed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    try:
        device = torch.device(arguments.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        subjects = load_validation_subjects(arguments.split, arguments.fold)
        checkpoints = {
            "best": arguments.checkpoint_best_name,
            "final": arguments.checkpoint_final_name,
        }
        hashes = {
            "best": arguments.checkpoint_best_sha256,
            "final": arguments.checkpoint_final_sha256,
        }
        for label in ("best", "final"):
            actual = sha256_file(arguments.model_folder / f"fold_{arguments.fold}" / checkpoints[label])
            if actual != hashes[label]:
                raise RuntimeError(f"{label} checkpoint SHA256 mismatch")
        evaluator_hash = _evaluator_sha256()
        atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "state": "admission",
                "expected_cases": 203,
                "subject_list_sha256": hashlib.sha256("\n".join(subjects).encode()).hexdigest(),
                "evaluator_sha256": evaluator_hash,
                "updated_at": time.time(),
            },
        )
        candidates = tuple(int(value) for value in arguments.batch_candidates.split(","))
        if not candidates or any(value <= 0 for value in candidates):
            raise ValueError("batch candidates must be positive")
        selection = run_admission(
            arguments.model_folder,
            checkpoints,
            arguments.fold,
            arguments.admission_subject or subjects[0],
            candidates,
            output / "admission",
            device,
        )
        records: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for label in ("best", "final"):
            records[label], summaries[label] = run_checkpoint(
                label,
                arguments.model_folder,
                checkpoints[label],
                hashes[label],
                arguments.fold,
                subjects,
                selection,
                output,
                evaluator_hash,
                status_path,
                device,
                tuple(arguments.compatible_evaluator_sha256),
            )
        atomic_write_json(status_path, {"schema_version": 1, "state": "analysis", "updated_at": time.time()})
        comparison = paired_comparison(records["best"], records["final"])
        comparison["dataset_summary"] = {
            label: {
                key: summaries[label][key]
                for key in ("dataset_ap50", "dataset_ap75", "small_lesion_box_recall50")
            }
            for label in ("best", "final")
        }
        atomic_write_json(output / "comparison.json", comparison)
        (output / "comparison.md").write_text(
            _comparison_markdown(comparison, summaries), encoding="utf-8"
        )
        atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "state": "complete",
                "completed_cases_best": 203,
                "completed_cases_final": 203,
                "selected_inference_mode": selection.mode,
                "selected_batch_size": selection.batch_size,
                "software": {"python": platform.python_version(), "torch": torch.__version__},
                "updated_at": time.time(),
            },
        )
        return 0
    except Exception as error:
        atomic_write_json(
            status_path,
            {
                "schema_version": 1,
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "updated_at": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
