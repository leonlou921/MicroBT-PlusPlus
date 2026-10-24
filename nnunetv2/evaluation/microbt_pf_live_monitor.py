from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nnunetv2.evaluation.microbt_pf_fixed_monitor import (
    evaluate_fixed_case,
    summarize_fixed_cases,
    validate_fixed_case_manifest,
)
from nnunetv2.evaluation.microbt_pf_training_monitor import upsert_jsonl


CaseRunner = Callable[[dict[str, Any]], dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_native(payload),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _git_head() -> str:
    repository = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class _TrainerCaseRunner:
    def __init__(self, trainer: Any, manifest: Mapping[str, Any]) -> None:
        if bool(getattr(trainer, "is_cascaded", False)):
            raise RuntimeError("fixed-case monitoring does not support cascaded training")
        from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor

        case_ids = [str(case["subject_id"]) for case in manifest["cases"]]
        self.trainer = trainer
        self.dataset = trainer.dataset_class(
            trainer.preprocessed_dataset_folder,
            case_ids,
            folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
        )
        inference = manifest["inference"]
        self.predictor = MicroBTPFPredictor(
            tile_step_size=float(inference["tile_step_size"]),
            use_gaussian=bool(inference["use_gaussian"]),
            use_mirroring=bool(inference["use_mirroring"]),
            perform_everything_on_device=True,
            device=trainer.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self.predictor.manual_initialization(
            trainer.network,
            trainer.plans_manager,
            trainer.configuration_manager,
            None,
            trainer.dataset_json,
            trainer.__class__.__name__,
            trainer.inference_allowed_mirroring_axes,
        )
        raw_root = Path(os.environ["nnUNet_raw"]) / str(manifest["dataset_name"])
        self.label_root = raw_root / "labelsTr"
        self.file_ending = str(trainer.dataset_json["file_ending"])
        self.small_diameter_mm = float(manifest["small_diameter_mm"])

    def __call__(self, case: dict[str, Any]) -> dict[str, Any]:
        import SimpleITK as sitk
        from nnunetv2.inference.microbt_pf_export import (
            restore_microbt_pf_prediction,
        )

        subject_id = str(case["subject_id"])
        self.trainer.print_to_log_file(
            f"MicroBT-PF fixed monitor predicting {subject_id}"
        )
        data, _, previous_stage, properties = self.dataset.load_case(subject_id)
        if previous_stage is not None:
            raise RuntimeError("fixed-case monitor received previous-stage segmentation")
        data_tensor = torch.from_numpy(np.asarray(data[:]))
        raw_logits = self.predictor.predict_sliding_window_return_logits(data_tensor)
        restored = restore_microbt_pf_prediction(
            raw_logits.cpu(),
            properties,
            self.trainer.configuration_manager,
            self.trainer.plans_manager,
            a0=False,
            spp=self.predictor.uses_support_preserving_persistence(),
        )
        label_path = self.label_root / f"{subject_id}{self.file_ending}"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        target_image = sitk.ReadImage(str(label_path))
        target_semantic = sitk.GetArrayFromImage(target_image)
        spacing_zyx = tuple(float(value) for value in target_image.GetSpacing()[::-1])
        row = evaluate_fixed_case(
            subject_id,
            restored,
            target_semantic,
            spacing_zyx=spacing_zyx,
            expected_count=int(case["expected_count"]),
            expected_small_count=int(case["expected_small_count"]),
            small_diameter_mm=self.small_diameter_mm,
        )
        row["stratum"] = str(case["stratum"])
        del data_tensor, raw_logits, restored
        if self.trainer.device.type == "cuda":
            torch.cuda.empty_cache()
        return row


class FixedCaseMonitor:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        manifest_sha256: str,
        output_root: Path,
        network: torch.nn.Module,
        case_runner: CaseRunner,
        total_epochs: int,
        git_head: str,
        checkpoint_metadata: Callable[[int, bool], Mapping[str, Any] | None],
        log: Callable[[str], None],
    ) -> None:
        self.manifest = manifest
        self.manifest_sha256 = str(manifest_sha256)
        self.output_root = Path(output_root)
        self.network = network
        self.case_runner = case_runner
        self.total_epochs = int(total_epochs)
        self.interval = int(manifest["milestone_every_epochs"])
        self.git_head = str(git_head)
        self.checkpoint_metadata = checkpoint_metadata
        self.log = log

    @classmethod
    def from_trainer(
        cls,
        trainer: Any,
        manifest_path: str | Path,
        output_root: str | Path,
        *,
        case_runner_factory: Callable[[Any, Mapping[str, Any]], CaseRunner] | None = None,
        git_head: str | None = None,
    ) -> "FixedCaseMonitor":
        source = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if int(manifest.get("fold", -1)) != int(trainer.fold):
            raise ValueError("fixed-case manifest fold differs from trainer fold")
        split_path = (
            Path(trainer.preprocessed_dataset_folder_base)
            / str(manifest.get("split_file", "splits_final.json"))
        )
        split_sha256 = _sha256(split_path)
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        fold = int(manifest["fold"])
        if fold >= len(splits):
            raise ValueError("fixed-case fold is missing from split file")
        validation_subjects = set(str(item) for item in splits[fold]["val"])
        validate_fixed_case_manifest(
            manifest,
            split_sha256=split_sha256,
            validation_subjects=validation_subjects,
        )

        destination_root = Path(output_root)
        frozen_manifest = destination_root / "fixed_cases_manifest.json"
        if frozen_manifest.is_file():
            existing = json.loads(frozen_manifest.read_text(encoding="utf-8"))
            if existing != manifest:
                raise ValueError("existing fixed-case output manifest differs")
        else:
            _atomic_write_json(frozen_manifest, manifest)

        runner_factory = case_runner_factory or _TrainerCaseRunner
        runner = runner_factory(trainer, manifest)

        def checkpoint_metadata(completed_epoch: int, final: bool) -> dict[str, Any] | None:
            filename = "checkpoint_final.pth" if final else "checkpoint_latest.pth"
            path = Path(trainer.output_folder) / filename
            if not path.is_file():
                return None
            stat = path.stat()
            return {
                "path": str(path),
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "completed_epoch": int(completed_epoch),
            }

        return cls(
            manifest=manifest,
            manifest_sha256=_sha256(source),
            output_root=destination_root,
            network=trainer.network,
            case_runner=runner,
            total_epochs=int(trainer.num_epochs),
            git_head=git_head or _git_head(),
            checkpoint_metadata=checkpoint_metadata,
            log=getattr(trainer, "print_to_log_file", print),
        )

    def is_due(self, completed_epoch: int, *, final: bool = False) -> bool:
        epoch = int(completed_epoch)
        if final:
            return epoch == self.total_epochs
        return 0 < epoch < self.total_epochs and epoch % self.interval == 0

    def run_if_due(
        self, completed_epoch: int, *, final: bool = False
    ) -> dict[str, Any] | None:
        if not self.is_due(completed_epoch, final=final):
            return None
        return self.run(completed_epoch, final=final)

    def run(self, completed_epoch: int, *, final: bool = False) -> dict[str, Any]:
        epoch = int(completed_epoch)
        epoch_root = self.output_root / f"epoch_{epoch:04d}"
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        was_training = bool(self.network.training)
        self.log(f"MicroBT-PF fixed monitor started for completed epoch {epoch}")
        try:
            self.network.eval()
            for case in self.manifest["cases"]:
                rows.append(self.case_runner(dict(case)))
            metrics = summarize_fixed_cases(rows, epoch=epoch)
            result: dict[str, Any] = {
                "schema_version": 1,
                "status": "success",
                "epoch": epoch,
                "duration_seconds": float(time.perf_counter() - started),
                "git_head": self.git_head,
                "manifest_sha256": self.manifest_sha256,
                "checkpoint": self.checkpoint_metadata(epoch, final),
                "metrics": metrics,
            }
        except Exception as error:  # runtime monitoring is deliberately fail-open
            result = {
                "schema_version": 1,
                "status": "failed",
                "epoch": epoch,
                "duration_seconds": float(time.perf_counter() - started),
                "git_head": self.git_head,
                "manifest_sha256": self.manifest_sha256,
                "checkpoint": self.checkpoint_metadata(epoch, final),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        finally:
            self.network.train(was_training)

        _atomic_write_json(
            epoch_root / "cases.json",
            {"schema_version": 1, "epoch": epoch, "cases": rows},
        )
        _atomic_write_json(epoch_root / "summary.json", result)
        upsert_jsonl(
            self.output_root / "fixed_case_history.jsonl",
            _json_native(result),
            key_fields=("epoch",),
        )
        self.log(
            "MicroBT-PF fixed monitor completed: "
            + json.dumps(
                {
                    "epoch": epoch,
                    "status": result["status"],
                    "duration_seconds": result["duration_seconds"],
                },
                sort_keys=True,
            )
        )
        return result
