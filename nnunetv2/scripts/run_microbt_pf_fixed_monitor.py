from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from nnunetv2.evaluation.microbt_pf_live_monitor import (
    FixedCaseMonitor,
    _atomic_write_json,
)
from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen MicroBT-PF fixed-case monitor from a checkpoint."
    )
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="checkpoint_latest.pth")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if arguments.case_limit is not None and arguments.case_limit <= 0:
        raise ValueError("case-limit must be positive")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.manifest.expanduser().resolve()
    if arguments.case_limit is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cases"] = manifest["cases"][: arguments.case_limit]
        manifest_path = arguments.output / "smoke_manifest.json"
        _atomic_write_json(manifest_path, manifest)

    predictor = MicroBTPFPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(arguments.model_folder),
        use_folds=(arguments.fold,),
        checkpoint_name=arguments.checkpoint_name,
    )

    from nnunetv2.paths import nnUNet_preprocessed

    preprocessed_base = Path(nnUNet_preprocessed) / predictor.plans_manager.dataset_name
    preprocessed_folder = (
        preprocessed_base / predictor.configuration_manager.data_identifier
    )
    runtime = SimpleNamespace(
        fold=arguments.fold,
        num_epochs=max(1, arguments.epoch + 1),
        network=predictor.network,
        device=device,
        plans_manager=predictor.plans_manager,
        configuration_manager=predictor.configuration_manager,
        dataset_json=predictor.dataset_json,
        inference_allowed_mirroring_axes=predictor.allowed_mirroring_axes,
        dataset_class=infer_dataset_class(str(preprocessed_folder)),
        preprocessed_dataset_folder_base=str(preprocessed_base),
        preprocessed_dataset_folder=str(preprocessed_folder),
        folder_with_segs_from_previous_stage=None,
        is_cascaded=False,
        output_folder=str(arguments.model_folder / f"fold_{arguments.fold}"),
        print_to_log_file=print,
    )
    monitor = FixedCaseMonitor.from_trainer(
        runtime,
        manifest_path,
        arguments.output,
    )
    result = monitor.run(arguments.epoch, final=False)
    print(json.dumps({"status": result["status"], "epoch": result["epoch"]}))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
