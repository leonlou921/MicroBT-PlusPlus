from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from types import MethodType

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)

import nnunetv2
from nnunetv2.scripts.microbt_pf_gpu_feasibility import (
    _protected_snapshot,
    build_nnunet_dataset,
    select_cached_cases,
)
from nnunetv2.topology.persistent_field_0d import persistence_pairs_0d
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ExactFast,
)


DATASET_NAME = "Dataset001_Example"
EPOCH_LIMIT_SECONDS = 360.0


def evaluate_exact_fast_gate(payload: dict) -> dict:
    reasons: list[str] = []
    if payload.get("status") != "complete":
        reasons.append("run_incomplete")
    hashes = payload.get("topology_hashes", {})
    if not hashes.get("precomputed") or hashes.get("precomputed") != hashes.get(
        "reference"
    ):
        reasons.append("topology_hash_mismatch")

    hundred = payload.get("hundred_step", {})
    if int(hundred.get("steps", -1)) != 100 or int(
        hundred.get("exact_h0_active_count", -1)
    ) != 100:
        reasons.append("hundred_step_h0_incomplete")
    if int(hundred.get("nonfinite_steps", -1)) != 0:
        reasons.append("hundred_step_nonfinite")
    if int(hundred.get("optimizer_skips", -1)) != 0:
        reasons.append("hundred_step_optimizer_skip")

    epoch = payload.get("full_epoch", {})
    if int(epoch.get("training_iterations", -1)) != 250:
        reasons.append("full_epoch_training_count")
    if int(epoch.get("validation_iterations", -1)) != 50:
        reasons.append("full_epoch_validation_count")
    if int(epoch.get("exact_h0_train_calls", -1)) != 250:
        reasons.append("full_epoch_h0_incomplete")
    if int(epoch.get("validation_exact_h0_calls", -1)) != 5:
        reasons.append("validation_h0_audit_count")
    if int(epoch.get("nonfinite_steps", -1)) != 0:
        reasons.append("full_epoch_nonfinite")
    if int(epoch.get("optimizer_skips", -1)) != 0:
        reasons.append("full_epoch_optimizer_skip")
    if float(epoch.get("wall_seconds", float("inf"))) > EPOCH_LIMIT_SECONDS:
        reasons.append("epoch_wall_exceeds_360_seconds")
    if payload.get("protected_state_unchanged") is not True:
        reasons.append("protected_state_drift")
    return {"admitted": not reasons, "reasons": reasons}


def atomic_write_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _pair_hash(pairs) -> str:
    digest = hashlib.sha256()
    digest.update(pairs.birth_flat.tobytes())
    digest.update(pairs.death_flat.tobytes())
    digest.update(pairs.essential.tobytes())
    return digest.hexdigest()


def _extension_hash() -> str:
    package = Path(nnunetv2.__file__).resolve().parent
    extension = next(iter(sorted(package.glob("_persistent_field_0d*.so"))), None)
    if extension is None:
        raise RuntimeError("compiled persistent-field extension is missing")
    digest = hashlib.sha256()
    with extension.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_unchanged(before: dict, after: dict) -> bool:
    return all(
        before.get(key) == after.get(key)
        for key in (
            "source_head",
            "source_dirty_count",
            "source_status_sha256",
            "protected_processes",
            "protected_checkpoints",
        )
    )


def _load_trainer_and_loaders(args):
    if (
        args.dataset != 1
        or args.configuration != "3d_fullres"
        or args.plans != "nnUNetResEncUNetMPlans"
        or args.fold != 0
    ):
        raise ValueError("exact-fast gate is frozen to Dataset001 ResEnc-M 3d_fullres fold 0")
    dataset_directory = Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME
    plans = json.loads(
        (dataset_directory / f"{args.plans}.json").read_text(encoding="utf-8")
    )
    dataset_json = json.loads(
        (dataset_directory / "dataset.json").read_text(encoding="utf-8")
    )
    cache = (
        Path(os.environ["MICROBT_PF_CACHE"])
        / DATASET_NAME
        / args.plans
        / args.configuration
    )
    cases = select_cached_cases(cache, count=5)
    source = dataset_directory / "nnUNetPlans_3d_fullres"
    trainer = MicroBTPFTrainerA3ExactFast(
        plans,
        args.configuration,
        args.fold,
        dataset_json,
        device=torch.device("cuda"),
    )
    trainer.batch_size = 1
    trainer.initialize()
    train_dataset = build_nnunet_dataset(source, cases)
    validation_dataset = build_nnunet_dataset(source, cases)
    trainer.get_tr_and_val_datasets = MethodType(
        lambda self: (train_dataset, validation_dataset), trainer
    )
    train_loader, validation_loader = trainer.get_dataloaders()
    if not isinstance(train_loader, NonDetMultiThreadedAugmenter) or not isinstance(
        validation_loader, NonDetMultiThreadedAugmenter
    ):
        raise RuntimeError("exact-fast gate requires real multi-process augmenters")
    return trainer, train_loader, validation_loader, cases


def _run_training_steps(trainer, loader, steps: int) -> tuple[dict, object]:
    losses: list[float] = []
    step_times: list[float] = []
    wait_times: list[float] = []
    first_persistent = None
    active_before = int(trainer.exact_fast_telemetry["exact_h0_train_calls"])
    skipped_before = int(trainer.exact_fast_telemetry["skipped_steps"])
    nonfinite_before = int(trainer.exact_fast_telemetry["nonfinite_steps"])
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    for _ in range(steps):
        wait_started = time.perf_counter()
        batch = next(loader)
        wait_times.append(time.perf_counter() - wait_started)
        if first_persistent is None:
            first_persistent = batch["persistent_targets"][0]
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        values = trainer.train_step(batch)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_started)
        losses.append(float(values["loss"]))
    wall_seconds = time.perf_counter() - wall_started
    if not losses or not np.isfinite(losses).all():
        raise RuntimeError("100-step exact-fast run produced a nonfinite loss")
    telemetry = trainer.exact_fast_telemetry
    return (
        {
            "steps": int(steps),
            "wall_seconds": float(wall_seconds),
            "median_step_seconds": float(statistics.median(step_times)),
            "median_batch_wait_seconds": float(statistics.median(wait_times)),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "exact_h0_active_count": int(telemetry["exact_h0_train_calls"])
            - active_before,
            "nonfinite_steps": int(telemetry["nonfinite_steps"])
            - nonfinite_before,
            "optimizer_skips": int(telemetry["skipped_steps"]) - skipped_before,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        first_persistent,
    )


def _run_full_epoch(trainer, train_loader, validation_loader) -> dict:
    torch.cuda.reset_peak_memory_stats()
    wall_started = time.perf_counter()
    trainer.on_train_epoch_start()
    train_outputs = [
        trainer.train_step(next(train_loader))
        for _ in range(trainer.num_iterations_per_epoch)
    ]
    trainer.on_train_epoch_end(train_outputs)
    with torch.no_grad():
        trainer.on_validation_epoch_start()
        validation_outputs = [
            trainer.validation_step(next(validation_loader))
            for _ in range(trainer.num_val_iterations_per_epoch)
        ]
        trainer.on_validation_epoch_end(validation_outputs)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    telemetry = trainer.exact_fast_telemetry
    losses = np.asarray([float(output["loss"]) for output in train_outputs])
    return {
        "training_iterations": int(len(train_outputs)),
        "validation_iterations": int(len(validation_outputs)),
        "wall_seconds": float(wall_seconds),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "exact_h0_train_calls": int(telemetry["exact_h0_train_calls"]),
        "validation_exact_h0_calls": int(
            telemetry["validation_exact_h0_calls"]
        ),
        "target_precompute_wait_seconds": float(
            telemetry["target_precompute_wait_seconds"]
        ),
        "topology_wall_seconds": float(telemetry["topology_wall_seconds"]),
        "nonfinite_steps": int(telemetry["nonfinite_steps"]),
        "optimizer_skips": int(telemetry["skipped_steps"]),
        "last_preclip_gradient_norm": float(
            telemetry["last_preclip_gradient_norm"]
        ),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _finish_loader(loader) -> None:
    if isinstance(loader, NonDetMultiThreadedAugmenter):
        loader._finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MicroBT-PF exact-fast admission gate")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    before = _protected_snapshot()
    payload = {
        "schema_version": 1,
        "status": "running",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "extension_sha256": _extension_hash(),
        "seed": int(args.seed),
        "device": torch.cuda.get_device_name(0),
        "bf16": True,
        "protected_state_before": before,
    }
    atomic_write_json(args.artifact, payload)
    train_loader = validation_loader = None
    try:
        trainer, train_loader, validation_loader, cases = _load_trainer_and_loaders(args)
        payload["cases"] = list(cases)
        hundred, first_persistent = _run_training_steps(trainer, train_loader, 100)
        payload["hundred_step"] = hundred
        reference_pairs = persistence_pairs_0d(first_persistent.field)
        payload["topology_hashes"] = {
            "precomputed": _pair_hash(first_persistent.topology.pairs),
            "reference": _pair_hash(reference_pairs),
        }
        atomic_write_json(args.artifact, payload)

        payload["full_epoch"] = _run_full_epoch(
            trainer, train_loader, validation_loader
        )
        after = _protected_snapshot()
        payload["protected_state_after"] = after
        payload["protected_state_unchanged"] = _protected_unchanged(before, after)
        payload["status"] = "complete"
        payload["admission"] = evaluate_exact_fast_gate(payload)
        atomic_write_json(args.artifact, payload)
        return 0 if payload["admission"]["admitted"] else 3
    except Exception as error:
        payload["status"] = "error"
        payload["error"] = f"{type(error).__name__}: {error}"
        after = _protected_snapshot()
        payload["protected_state_after"] = after
        payload["protected_state_unchanged"] = _protected_unchanged(before, after)
        payload["admission"] = evaluate_exact_fast_gate(payload)
        atomic_write_json(args.artifact, payload)
        return 3
    finally:
        if train_loader is not None:
            _finish_loader(train_loader)
        if validation_loader is not None:
            _finish_loader(validation_loader)


if __name__ == "__main__":
    raise SystemExit(main())

