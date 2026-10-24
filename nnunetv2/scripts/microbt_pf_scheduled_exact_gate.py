from __future__ import annotations

import argparse
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

from nnunetv2.scripts.microbt_pf_exact_fast_gate import (
    _extension_hash,
    _finish_loader,
    _pair_hash,
    _protected_unchanged,
    atomic_write_json,
)
from nnunetv2.scripts.microbt_pf_gpu_feasibility import (
    _protected_snapshot,
    build_nnunet_dataset,
    select_cached_cases,
)
from nnunetv2.topology.persistent_field_0d import persistence_pairs_0d
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    precompute_persistent_target_topology,
)
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ScheduledExact,
)


DATASET_NAME = "Dataset001_Example"
EPOCH_LIMIT_SECONDS = 360.0
EXPECTED_SCHEDULE = {
    "num_epochs": 200,
    "tail_start_epoch": 197,
    "sparse_interval": 10,
    "sparse_h0_weight": 10.0,
    "tail_h0_weight": 1.0,
}


def target_topology_hashes(target: PersistentFieldTarget) -> dict[str, str]:
    enriched = precompute_persistent_target_topology(target)
    reference_pairs = persistence_pairs_0d(target.field)
    return {
        "precomputed": _pair_hash(enriched.topology.pairs),
        "reference": _pair_hash(reference_pairs),
    }


def evaluate_scheduled_exact_gate(payload: dict) -> dict:
    reasons: list[str] = []
    if payload.get("status") != "complete":
        reasons.append("run_incomplete")
    if payload.get("schedule") != EXPECTED_SCHEDULE:
        reasons.append("schedule_contract_mismatch")

    hashes = payload.get("topology_hashes", {})
    if not hashes.get("precomputed") or hashes.get("precomputed") != hashes.get(
        "reference"
    ):
        reasons.append("topology_hash_mismatch")

    hundred = payload.get("hundred_step", {})
    if int(hundred.get("steps", -1)) != 100:
        reasons.append("hundred_step_count")
    if hundred.get("schedule_mode") != "sparse_exact":
        reasons.append("hundred_step_schedule_mode")
    if int(hundred.get("exact_h0_active_count", -1)) != 10:
        reasons.append("hundred_step_h0_count")
    if int(hundred.get("schedule_skip_count", -1)) != 90:
        reasons.append("hundred_step_schedule_skip_count")
    if int(hundred.get("nonfinite_steps", -1)) != 0:
        reasons.append("hundred_step_nonfinite")
    if int(hundred.get("optimizer_skips", -1)) != 0:
        reasons.append("hundred_step_optimizer_skip")

    epoch = payload.get("full_epoch", {})
    if int(epoch.get("training_iterations", -1)) != 250:
        reasons.append("full_epoch_training_count")
    if int(epoch.get("validation_iterations", -1)) != 50:
        reasons.append("full_epoch_validation_count")
    if epoch.get("schedule_mode") != "sparse_exact":
        reasons.append("full_epoch_schedule_mode")
    if int(epoch.get("exact_h0_train_calls", -1)) != 25:
        reasons.append("full_epoch_h0_count")
    if int(epoch.get("schedule_skip_count", -1)) != 225:
        reasons.append("full_epoch_schedule_skip_count")
    if int(epoch.get("validation_exact_h0_calls", -1)) != 5:
        reasons.append("validation_h0_audit_count")
    if int(epoch.get("nonfinite_steps", -1)) != 0:
        reasons.append("full_epoch_nonfinite")
    if int(epoch.get("optimizer_skips", -1)) != 0:
        reasons.append("full_epoch_optimizer_skip")
    if float(epoch.get("wall_seconds", float("inf"))) > EPOCH_LIMIT_SECONDS:
        reasons.append("epoch_wall_exceeds_360_seconds")

    tail = payload.get("tail_step", {})
    tail_contract = (
        int(tail.get("epoch", -1)) == 197
        and tail.get("schedule_mode") == "full_exact"
        and float(tail.get("effective_h0_weight", -1.0)) == 1.0
        and int(tail.get("exact_h0_active_count", -1)) == 1
        and int(tail.get("schedule_skip_count", -1)) == 0
        and int(tail.get("nonfinite_steps", -1)) == 0
        and int(tail.get("optimizer_skips", -1)) == 0
        and tail.get("finite_loss") is True
    )
    if not tail_contract:
        reasons.append("tail_step_contract")
    if payload.get("protected_state_unchanged") is not True:
        reasons.append("protected_state_drift")
    return {"admitted": not reasons, "reasons": reasons}


def _load_trainer_and_loaders(args):
    if (
        args.dataset != 1
        or args.configuration != "3d_fullres"
        or args.plans != "nnUNetResEncUNetMPlans"
        or args.fold != 0
    ):
        raise ValueError(
            "scheduled-exact gate is frozen to Dataset001 ResEnc-M 3d_fullres fold 0"
        )
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
    trainer = MicroBTPFTrainerA3ScheduledExact(
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
        raise RuntimeError("scheduled-exact gate requires real multi-process augmenters")
    return trainer, train_loader, validation_loader, cases


def _run_training_steps(trainer, loader, steps: int) -> tuple[dict, object]:
    losses: list[float] = []
    step_times: list[float] = []
    wait_times: list[float] = []
    first_persistent = None
    exact_before = int(trainer.exact_fast_telemetry["exact_h0_train_calls"])
    schedule_before = int(
        trainer.scheduled_exact_telemetry["schedule_skips"]
    )
    nonfinite_before = int(trainer.exact_fast_telemetry["nonfinite_steps"])
    optimizer_skips_before = int(trainer.exact_fast_telemetry["skipped_steps"])
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
        raise RuntimeError("scheduled-exact probe produced a nonfinite loss")
    exact = trainer.exact_fast_telemetry
    scheduled = trainer.scheduled_exact_telemetry
    return (
        {
            "steps": int(steps),
            "schedule_mode": scheduled["schedule_mode"],
            "wall_seconds": float(wall_seconds),
            "median_step_seconds": float(statistics.median(step_times)),
            "median_batch_wait_seconds": float(statistics.median(wait_times)),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "exact_h0_active_count": int(exact["exact_h0_train_calls"])
            - exact_before,
            "schedule_skip_count": int(scheduled["schedule_skips"])
            - schedule_before,
            "nonfinite_steps": int(exact["nonfinite_steps"]) - nonfinite_before,
            "optimizer_skips": int(exact["skipped_steps"])
            - optimizer_skips_before,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        first_persistent,
    )


def _run_full_epoch(trainer, train_loader, validation_loader) -> dict:
    trainer.current_epoch = 0
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
    exact = trainer.exact_fast_telemetry
    scheduled = trainer.scheduled_exact_telemetry
    losses = np.asarray([float(output["loss"]) for output in train_outputs])
    return {
        "training_iterations": int(len(train_outputs)),
        "validation_iterations": int(len(validation_outputs)),
        "schedule_mode": scheduled["schedule_mode"],
        "wall_seconds": float(wall_seconds),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "exact_h0_train_calls": int(exact["exact_h0_train_calls"]),
        "schedule_skip_count": int(scheduled["schedule_skips"]),
        "validation_exact_h0_calls": int(exact["validation_exact_h0_calls"]),
        "target_precompute_wait_seconds": float(
            exact["target_precompute_wait_seconds"]
        ),
        "topology_wall_seconds": float(exact["topology_wall_seconds"]),
        "nonfinite_steps": int(exact["nonfinite_steps"]),
        "optimizer_skips": int(exact["skipped_steps"]),
        "last_preclip_gradient_norm": float(exact["last_preclip_gradient_norm"]),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _run_tail_step(trainer, train_loader) -> dict:
    trainer.current_epoch = 197
    trainer.on_train_epoch_start()
    exact_before = int(trainer.exact_fast_telemetry["exact_h0_train_calls"])
    skip_before = int(trainer.scheduled_exact_telemetry["schedule_skips"])
    nonfinite_before = int(trainer.exact_fast_telemetry["nonfinite_steps"])
    optimizer_skips_before = int(trainer.exact_fast_telemetry["skipped_steps"])
    values = trainer.train_step(next(train_loader))
    torch.cuda.synchronize()
    loss = float(values["loss"])
    return {
        "epoch": 197,
        "schedule_mode": trainer.scheduled_exact_telemetry["schedule_mode"],
        "effective_h0_weight": float(
            trainer.scheduled_exact_telemetry["effective_h0_weight"]
        ),
        "exact_h0_active_count": int(
            trainer.exact_fast_telemetry["exact_h0_train_calls"]
        )
        - exact_before,
        "schedule_skip_count": int(
            trainer.scheduled_exact_telemetry["schedule_skips"]
        )
        - skip_before,
        "nonfinite_steps": int(trainer.exact_fast_telemetry["nonfinite_steps"])
        - nonfinite_before,
        "optimizer_skips": int(trainer.exact_fast_telemetry["skipped_steps"])
        - optimizer_skips_before,
        "finite_loss": bool(np.isfinite(loss)),
        "loss": loss,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MicroBT-PF scheduled-exact admission gate"
    )
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
        "schedule": dict(EXPECTED_SCHEDULE),
        "protected_state_before": before,
    }
    atomic_write_json(args.artifact, payload)
    train_loader = validation_loader = None
    try:
        trainer, train_loader, validation_loader, cases = _load_trainer_and_loaders(
            args
        )
        payload["cases"] = list(cases)
        hundred, first_persistent = _run_training_steps(trainer, train_loader, 100)
        payload["hundred_step"] = hundred
        payload["topology_hashes"] = target_topology_hashes(first_persistent)
        atomic_write_json(args.artifact, payload)

        payload["full_epoch"] = _run_full_epoch(
            trainer, train_loader, validation_loader
        )
        atomic_write_json(args.artifact, payload)
        payload["tail_step"] = _run_tail_step(trainer, train_loader)
        after = _protected_snapshot()
        payload["protected_state_after"] = after
        payload["protected_state_unchanged"] = _protected_unchanged(before, after)
        payload["status"] = "complete"
        payload["admission"] = evaluate_scheduled_exact_gate(payload)
        atomic_write_json(args.artifact, payload)
        return 0 if payload["admission"]["admitted"] else 3
    except Exception as error:
        payload["status"] = "error"
        payload["error"] = f"{type(error).__name__}: {error}"
        after = _protected_snapshot()
        payload["protected_state_after"] = after
        payload["protected_state_unchanged"] = _protected_unchanged(before, after)
        payload["admission"] = evaluate_scheduled_exact_gate(payload)
        atomic_write_json(args.artifact, payload)
        return 3
    finally:
        if train_loader is not None:
            _finish_loader(train_loader)
        if validation_loader is not None:
            _finish_loader(validation_loader)


if __name__ == "__main__":
    raise SystemExit(main())
