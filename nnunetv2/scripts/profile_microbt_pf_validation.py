from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from nnunetv2.scripts.microbt_pf_exact_fast_gate import (
    _finish_loader,
    _load_trainer_and_loaders,
    _protected_snapshot,
    _protected_unchanged,
    atomic_write_json,
)


def _synchronize(trainer) -> None:
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)


def run_validation_profile(
    trainer,
    validation_loader,
    steps: int = 50,
    warmup: bool = True,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup:
        trainer.on_validation_epoch_start()
        trainer._exact_fast_validation_index = trainer.validation_h0_audit_iterations
        with torch.no_grad():
            trainer.validation_step(next(validation_loader))
        _synchronize(trainer)

    trainer.on_validation_epoch_start()
    outputs: list[dict] = []
    wait_times: list[float] = []
    step_times: list[float] = []
    topology_present_batches = 0
    _synchronize(trainer)
    wall_started = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            wait_started = time.perf_counter()
            batch = next(validation_loader)
            wait_times.append(time.perf_counter() - wait_started)
            topology_present_batches += int(
                any(
                    target.topology is not None
                    for target in batch["persistent_targets"]
                )
            )
            _synchronize(trainer)
            step_started = time.perf_counter()
            outputs.append(trainer.validation_step(batch))
            _synchronize(trainer)
            step_times.append(time.perf_counter() - step_started)
        trainer.on_validation_epoch_end(outputs)
    _synchronize(trainer)
    wall_seconds = time.perf_counter() - wall_started
    losses = np.asarray([float(output["loss"]) for output in outputs])
    if not np.isfinite(losses).all():
        raise RuntimeError("validation profile produced a nonfinite loss")
    return {
        "steps": int(steps),
        "warmup": bool(warmup),
        "wall_seconds": float(wall_seconds),
        "median_batch_wait_seconds": float(statistics.median(wait_times)),
        "p95_batch_wait_seconds": float(np.percentile(wait_times, 95)),
        "median_step_seconds": float(statistics.median(step_times)),
        "p95_step_seconds": float(np.percentile(step_times, 95)),
        "exact_h0_calls": int(
            trainer.exact_fast_telemetry["validation_exact_h0_calls"]
        ),
        "topology_present_batches": int(topology_present_batches),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile MicroBT-PF validation")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
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
        "protected_state_before": before,
    }
    atomic_write_json(args.artifact, payload)
    train_loader = None
    validation_loader = None
    try:
        trainer, train_loader, validation_loader, cases = _load_trainer_and_loaders(args)
        payload["cases"] = cases
        payload["profile"] = run_validation_profile(
            trainer,
            validation_loader,
            steps=args.steps,
            warmup=True,
        )
        after = _protected_snapshot()
        payload["protected_state_after"] = after
        payload["protected_state_unchanged"] = _protected_unchanged(before, after)
        payload["status"] = "complete"
        atomic_write_json(args.artifact, payload)
        print(json.dumps(payload["profile"], indent=2, sort_keys=True))
        return 0
    except Exception as error:
        payload["status"] = "error"
        payload["error"] = f"{type(error).__name__}: {error}"
        after = _protected_snapshot()
        payload["protected_state_after"] = after
        payload["protected_state_unchanged"] = _protected_unchanged(before, after)
        atomic_write_json(args.artifact, payload)
        raise
    finally:
        if train_loader is not None:
            _finish_loader(train_loader)
        if validation_loader is not None:
            _finish_loader(validation_loader)


if __name__ == "__main__":
    raise SystemExit(main())
