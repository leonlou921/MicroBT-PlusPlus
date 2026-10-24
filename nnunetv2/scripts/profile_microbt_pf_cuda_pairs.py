from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from nnunetv2.scripts.microbt_pf_gpu_feasibility import _protected_snapshot
from nnunetv2.topology.persistent_field_0d import (
    exact_backend_info,
    persistence_pairs_0d,
    persistence_pairs_0d_cuda,
)


def evaluate_cuda_pairs_probe(payload: dict) -> dict:
    reasons: list[str] = []
    if int(payload.get("randomized_mismatches", -1)) != 0:
        reasons.append("randomized_exactness_mismatch")
    if int(payload.get("tie_heavy_mismatches", -1)) != 0:
        reasons.append("tie_heavy_exactness_mismatch")
    if payload.get("deterministic") is not True:
        reasons.append("repeated_run_nondeterminism")
    if float(payload.get("constant_fullres_median_seconds", float("inf"))) > 0.50:
        reasons.append("median_exceeds_0.50_seconds")
    if float(payload.get("constant_fullres_p95_seconds", float("inf"))) > 0.70:
        reasons.append("p95_exceeds_0.70_seconds")
    if payload.get("protected_state_unchanged") is not True:
        reasons.append("protected_state_drift")
    return {"admitted": not reasons, "reasons": reasons}


def _pair_hash(pairs) -> str:
    digest = hashlib.sha256()
    digest.update(pairs.birth_flat.tobytes())
    digest.update(pairs.death_flat.tobytes())
    digest.update(pairs.essential.tobytes())
    return digest.hexdigest()


def _pairs_equal(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in ("birth_flat", "death_flat", "essential")
    )


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _protected_equal(before: dict, after: dict) -> bool:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact CUDA H0 pair prerequisite probe")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--randomized", type=int, default=1000)
    parser.add_argument("--tie-heavy", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.randomized <= 0 or args.tie_heavy <= 0 or args.repeats < 3:
        raise ValueError("probe counts are below the frozen minimum")
    before = _protected_snapshot()
    payload = {
        "schema_version": 1,
        "status": "running",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "backend": exact_backend_info(),
        "device": torch.cuda.get_device_name(0),
        "shape": [128, 160, 112],
        "randomized_cases": int(args.randomized),
        "tie_heavy_cases": int(args.tie_heavy),
        "seed": int(args.seed),
        "protected_state_before": before,
    }
    _atomic_write(args.artifact, payload)
    generator = np.random.default_rng(args.seed)
    randomized_mismatches = 0
    for _ in range(args.randomized):
        shape = tuple(int(value) for value in generator.integers(1, 9, size=3))
        field = generator.normal(size=shape).astype(np.float32)
        randomized_mismatches += not _pairs_equal(
            persistence_pairs_0d(field),
            persistence_pairs_0d_cuda(torch.from_numpy(field).cuda()),
        )
    tie_heavy_mismatches = 0
    for _ in range(args.tie_heavy):
        shape = tuple(int(value) for value in generator.integers(1, 10, size=3))
        field = generator.integers(-4, 5, size=shape).astype(np.float32)
        tie_heavy_mismatches += not _pairs_equal(
            persistence_pairs_0d(field),
            persistence_pairs_0d_cuda(torch.from_numpy(field).cuda()),
        )

    constant = torch.zeros((128, 160, 112), device="cuda", dtype=torch.float32)
    timings: list[float] = []
    hashes: list[str] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        pairs = persistence_pairs_0d_cuda(constant)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        hashes.append(_pair_hash(pairs))
    after = _protected_snapshot()
    payload.update(
        {
            "randomized_mismatches": int(randomized_mismatches),
            "tie_heavy_mismatches": int(tie_heavy_mismatches),
            "constant_fullres_seconds": timings,
            "constant_fullres_median_seconds": float(np.median(timings)),
            "constant_fullres_p95_seconds": float(np.percentile(timings, 95)),
            "deterministic": len(set(hashes)) == 1,
            "repeated_hash": hashes[0],
            "fullres_real_fields_tested": 0,
            "fullres_real_fields_reason": "performance_prerequisite_failed",
            "protected_state_after": after,
            "protected_state_unchanged": _protected_equal(before, after),
            "status": "complete",
        }
    )
    payload["admission"] = evaluate_cuda_pairs_probe(payload)
    _atomic_write(args.artifact, payload)
    return 0 if payload["admission"]["admitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
