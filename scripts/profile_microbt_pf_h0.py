"""Profile the exact MicroBT-PF H0 backend on bounded synthetic and cached patches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from nnunetv2.topology.persistent_field_0d import (
    induced_matching_0d_precomputed_target,
    profile_induced_matching_0d,
)
from nnunetv2 import _persistent_field_0d as _backend
from nnunetv2.training.dataloading.microbt_pf_targets import (
    build_persistent_field_target,
)


PATCH_SHAPE = (112, 160, 128)
SYNTHETIC_SHAPES = ((32, 32, 32), (64, 64, 64), PATCH_SHAPE)
PROFILE_STAGE_KEYS = {
    "prediction_pairs",
    "target_pairs",
    "comparison_matching",
    "total",
}


def atomic_write_json(output: Path, payload: dict) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def select_cached_cases(cache_root: Path, limit: int = 5) -> tuple[Path, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    cache_root = Path(cache_root)
    cases = sorted(
        archive.parent
        for archive in cache_root.rglob("instances.npz")
        if (archive.parent / "metadata.json").is_file()
    )
    if len(cases) < limit:
        raise RuntimeError(
            f"need {limit} complete cached cases under {cache_root}, found {len(cases)}"
        )
    return tuple(cases[:limit])


def _structured_target(shape: tuple[int, int, int]) -> np.ndarray:
    target = np.full(shape, -1.0, dtype=np.float32)
    for fractions in (
        (0.20, 0.20, 0.20),
        (0.20, 0.75, 0.70),
        (0.50, 0.50, 0.50),
        (0.75, 0.25, 0.70),
        (0.78, 0.78, 0.25),
    ):
        center = tuple(
            min(shape[axis] - 2, max(1, int(round(fractions[axis] * (shape[axis] - 1)))))
            for axis in range(3)
        )
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    coordinate = (center[0] + dz, center[1] + dy, center[2] + dx)
                    if any(
                        value < 0 or value >= shape[axis]
                        for axis, value in enumerate(coordinate)
                    ):
                        continue
                    value = 1.0 - 0.18 * (abs(dz) + abs(dy) + abs(dx))
                    target[coordinate] = max(target[coordinate], value)
    return target


def _random_prediction(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def profile_field_pair(
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    source: str,
) -> dict:
    prediction_array = np.ascontiguousarray(prediction, dtype=np.float32)
    target_array = np.ascontiguousarray(target, dtype=np.float32)
    matching, profile_seconds = profile_induced_matching_0d(
        prediction_array, target_array, floor=-1.0
    )
    if set(profile_seconds) != PROFILE_STAGE_KEYS:
        raise RuntimeError("profile wrapper returned an incomplete timing schema")
    precomputed_start = time.perf_counter()
    precomputed = induced_matching_0d_precomputed_target(
        prediction_array,
        target_array,
        matching.target,
        floor=-1.0,
    )
    precomputed_seconds = time.perf_counter() - precomputed_start
    matching_arrays = (
        (matching.prediction.birth_flat, precomputed.prediction.birth_flat),
        (matching.prediction.death_flat, precomputed.prediction.death_flat),
        (matching.prediction.essential, precomputed.prediction.essential),
        (matching.target.birth_flat, precomputed.target.birth_flat),
        (matching.target.death_flat, precomputed.target.death_flat),
        (matching.target.essential, precomputed.target.essential),
        (matching.matched_prediction, precomputed.matched_prediction),
        (matching.matched_target, precomputed.matched_target),
        (matching.unmatched_prediction, precomputed.unmatched_prediction),
        (matching.unmatched_target, precomputed.unmatched_target),
    )
    precomputed_exact = all(
        np.array_equal(reference, observed) for reference, observed in matching_arrays
    )
    if not precomputed_exact:
        raise RuntimeError("precomputed-target matching differs from the reference path")

    ordering = _backend.descending_order_profiled(prediction_array)
    ordering_exact = np.array_equal(
        ordering["radix_order"], ordering["reference_order"]
    )
    if not ordering_exact:
        raise RuntimeError("radix ordering differs from the reference ordering")

    return {
        "name": str(name),
        "source": str(source),
        "shape": [int(value) for value in prediction_array.shape],
        "voxel_count": int(prediction_array.size),
        "prediction_pair_count": int(matching.prediction.birth_flat.size),
        "target_pair_count": int(matching.target.birth_flat.size),
        "matched_pair_count": int(matching.matched_prediction.size),
        "unmatched_prediction_count": int(matching.unmatched_prediction.size),
        "unmatched_target_count": int(matching.unmatched_target.size),
        "profile_seconds": profile_seconds,
        "precomputed_target_exact": precomputed_exact,
        "precomputed_target_total_seconds": float(precomputed_seconds),
        "ordering_exact": ordering_exact,
        "ordering_seconds": {
            "radix": float(ordering["radix_seconds"]),
            "reference": float(ordering["reference_seconds"]),
        },
    }


def _extract_patch(
    instances: np.ndarray,
    center_zyx: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center_zyx, dtype=np.int64)
    patch_size = np.asarray(patch_shape, dtype=np.int64)
    lower = center - patch_size // 2
    upper = lower + patch_size
    source_lower = np.maximum(lower, 0)
    source_upper = np.minimum(upper, np.asarray(instances.shape, dtype=np.int64))
    destination_lower = source_lower - lower
    destination_upper = destination_lower + (source_upper - source_lower)
    patch = np.zeros(patch_shape, dtype=np.int32)
    patch[
        tuple(slice(int(a), int(b)) for a, b in zip(destination_lower, destination_upper))
    ] = instances[
        tuple(slice(int(a), int(b)) for a, b in zip(source_lower, source_upper))
    ]
    return patch, lower


def cached_target_field(
    case_directory: Path,
    patch_shape: tuple[int, int, int] = PATCH_SHAPE,
) -> np.ndarray:
    case_directory = Path(case_directory)
    with np.load(case_directory / "instances.npz") as archive:
        instances = np.asarray(archive["instance_ids"], dtype=np.int32)
    metadata = json.loads(
        (case_directory / "metadata.json").read_text(encoding="utf-8")
    )
    instance_metadata = metadata.get("instances", [])
    if not instance_metadata:
        raise RuntimeError(f"cached case {case_directory.name} has no instances")
    anchor = max(
        instance_metadata,
        key=lambda item: (int(item["voxel_count"]), -int(item["id"])),
    )
    patch, lower = _extract_patch(
        instances,
        tuple(int(value) for value in anchor["center_zyx"]),
        patch_shape,
    )
    centers = {
        int(item["id"]): tuple(
            float(item["center_zyx"][axis]) - float(lower[axis]) for axis in range(3)
        )
        for item in instance_metadata
    }
    target = build_persistent_field_target(
        patch,
        centers,
        tuple(float(value) for value in metadata["spacing_zyx"]),
        topology_margin=1,
    )
    return target.field.numpy()


def build_profiles(cache_root: Path, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    profiles: list[dict] = []
    for shape in SYNTHETIC_SHAPES:
        profiles.append(
            profile_field_pair(
                "synthetic_" + "x".join(str(value) for value in shape),
                _random_prediction(shape, rng),
                _structured_target(shape),
                source="synthetic",
            )
        )
    cached_cases = select_cached_cases(cache_root, limit=5)
    for case_directory in cached_cases:
        target = cached_target_field(case_directory, PATCH_SHAPE)
        profiles.append(
            profile_field_pair(
                case_directory.name,
                _random_prediction(PATCH_SHAPE, rng),
                target,
                source="cached_target_patch",
            )
        )
    return {
        "schema_version": 1,
        "method": "exact_full_resolution_26_connected_h0_induced_matching",
        "seed": int(seed),
        "patch_shape": list(PATCH_SHAPE),
        "synthetic_profile_count": len(SYNTHETIC_SHAPES),
        "cached_profile_count": len(cached_cases),
        "cached_cases": [case.name for case in cached_cases],
        "profiles": profiles,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default=os.environ.get("MICROBT_PF_CACHE"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.cache_root:
        parser.error("--cache-root or MICROBT_PF_CACHE is required")

    isolated_root = Path(
        os.environ.get("MICROBT_PF_ISOLATED_ROOT", "/path/to/example")
    ).resolve()
    artifact_root = (isolated_root / "artifacts").resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(artifact_root):
        parser.error(f"--output must be under {artifact_root}")

    payload = build_profiles(Path(args.cache_root), seed=args.seed)
    atomic_write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
