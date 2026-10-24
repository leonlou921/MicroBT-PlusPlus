from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time

import numpy as np

from nnunetv2.topology.persistent_field_0d import persistence_pairs_0d


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable")


def _available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _structured_field(shape: tuple[int, int, int]) -> np.ndarray:
    field = np.full(shape, -1.0, dtype=np.float32)
    fractions = (
        (0.20, 0.20, 0.20),
        (0.20, 0.75, 0.70),
        (0.50, 0.50, 0.50),
        (0.75, 0.25, 0.70),
        (0.78, 0.78, 0.25),
    )
    for fraction in fractions:
        center = tuple(
            min(shape[axis] - 2, max(1, int(round(fraction[axis] * (shape[axis] - 1)))))
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
                    field[coordinate] = max(field[coordinate], value)
    return field


def _pair_hash(pairs) -> str:
    digest = hashlib.sha256()
    digest.update(pairs.birth_flat.tobytes())
    digest.update(pairs.death_flat.tobytes())
    digest.update(pairs.essential.tobytes())
    return digest.hexdigest()


def benchmark_one_shape(
    shape: tuple[int, int, int], repeats: int = 2
) -> dict:
    if len(shape) != 3 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape must contain three positive integers")
    if repeats < 2:
        raise ValueError("at least two repeats are required for determinism")
    shape = tuple(int(value) for value in shape)
    available_before = _available_memory_bytes()
    baseline_rss = _rss_bytes()
    field = _structured_field(shape)
    peak = [max(baseline_rss, _rss_bytes())]
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.wait(0.005):
            peak[0] = max(peak[0], _rss_bytes())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    hashes: list[str] = []
    wall_times: list[float] = []
    pair_counts: list[int] = []
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            pairs = persistence_pairs_0d(field)
            wall_times.append(time.perf_counter() - started)
            hashes.append(_pair_hash(pairs))
            pair_counts.append(int(pairs.birth_flat.size))
            peak[0] = max(peak[0], _rss_bytes())
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    deterministic = len(set(hashes)) == 1 and len(set(pair_counts)) == 1
    peak_delta = max(0, peak[0] - baseline_rss)
    if not deterministic:
        raise RuntimeError(f"nondeterministic topology result at shape {shape}")
    if peak_delta >= available_before:
        raise RuntimeError(f"topology call exceeded available memory at shape {shape}")
    return {
        "shape": list(shape),
        "voxel_count": int(np.prod(shape)),
        "wall_times_seconds": wall_times,
        "median_wall_time_seconds": float(statistics.median(wall_times)),
        "baseline_rss_bytes": int(baseline_rss),
        "peak_rss_bytes": int(peak[0]),
        "peak_rss_delta_bytes": int(peak_delta),
        "available_memory_before_bytes": int(available_before),
        "pair_count": pair_counts[0],
        "hashes": hashes,
        "deterministic": deterministic,
    }


def _full_validation_shape() -> tuple[int, int, int]:
    explicit = os.environ.get("MICROBT_PF_FULL_VOLUME_SHAPE")
    if explicit:
        values = tuple(int(value) for value in explicit.split(","))
        if len(values) != 3:
            raise ValueError("MICROBT_PF_FULL_VOLUME_SHAPE must be D,H,W")
        return values
    from batchgenerators.utilities.file_and_folder_operations import load_pickle

    preprocessed = Path(os.environ["nnUNet_preprocessed"])
    folder = preprocessed / "Dataset001_Example" / "nnUNetPlans_3d_fullres"
    properties_file = next(iter(sorted(folder.glob("*.pkl"))), None)
    if properties_file is None:
        raise RuntimeError("no Dataset001 validation properties file is available")
    properties = load_pickle(properties_file)
    return tuple(int(value) for value in properties["shape_before_cropping"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    shapes = [
        (32, 32, 32),
        (64, 64, 64),
        (112, 160, 128),
        _full_validation_shape(),
    ]
    results = [benchmark_one_shape(shape, repeats=args.repeats) for shape in shapes]
    payload = {
        "schema_version": 1,
        "backend": "deterministic_26_connected_h0",
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
