from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(repo: Path, data_root: Path, dataset_name: str) -> dict[str, str]:
    patterns = (
        "nnunetv2/training/nnUNetTrainer/*.py",
        "nnunetv2/training/loss/*.py",
        "nnunetv2/inference/microbt*.py",
        "nnunetv2/evaluation/microbt*.py",
        "scripts/*crossdataset*.py",
        "scripts/run_microbt_crossdataset*.sh",
        "scripts/run_phase_training_stage.py",
    )
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in repo.glob(pattern) if path.is_file())
    pre = data_root / "nnUNet_preprocessed" / dataset_name
    raw = data_root / "nnUNet_raw" / dataset_name
    files.update(
        path
        for path in (
            raw / "dataset.json",
            pre / "dataset_fingerprint.json",
            pre / "nnUNetResEncUNetMPlans.json",
        )
        if path.is_file()
    )
    return {str(path.resolve()): sha256(path) for path in sorted(files)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    current = collect(args.repo.resolve(), args.data_root.resolve(), args.dataset_name)
    if args.mode == "write":
        payload = {"schema_version": 1, "dataset_name": args.dataset_name, "files": current}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        expected = json.loads(args.manifest.read_text())
        if expected.get("dataset_name") != args.dataset_name or expected.get("files") != current:
            raise RuntimeError("cross-dataset source manifest drift")
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
