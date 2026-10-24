from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nnunetv2.scripts.microbt_pf_scheduled_exact_gate import (
    EXPECTED_SCHEDULE,
    evaluate_scheduled_exact_gate,
)
from nnunetv2.evaluation.microbt_pf_fixed_monitor import (
    validate_fixed_case_manifest,
)


TRAINER_NAME = "MicroBTPFTrainerA3ScheduledExact"
DEFAULT_AUGMENTATION_PROCESSES = 8
DATASET_NAME = "Dataset001_Example"
PLANS_NAME = "nnUNetResEncUNetMPlans"
CONFIGURATION_NAME = "3d_fullres"
PREPROCESSED_ROOT = Path(
    "/path/to/example"
)
RAW_ROOT = Path("/path/to/example")


def resolve_inside(root: Path, candidate: Path, label: str) -> Path:
    resolved_root = Path(root).resolve()
    resolved_candidate = Path(candidate).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} is outside isolated root") from error
    return resolved_candidate


def validate_admission_payload(payload: dict, current_head: str) -> None:
    if payload.get("schema_version") != 1:
        raise RuntimeError("admission schema mismatch")
    if payload.get("git_head") != current_head:
        raise RuntimeError("admission commit mismatch")
    if payload.get("schedule") != EXPECTED_SCHEDULE:
        raise RuntimeError("admission schedule mismatch")
    recorded = payload.get("admission", {})
    if recorded.get("admitted") is not True or recorded.get("reasons") != []:
        raise RuntimeError("admission artifact is not admitted")
    recomputed = evaluate_scheduled_exact_gate(payload)
    if recomputed != {"admitted": True, "reasons": []}:
        raise RuntimeError(
            "admission artifact failed recomputation: "
            + ",".join(recomputed["reasons"])
        )


def fold_zero_identifiers(split_file: Path) -> tuple[str, ...]:
    split_file = Path(split_file)
    if not split_file.is_file():
        raise RuntimeError(f"fold-0 split file is missing: {split_file}")
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(splits, list) or not splits:
        raise RuntimeError("fold-0 split file has no splits")
    fold_zero = splits[0]
    try:
        identifiers = tuple(
            sorted(
                {str(value) for value in fold_zero["train"]}
                | {str(value) for value in fold_zero["val"]}
            )
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("fold-0 split has no train/val identifiers") from error
    if not identifiers:
        raise RuntimeError("fold-0 split is empty")
    return identifiers


def validate_metadata_cache(
    cache_directory: Path, identifiers: tuple[str, ...]
) -> dict:
    cache_directory = Path(cache_directory).resolve()
    missing = [
        identifier
        for identifier in identifiers
        if not (cache_directory / identifier / "metadata.json").is_file()
        or not (cache_directory / identifier / "instances.npz").is_file()
    ]
    if missing:
        preview = ",".join(missing[:5])
        raise RuntimeError(
            f"metadata cache is incomplete ({len(missing)} missing): {preview}"
        )
    return {
        "case_count": len(identifiers),
        "cache_directory": str(cache_directory),
    }


def validate_monitor_manifest(
    monitor_manifest: Path, split_file: Path
) -> dict[str, int | str]:
    monitor_manifest = Path(monitor_manifest)
    split_file = Path(split_file)
    if not monitor_manifest.is_file():
        raise RuntimeError("fixed-case monitor manifest is missing")
    if not split_file.is_file():
        raise RuntimeError("fixed-case split file is missing")
    manifest = json.loads(monitor_manifest.read_text(encoding="utf-8"))
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    fold = int(manifest.get("fold", -1))
    if not isinstance(splits, list) or fold < 0 or fold >= len(splits):
        raise RuntimeError("fixed-case split does not contain the requested fold")
    if manifest.get("dataset_name") != DATASET_NAME:
        raise RuntimeError("fixed-case dataset mismatch")
    inference = manifest.get("inference")
    if inference != {
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
    }:
        raise RuntimeError("fixed-case inference contract mismatch")
    contract = validate_fixed_case_manifest(
        manifest,
        split_sha256=_sha256(split_file),
        validation_subjects=set(str(item) for item in splits[fold]["val"]),
    )
    return {
        "case_count": len(contract["case_ids"]),
        "interval_epochs": int(contract["milestone_every_epochs"]),
        "sha256": _sha256(monitor_manifest),
    }


def build_foreground_command(repo_root: Path) -> list[str]:
    return [
        "bash",
        str(Path(repo_root) / "scripts" / "run_microbt_pf_scheduled_exact_fold0.sh"),
    ]


def build_launch_manifest(
    *,
    pid: int,
    git_head: str,
    admission_artifact: Path,
    admission_sha256: str,
    results_root: Path,
    log_path: Path,
    command: list[str],
    launched_at: str,
    monitor_manifest: Path,
    monitor_sha256: str,
    monitor_output: Path,
) -> dict:
    return {
        "schema_version": 1,
        "status": "launched",
        "pid": int(pid),
        "git_head": git_head,
        "trainer": TRAINER_NAME,
        "num_epochs": 200,
        "admission_artifact": str(admission_artifact),
        "admission_sha256": admission_sha256,
        "results_root": str(results_root),
        "log_path": str(log_path),
        "monitor_manifest": str(monitor_manifest),
        "monitor_sha256": str(monitor_sha256),
        "monitor_output": str(monitor_output),
        "monitor_interval_epochs": 25,
        "command": list(command),
        "launched_at": launched_at,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_and_validate(admission_artifact: Path, current_head: str) -> dict:
    if not admission_artifact.is_file():
        raise RuntimeError("admission artifact is missing")
    payload = json.loads(admission_artifact.read_text(encoding="utf-8"))
    validate_admission_payload(payload, current_head)
    return payload


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch admitted MicroBT-PF scheduled-exact fold-0 training"
    )
    parser.add_argument(
        "--isolated-root",
        type=Path,
        default=Path("/path/to/example"),
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--admission-artifact", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--monitor-manifest", type=Path)
    parser.add_argument("--monitor-output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    isolated_root = Path(args.isolated_root).resolve()
    repo_root = resolve_inside(
        isolated_root,
        args.repo_root or isolated_root / "nnUNet",
        "repository",
    )
    admission_artifact = resolve_inside(
        isolated_root,
        args.admission_artifact
        or isolated_root / "artifacts" / "scheduled_exact_monitored_gate.json",
        "admission artifact",
    )
    results_root = resolve_inside(
        isolated_root,
        args.results_root
        or isolated_root / "nnUNet_results_scheduled_exact_200_monitored",
        "results root",
    )
    log_path = resolve_inside(
        isolated_root,
        args.log_path
        or isolated_root / "logs" / "scheduled_exact_200_monitored_fold0.log",
        "log path",
    )
    manifest_path = resolve_inside(
        isolated_root,
        args.manifest
        or isolated_root / "artifacts" / "scheduled_exact_monitored_launch.json",
        "launch manifest",
    )
    monitor_manifest = resolve_inside(
        isolated_root,
        args.monitor_manifest or repo_root / "monitor" / "fixed_cases_fold0.json",
        "monitor manifest",
    )
    default_monitor_output = (
        results_root
        / DATASET_NAME
        / f"{TRAINER_NAME}__{PLANS_NAME}__{CONFIGURATION_NAME}"
        / "fold_0"
        / "monitor"
    )
    monitor_output = resolve_inside(
        results_root,
        args.monitor_output or default_monitor_output,
        "monitor output",
    )
    current_head = _git_head(repo_root)
    _load_and_validate(admission_artifact, current_head)
    dataset_directory = PREPROCESSED_ROOT / DATASET_NAME
    monitor_contract = validate_monitor_manifest(
        monitor_manifest, dataset_directory / "splits_final.json"
    )
    identifiers = fold_zero_identifiers(dataset_directory / "splits_final.json")
    metadata_cache = resolve_inside(
        isolated_root,
        isolated_root
        / "cache"
        / DATASET_NAME
        / PLANS_NAME
        / CONFIGURATION_NAME,
        "metadata cache",
    )
    validate_metadata_cache(metadata_cache, identifiers)
    if args.verify_only:
        print(
            json.dumps(
                {
                    "admitted": True,
                    "git_head": current_head,
                    "admission_sha256": _sha256(admission_artifact),
                    "monitor_sha256": monitor_contract["sha256"],
                    "monitor_case_count": monitor_contract["case_count"],
                },
                sort_keys=True,
            )
        )
        return 0

    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _pid_is_alive(int(previous.get("pid", -1))):
            raise RuntimeError("an admitted scheduled-exact process is already alive")
    if results_root.is_dir() and next(results_root.rglob("checkpoint_*.pth"), None):
        raise RuntimeError("scheduled-exact results already contain a checkpoint")

    results_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_foreground_command(repo_root)
    environment = os.environ.copy()
    environment.update(
        {
            "MICROBT_PF_ISOLATED_ROOT": str(isolated_root),
            "MICROBT_PF_SCHEDULED_ADMISSION": str(admission_artifact),
            "MICROBT_PF_CACHE": str(isolated_root / "cache"),
            "MICROBT_PF_PYTHON": sys.executable,
            "nnUNet_preprocessed": str(PREPROCESSED_ROOT),
            "nnUNet_raw": str(RAW_ROOT),
            "nnUNet_results": str(results_root),
            "MICROBT_PF_MONITOR_MANIFEST": str(monitor_manifest),
            "MICROBT_PF_MONITOR_OUTPUT": str(monitor_output),
            "PYTHONPATH": str(repo_root)
            + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES", "0"),
            "nnUNet_n_proc_DA": environment.get(
                "nnUNet_n_proc_DA", str(DEFAULT_AUGMENTATION_PROCESSES)
            ),
        }
    )
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(5.0)
    if process.poll() is not None:
        raise RuntimeError(
            f"scheduled-exact training exited during launch with {process.returncode}"
        )
    manifest = build_launch_manifest(
        pid=process.pid,
        git_head=current_head,
        admission_artifact=admission_artifact,
        admission_sha256=_sha256(admission_artifact),
        results_root=results_root,
        log_path=log_path,
        command=command,
        launched_at=datetime.now(timezone.utc).isoformat(),
        monitor_manifest=monitor_manifest,
        monitor_sha256=str(monitor_contract["sha256"]),
        monitor_output=monitor_output,
    )
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
