#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TRAINER = "MicroBTPFTrainerA3ScheduledExact"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process_command_lines() -> list[str]:
    commands: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return commands
    for item in proc.iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            raw = (item / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command:
            commands.append(command)
    return commands


def validate_training_preflight(
    training_log: Path,
    checkpoint_best: Path,
    checkpoint_final: Path,
    active_command_lines: list[str],
) -> dict[str, Any]:
    if not training_log.is_file() or "Training done." not in training_log.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise RuntimeError("training log does not contain Training done.")
    active = [command for command in active_command_lines if TRAINER in command]
    if active:
        raise RuntimeError("active MicroBT-PF trainer process still exists")
    for path in (checkpoint_best, checkpoint_final):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"checkpoint is missing or empty: {path}")
    return {
        "completed_epoch": 200,
        "checkpoint_best_size": checkpoint_best.stat().st_size,
        "checkpoint_final_size": checkpoint_final.stat().st_size,
        "final_eight_case_monitor": "skipped_by_user_contract",
    }


def _stable(path: Path, seconds: float) -> dict[str, int]:
    first = path.stat()
    time.sleep(max(0.0, float(seconds)))
    second = path.stat()
    before = (first.st_size, first.st_mtime_ns)
    after = (second.st_size, second.st_mtime_ns)
    if before != after:
        raise RuntimeError(f"checkpoint changed during stability observation: {path}")
    return {"size": int(second.st_size), "mtime_ns": int(second.st_mtime_ns)}


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != _sha256(source):
            raise RuntimeError(f"existing frozen artifact differs: {destination}")
    else:
        try:
            subprocess.run(
                ["cp", "--reflink=auto", "--preserve=timestamps", str(source), str(destination)],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            shutil.copy2(source, destination)
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"frozen artifact hash mismatch: {destination}")
    return {"path": str(destination), "size": destination.stat().st_size, "sha256": destination_hash}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return int(pid) > 0


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch post-training MicroBT-PF full203 evaluation")
    parser.add_argument("--isolated-root", type=Path, default=Path("/path/to/example"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("python3"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--stability-seconds", type=float, default=10.0)
    parser.add_argument("--admission-subject")
    parser.add_argument("--compatible-evaluator-sha256", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    isolated = arguments.isolated_root.resolve()
    repo = arguments.repo_root.resolve()
    output = arguments.output.resolve()
    manifest = arguments.manifest.resolve()
    for label, path in (("repository", repo), ("output", output), ("manifest", manifest)):
        try:
            path.relative_to(isolated)
        except ValueError as error:
            raise RuntimeError(f"{label} is outside isolated root") from error
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        pid = int(existing.get("pid", 0))
        if existing.get("status") == "launched" and _pid_alive(pid):
            print(json.dumps({"status": "already_running", "pid": pid}))
            return 0
        status_path = output / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text()).get("state") == "complete":
            print(json.dumps({"status": "already_complete", "pid": pid}))
            return 0

    fold_root = arguments.model_folder.resolve() / f"fold_{arguments.fold}"
    best = fold_root / "checkpoint_best.pth"
    final = fold_root / "checkpoint_final.pth"
    preflight = validate_training_preflight(
        arguments.training_log.resolve(), best, final, _process_command_lines()
    )
    preflight["best_stability"] = _stable(best, arguments.stability_seconds)
    preflight["final_stability"] = _stable(final, arguments.stability_seconds)

    frozen_model = output / "checkpoints" / "model"
    frozen_fold = frozen_model / f"fold_{arguments.fold}"
    metadata = {
        name: _copy_verified(arguments.model_folder / name, frozen_model / name)
        for name in ("dataset.json", "dataset_fingerprint.json", "plans.json")
    }
    frozen = {
        "best": _copy_verified(best, frozen_fold / "checkpoint_best.pth"),
        "final": _copy_verified(final, frozen_fold / "checkpoint_final.pth"),
    }
    _atomic_json(output / "checkpoints" / "manifest.json", {"metadata": metadata, "checkpoints": frozen})

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "execution.log"
    command = [
        str(arguments.python),
        "-m",
        "nnunetv2.scripts.run_microbt_pf_full203",
        "--model-folder",
        str(frozen_model),
        "--split",
        str(arguments.split.resolve()),
        "--output",
        str(output),
        "--fold",
        str(arguments.fold),
        "--checkpoint-best-sha256",
        frozen["best"]["sha256"],
        "--checkpoint-final-sha256",
        frozen["final"]["sha256"],
    ]
    if arguments.admission_subject:
        command.extend(["--admission-subject", arguments.admission_subject])
    for evaluator_hash in arguments.compatible_evaluator_sha256:
        command.extend(["--compatible-evaluator-sha256", str(evaluator_hash)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo) + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.setdefault("nnUNet_preprocessed", "/path/to/example")
    environment.setdefault("nnUNet_raw", "/path/to/example")
    environment.setdefault("nnUNet_results", str(isolated / "nnUNet_results_scheduled_exact_200_monitored"))
    with log_path.open("ab", buffering=0) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = {
        "schema_version": 1,
        "status": "launched",
        "pid": process.pid,
        "launched_at": time.time(),
        "preflight": preflight,
        "checkpoints": frozen,
        "model_folder": str(frozen_model),
        "output": str(output),
        "log": str(log_path),
        "command": command,
        "subject_count": 203,
        "order": ["checkpoint_best", "checkpoint_final", "paired_analysis"],
        "epoch_200_eight_case_monitor": "skipped",
        "training_git_head": _git_head(repo),
        "evaluator_git_head": _git_head(repo),
    }
    _atomic_json(manifest, launch)
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f"full203 evaluator exited during launch with code {process.returncode}")
    print(json.dumps({"status": "launched", "pid": process.pid, "manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
