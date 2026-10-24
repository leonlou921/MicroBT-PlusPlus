from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_exit_file(path: str | Path) -> None:
    target = Path(path)
    if not target.is_file():
        raise RuntimeError(f"missing exit file: {target}")
    value = int(target.read_text(encoding="utf-8").strip())
    if value != 0:
        raise RuntimeError(f"{target} reported exit={value}")


def validate_test_summary(path: str | Path, *, expected_count: int = 251) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    subjects = payload.get("subject_ids", [])
    metrics = payload.get("metrics", {})
    if payload.get("status") != "complete" or int(metrics.get("case_count", -1)) != expected_count:
        raise RuntimeError("test summary is incomplete")
    if len(subjects) != expected_count or len(set(subjects)) != expected_count:
        raise RuntimeError("test summary subjects are not exact and unique")
    if int(metrics.get("nonfinite_case_count", -1)) != 0:
        raise RuntimeError("test summary contains nonfinite cases")
    if int(payload.get("prediction_file_count", -1)) != expected_count:
        raise RuntimeError("test summary does not contain all saved predictions")


def _wait_for_exit(path: Path, poll_seconds: int, state_path: Path, state: dict[str, Any]) -> None:
    while not path.is_file():
        state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
        state["waiting_for"] = str(path)
        _atomic_json(state_path, state)
        time.sleep(poll_seconds)
    validate_exit_file(path)


def _validate_step(step: dict[str, Any]) -> None:
    if step.get("exit_file"):
        validate_exit_file(step["exit_file"])
    if step.get("summary"):
        validate_test_summary(step["summary"], expected_count=int(step.get("expected_count", 251)))


def run_queue(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_path = Path(config["state_path"])
    state: dict[str, Any] = {
        "schema_version": 1,
        "queue_id": config["queue_id"],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_steps": [],
    }
    _atomic_json(state_path, state)
    for index, step in enumerate(config["steps"]):
        state.update({"active_step": step["name"], "active_index": index, "waiting_for": None})
        _atomic_json(state_path, state)
        free_gib = shutil.disk_usage(config["disk_path"]).free / 1024**3
        if free_gib < float(step.get("min_free_gib", config.get("min_free_gib", 4))):
            raise RuntimeError(f"disk safety gate failed before {step['name']}: {free_gib:.2f} GiB")
        if step["type"] == "wait_exit":
            _wait_for_exit(Path(step["path"]), int(config.get("poll_seconds", 300)), state_path, state)
        elif step["type"] == "command":
            log_path = Path(step["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update({str(k): str(v) for k, v in step.get("env", {}).items()})
            with log_path.open("ab", buffering=0) as stream:
                completed = subprocess.run(
                    step["command"], cwd=step.get("cwd"), env=environment, stdout=stream, stderr=subprocess.STDOUT
                )
            if completed.returncode != 0:
                raise RuntimeError(f"step {step['name']} exited {completed.returncode}")
        else:
            raise ValueError(f"unknown queue step type: {step['type']}")
        _validate_step(step)
        state["completed_steps"].append(step["name"])
        state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(state_path, state)
    state.update({"status": "complete", "active_step": None, "completed_at": datetime.now(timezone.utc).isoformat()})
    _atomic_json(state_path, state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        run_queue(args.config)
    except Exception as error:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        state_path = Path(config["state_path"])
        state = json.loads(state_path.read_text()) if state_path.is_file() else {"queue_id": config["queue_id"]}
        state.update({"status": "blocked", "error": str(error), "blocked_at": datetime.now(timezone.utc).isoformat()})
        _atomic_json(state_path, state)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
