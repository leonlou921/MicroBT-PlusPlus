#!/usr/bin/env python3
"""Read-only safety gate for isolated MicroBT-PF work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


ISOLATED_ROOT = "/path/to/example"
PROTECTED_ROOTS = (
    "/path/to/example",
    "/path/to/example",
    "/path/to/example",
    "/path/to/example",
    "/path/to/example",
    "/path/to/example",
    "/path/to/example",
    "/path/to/example",
)
RESOURCE_MODES = frozenset(("cpu-test", "gpu-smoke", "gpu-gate"))
SOURCE_REPOSITORIES = ("/path/to/example",)
MIN_AVAILABLE_MEMORY_BYTES = 8 * 1024**3
MAX_LOAD_PER_CPU = 0.75


def _canonical_posix(path: str) -> str:
    if not path.startswith("/"):
        return ""
    return posixpath.normpath(path)


def _inside(path: str, root: str) -> bool:
    canonical_path = _canonical_posix(path)
    canonical_root = _canonical_posix(root)
    return canonical_path == canonical_root or canonical_path.startswith(
        canonical_root + "/"
    )


def _source_drifted(baseline: dict[str, Any], current: dict[str, Any]) -> bool:
    baseline_sources = baseline.get("sources", {})
    current_sources = current.get("sources", {})
    for source, expected in baseline_sources.items():
        observed = current_sources.get(source)
        if observed is None:
            return True
        for key in ("head", "status_sha256", "dirty_count"):
            if observed.get(key) != expected.get(key):
                return True
    return False


def assess_safety(
    baseline: dict[str, Any],
    current: dict[str, Any],
    destination: str,
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not _inside(destination, ISOLATED_ROOT):
        reasons.append("destination_outside_isolated_root")
    if any(_inside(destination, root) for root in PROTECTED_ROOTS):
        reasons.append("destination_inside_protected_root")
    if _source_drifted(baseline, current):
        reasons.append("protected_source_drift")
    if mode in RESOURCE_MODES:
        if current.get("protected_processes") or current.get("tmux_sessions"):
            reasons.append("protected_workload_active")
        gpu = current.get("gpu", {})
        if gpu.get("compute_processes") or int(gpu.get("memory_used_mib", 0)) > 1024:
            reasons.append("gpu_occupied")
        host = current.get("host", {})
        cpu_count = max(1, int(host.get("cpu_count", 1)))
        if float(host.get("load_1m", 0.0)) / cpu_count > MAX_LOAD_PER_CPU:
            reasons.append("host_load_high")
        if int(host.get("memory_available_bytes", 0)) < MIN_AVAILABLE_MEMORY_BYTES:
            reasons.append("host_memory_low")
    return {"mode": mode, "reasons": reasons, "safe": not reasons}


def _run_text(command: list[str], tolerate_failure: bool = False) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        if tolerate_failure:
            return ""
        raise RuntimeError(
            f"Read-only probe failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _hostname() -> str:
    return socket.gethostname()


def _cpu_count() -> int:
    return int(os.cpu_count() or 1)


def _load_1m() -> float:
    return float(os.getloadavg()[0])


def _memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _process_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.strip().split(None, 7)
        if len(parts) != 8:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "start_time": " ".join(parts[2:7]),
                "command": parts[7],
            }
        )
    return rows


def _self_and_ancestor_pids(text: str, current_pid: int) -> set[int]:
    rows_by_pid = {row["pid"]: row for row in _process_rows(text)}
    excluded: set[int] = set()
    pid = current_pid
    while pid not in excluded:
        row = rows_by_pid.get(pid)
        if row is None:
            break
        if pid != current_pid and "microbt_pf_guard.py" not in row["command"]:
            break
        excluded.add(pid)
        pid = row["ppid"]
    return excluded


def _parse_protected_processes(
    text: str, excluded_pids: set[int] | frozenset[int] = frozenset()
) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for row in _process_rows(text):
        if row["pid"] in excluded_pids:
            continue
        if not any(root in row["command"] for root in PROTECTED_ROOTS):
            continue
        processes.append(row)
    return processes


def _parse_gpu(gpu_text: str, process_text: str) -> dict[str, Any]:
    memory_used_mib = 0
    utilization_percent = 0
    first_gpu = next((line for line in gpu_text.splitlines() if line.strip()), "")
    if first_gpu:
        values = [value.strip() for value in first_gpu.split(",")]
        if len(values) >= 2:
            memory_used_mib = int(values[0])
            utilization_percent = int(values[1])
    compute_processes: list[dict[str, int]] = []
    for line in process_text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 2 or not values[0].isdigit() or not values[1].isdigit():
            continue
        compute_processes.append(
            {"pid": int(values[0]), "used_memory_mib": int(values[1])}
        )
    return {
        "memory_used_mib": memory_used_mib,
        "utilization_percent": utilization_percent,
        "compute_processes": compute_processes,
    }


def collect_live_snapshot() -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for source in SOURCE_REPOSITORIES:
        head = _run_text(["git", "-C", source, "rev-parse", "HEAD"]).strip()
        status = _run_text(
            ["git", "-C", source, "status", "--porcelain=v1"]
        )
        sources[source] = {
            "head": head,
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "dirty_count": len([line for line in status.splitlines() if line]),
        }

    tmux_text = _run_text(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        tolerate_failure=True,
    )
    process_text = _run_text(
        ["ps", "-eo", "pid=,ppid=,lstart=,args="],
    )
    gpu_text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        tolerate_failure=True,
    )
    gpu_process_text = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        tolerate_failure=True,
    )
    return {
        "schema_version": 1,
        "captured_at": _now_iso(),
        "hostname": _hostname(),
        "sources": sources,
        "tmux_sessions": [line.strip() for line in tmux_text.splitlines() if line.strip()],
        "protected_processes": _parse_protected_processes(
            process_text,
            excluded_pids=_self_and_ancestor_pids(process_text, os.getpid()),
        ),
        "host": {
            "cpu_count": _cpu_count(),
            "load_1m": _load_1m(),
            "memory_available_bytes": _memory_available_bytes(),
        },
        "gpu": _parse_gpu(gpu_text, gpu_process_text),
    }


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _verify(args: argparse.Namespace) -> int:
    baseline = _load_json(args.baseline)
    current = (
        collect_live_snapshot()
        if args.current_state is None
        else _load_json(args.current_state)
    )
    result = assess_safety(baseline, current, args.destination, args.mode)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["safe"] else 2


def _snapshot(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect_live_snapshot(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="Write a read-only live snapshot")
    snapshot.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify", help="Evaluate a safety snapshot")
    verify.add_argument("--baseline", required=True)
    verify.add_argument("--current-state")
    verify.add_argument(
        "--mode",
        choices=("code-write", "cpu-test", "gpu-smoke", "gpu-gate"),
        required=True,
    )
    verify.add_argument("--destination", default=f"{ISOLATED_ROOT}/nnUNet")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        return _snapshot(args)
    if args.command == "verify":
        return _verify(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
