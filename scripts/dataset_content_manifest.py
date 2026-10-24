#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def write_manifest(root: Path, output: Path) -> dict:
    paths = files_under(root)
    files: dict[str, dict[str, int | str]] = {}
    total = len(paths)
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
        if index % 100 == 0 or index == total:
            print(f"hashed {index}/{total}", file=sys.stderr, flush=True)
    payload = {
        "root_at_creation": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files.values()),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return payload


def verify_manifest(root: Path, manifest: Path, allow_extra: bool) -> tuple[int, dict]:
    expected = json.loads(manifest.read_text())
    drift: dict[str, dict[str, object]] = {}
    for index, (relative, reference) in enumerate(expected["files"].items(), 1):
        path = root / relative
        if not path.is_file():
            drift[relative] = {"reason": "missing"}
            continue
        actual_size = path.stat().st_size
        if actual_size != reference["size"]:
            drift[relative] = {
                "reason": "size",
                "expected": reference["size"],
                "actual": actual_size,
            }
            continue
        actual_hash = sha256(path)
        if actual_hash != reference["sha256"]:
            drift[relative] = {
                "reason": "sha256",
                "expected": reference["sha256"],
                "actual": actual_hash,
            }
        if index % 100 == 0 or index == expected["file_count"]:
            print(f"verified {index}/{expected['file_count']}", file=sys.stderr, flush=True)

    actual_relative = {path.relative_to(root).as_posix() for path in files_under(root)}
    extra = sorted(actual_relative - set(expected["files"]))
    status = "passed" if not drift and (allow_extra or not extra) else "failed"
    report = {
        "status": status,
        "expected_file_count": expected["file_count"],
        "drift_count": len(drift),
        "drift": drift,
        "extra_count": len(extra),
        "extra": extra[:100],
        "allow_extra": allow_extra,
    }
    return (0 if status == "passed" else 3), report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--root", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--allow-extra", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if args.command == "write":
        payload = write_manifest(root, args.output.resolve())
        print(json.dumps({
            "status": "passed",
            "file_count": payload["file_count"],
            "total_bytes": payload["total_bytes"],
            "output": str(args.output.resolve()),
        }))
        return 0
    code, report = verify_manifest(root, args.manifest.resolve(), args.allow_extra)
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
