#!/usr/bin/env python3
"""Build or verify the sanitized source manifest; no external data is read."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/source_manifest_umctf_b0_sacl_c2.json'
SUFFIXES = {'.py', '.sh', '.cpp', '.cu'}


def snapshot() -> dict[str, str]:
    paths = sorted(path for directory in ('nnunetv2', 'scripts')
                   for path in (ROOT / directory).rglob('*')
                   if path.is_file() and path.suffix in SUFFIXES)
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('write', 'verify'))
    args = parser.parse_args()
    actual = snapshot()
    if args.mode == 'write':
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        payload = {'release_type': 'sanitized-public-source', 'files': actual}
        OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'status': 'passed', 'file_count': len(actual)}))
        return 0
    recorded = json.loads(OUTPUT.read_text())['files']
    drift = sorted(name for name, expected in recorded.items() if actual.get(name) != expected)
    extra = sorted(set(actual) - set(recorded))
    status = 'passed' if not drift and not extra else 'failed'
    print(json.dumps({'status': status, 'file_count': len(actual), 'drift': drift, 'extra': extra}))
    return 0 if status == 'passed' else 3


if __name__ == '__main__':
    raise SystemExit(main())
