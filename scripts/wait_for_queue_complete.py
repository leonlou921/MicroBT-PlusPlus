from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    while True:
        if args.state.is_file():
            state = json.loads(args.state.read_text())
            if state.get("status") == "complete":
                return 0
            if state.get("status") == "blocked":
                raise RuntimeError(f"upstream queue blocked: {state.get('error')}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
