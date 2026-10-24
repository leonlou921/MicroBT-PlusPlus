from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_formal_launcher_hard_locks_eight_workers_after_smoke_pollution() -> None:
    root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["nnUNet_n_proc_DA"] = "2"

    completed = subprocess.run(
        [
            "bash",
            str(root / "scripts" / "run_microbt_umctf_hs_scratch200.sh"),
            "--print-contract",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    contract = json.loads(completed.stdout)

    assert contract["da_workers"] == 8
    assert contract["da_workers_policy"] == "hard_locked"
