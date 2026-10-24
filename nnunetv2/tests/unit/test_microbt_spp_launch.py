from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_spp_launcher_shell_and_public_contract() -> None:
    launcher = ROOT / "scripts" / "run_microbt_spp_fold0.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(launcher)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr

    result = subprocess.run(
        ["bash", str(launcher), "--print-contract"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert contract == {
        "auto_full203": True,
        "formal_epochs": 200,
        "formal_trainer": "MicroBTSPPTrainerScheduledExact",
        "gate_epochs": 5,
        "gate_trainer": "MicroBTSPPTrainerGate5",
        "isolated_root": "/path/to/example",
        "source_checkpoint": "checkpoint_final.pth",
    }
