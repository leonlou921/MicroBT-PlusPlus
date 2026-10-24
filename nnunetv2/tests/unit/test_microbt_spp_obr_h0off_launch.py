from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_h0off_launcher_contract_is_continuous_and_causal() -> None:
    launcher = ROOT / "scripts" / "run_microbt_spp_obr_h0off_causal60.sh"
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
    assert json.loads(result.stdout) == {
        "continuous_process": True,
        "exact_h0": False,
        "formal_epochs": 60,
        "from_scratch": True,
        "full203": False,
        "isolated_root": "/path/to/example",
        "random_seed": 42,
        "snapshot_epochs": [15, 30, 35, 40, 45, 50, 55, 60],
        "trainer": "MicroBTSPPOBRH0OffCausalTrainerScratch60",
    }
    text = launcher.read_text(encoding="utf-8")
    assert text.count("run_phase_training_stage.py") == 1
    assert "--continue-training" not in text
    assert "pretrained" not in text.lower()
    assert "run_microbt_pf_full203.py" not in text
    assert "for epoch in 15 30 35 40 45 50 55 60" in text

