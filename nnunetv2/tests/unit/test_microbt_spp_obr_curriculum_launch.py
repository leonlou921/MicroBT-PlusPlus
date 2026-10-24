from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_curriculum_launcher_contract_and_resume_chain() -> None:
    launcher = ROOT / "scripts" / "run_microbt_spp_obr_curriculum200.sh"
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
        "auto_full203": True,
        "formal_epochs": 200,
        "from_scratch": True,
        "gate_epochs": 30,
        "isolated_root": "/path/to/example",
        "n_proc_da": 8,
        "random_seed": 42,
        "resume_same_optimizer": True,
        "trainer": "MicroBTSPPOBRCurriculumTrainerScratch200",
    }

    text = launcher.read_text(encoding="utf-8")
    assert "refusing to overwrite" in text
    assert "active_fold0_trainer" in text
    assert "MICROBT_CURRICULUM_GATE_EPOCHS=30" in text
    assert ' -c' in text
    assert text.count("run_microbt_pf_full203.py") == 1
    assert "-pretrained_weights" not in text
