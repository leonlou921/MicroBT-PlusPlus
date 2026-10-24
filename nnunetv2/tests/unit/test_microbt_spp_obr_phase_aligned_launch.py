from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase_training_stage import _normalize_fold  # noqa: E402


def test_launcher_contract_is_isolated_staged_and_from_scratch() -> None:
    launcher = ROOT / "scripts" / "run_microbt_spp_obr_phase_aligned200.sh"
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
        "fixed_probe_epochs": [15, 30, 45, 60],
        "fixed_probe_objects": 52,
        "formal_epochs": 200,
        "from_scratch": True,
        "hard_gate_epoch": 60,
        "isolated_root": "/path/to/example",
        "momentum_reset_epoch": 15,
        "phase_learning_rate": 0.003,
        "random_seed": 42,
        "trainer": "MicroBTSPPOBRPhaseAlignedTrainerScratch200",
    }

    text = launcher.read_text(encoding="utf-8")
    assert "refusing to overwrite" in text
    assert "active_fold0_trainer" in text
    assert 'for stage in 15 30 45 60' in text
    assert "check_microbt_spp_obr_phase_aligned_gate.py" in text
    assert "run_phase_training_stage.py" in text
    assert "train_bin" not in text
    assert text.count("run_microbt_pf_full203.py") == 1
    assert "-pretrained_weights" not in text


def test_stage_runner_explicitly_skips_automatic_full_volume_validation() -> None:
    runner = ROOT / "scripts" / "run_phase_training_stage.py"
    result = subprocess.run(
        [sys.executable, str(runner), "--print-contract"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "from_scratch_without_continue": True,
        "performs_actual_validation": False,
        "requires_checkpoint_to_continue": True,
    }
    text = runner.read_text(encoding="utf-8")
    assert "run_training()" in text
    assert "perform_actual_validation" not in text
    assert "pretrained" not in text.lower()


def test_stage_runner_normalizes_numeric_fold_before_trainer_construction() -> None:
    assert _normalize_fold("0") == 0
    assert isinstance(_normalize_fold("0"), int)
    assert _normalize_fold("all") == "all"
