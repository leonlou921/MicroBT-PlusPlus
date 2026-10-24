from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run_microbt_umctf_sacl_c2_foldall_scratch200.sh"
LAUNCHER = ROOT / "scripts" / "launch_formal_umctf_sacl_c2_foldall.sh"


def test_foldall_runner_contract_targets_all_training_and_test_evaluation() -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER), "--print-contract"],
        check=True,
        text=True,
        capture_output=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["fold"] == "all"
    assert contract["training_case_count"] == 1011
    assert contract["formal_evaluation_set"] == "test"
    assert contract["test_case_count"] == 251
    assert contract["fixed8"] == "abolished"
    assert contract["architecture_carrier"] == "B0"
    assert contract["causal_change"] == "support_to_atom_continuation_learning"


def test_foldall_runner_and_launcher_cannot_fall_back_to_fold_zero() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "--fold all" in runner
    assert "fold_all" in runner
    assert "--fold 0" not in runner
    assert "nnUNet_results_umctf_b0_sacl_c2_foldall_da8" in runner
    assert "microbt_umctf_b0_sacl_c2_foldall_da8_200" in launcher
    assert "formal_umctf_sacl_c2_foldall_pipeline.exit" in launcher
    assert (
        "run_phase_training_stage.py.*--fold all.*MicroBTUnitMassSACLTrainerScratch60"
        in launcher
    )
