from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_sbpf_launcher_is_isolated_warmstarted_and_chains_full203() -> None:
    text = (ROOT / "scripts" / "run_microbt_sbpf_fold0.sh").read_text(encoding="utf-8")

    assert "/path/to/example" in text
    assert "/path/to/example" in text
    assert "MicroBTSBPFTrainerScheduledExact" in text
    assert "-pretrained_weights" in text
    assert "microbt_sbpf_checkpoint" in text
    assert "launch_microbt_pf_fast_full203.py" in text
    assert "checkpoint_best.pth" in text
    assert "checkpoint_final.pth" in text
    assert "full203_paired" in text
    assert "MICROBT_PF_MONITOR_MANIFEST" in text
    assert 'export MICROBT_PF_CACHE="${source_root}/cache"' in text


def test_full203_audit_hash_covers_sbpf_decoder_and_reports_live_git_head() -> None:
    evaluator = (ROOT / "nnunetv2" / "scripts" / "run_microbt_pf_full203.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "launch_microbt_pf_fast_full203.py").read_text(
        encoding="utf-8"
    )

    assert "nnunetv2/inference/microbt_pf_export.py" in evaluator
    assert "nnunetv2/topology/persistent_field_0d.py" in evaluator
    assert '"training_git_head": _git_head(repo)' in launcher


def test_stable_launcher_gates_before_formal_training_and_full203() -> None:
    text = (ROOT / "scripts" / "run_microbt_sbpf_stable_fold0.sh").read_text(
        encoding="utf-8"
    )

    assert "MicroBTSBPFTrainerStableGate5" in text
    assert "MicroBTSBPFTrainerStableScheduledExact" in text
    assert "nnUNet_results_sbpf_stable_gate5" in text
    assert "nnUNet_results_sbpf_stable_200" in text
    assert "full203_paired_sbpf_stable" in text
    assert "microbt_sbpf_checkpoint" in text
    assert "-pretrained_weights" in text
    assert "GATE_CURVE_OK" in text
    assert "GATE_FIXED_CASE_OK" in text
    assert "launch_microbt_pf_fast_full203.py" in text
    assert "active_fold0_trainer()" in text
    assert 'Path(value).name == "nnUNetv2_train"' in text
    assert "ordinary = completed[1:]" in text
    assert "all(math.isfinite(value) and value > 0 for value in row[\"dice\"])" in text
    assert 'reuse_gate="${MICROBT_SBPF_REUSE_GATE:-0}"' in text
    assert 'reusing completed stable gate' in text
    assert '[[ -f "${gate_fold}/checkpoint_final.pth" ]]' in text
    assert text.index("MicroBTSBPFTrainerStableGate5") < text.index(
        "MicroBTSBPFTrainerStableScheduledExact"
    )
    assert text.index("GATE_FIXED_CASE_OK") < text.index(
        'export nnUNet_results="${formal_results}"'
    )
