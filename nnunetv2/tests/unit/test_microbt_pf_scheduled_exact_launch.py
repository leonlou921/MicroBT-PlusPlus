from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from nnunetv2.scripts.launch_microbt_pf_scheduled_exact import (
    DEFAULT_AUGMENTATION_PROCESSES,
    build_foreground_command,
    build_launch_manifest,
    fold_zero_identifiers,
    resolve_inside,
    validate_admission_payload,
    validate_metadata_cache,
    validate_monitor_manifest,
)


def _admitted_payload(head: str = "abc123") -> dict:
    return {
        "schema_version": 1,
        "status": "complete",
        "git_head": head,
        "schedule": {
            "num_epochs": 200,
            "tail_start_epoch": 197,
            "sparse_interval": 10,
            "sparse_h0_weight": 10.0,
            "tail_h0_weight": 1.0,
        },
        "topology_hashes": {"precomputed": "same", "reference": "same"},
        "hundred_step": {
            "steps": 100,
            "schedule_mode": "sparse_exact",
            "exact_h0_active_count": 10,
            "schedule_skip_count": 90,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
        },
        "full_epoch": {
            "training_iterations": 250,
            "validation_iterations": 50,
            "schedule_mode": "sparse_exact",
            "exact_h0_train_calls": 25,
            "schedule_skip_count": 225,
            "validation_exact_h0_calls": 5,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
            "wall_seconds": 330.0,
        },
        "tail_step": {
            "epoch": 197,
            "schedule_mode": "full_exact",
            "effective_h0_weight": 1.0,
            "exact_h0_active_count": 1,
            "schedule_skip_count": 0,
            "nonfinite_steps": 0,
            "optimizer_skips": 0,
            "finite_loss": True,
        },
        "protected_state_unchanged": True,
        "admission": {"admitted": True, "reasons": []},
    }


def test_validate_admission_accepts_current_complete_gate() -> None:
    payload = _admitted_payload()
    validate_admission_payload(payload, current_head="abc123")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(git_head="stale"),
        lambda payload: payload.update(status="running"),
        lambda payload: payload["admission"].update(admitted=False),
        lambda payload: payload.update(protected_state_unchanged=False),
        lambda payload: payload["full_epoch"].update(wall_seconds=361.0),
    ],
)
def test_validate_admission_rejects_stale_or_unadmitted_gate(mutation) -> None:
    payload = _admitted_payload()
    mutation(payload)
    with pytest.raises(RuntimeError, match="admission"):
        validate_admission_payload(payload, current_head="abc123")


def test_resolve_inside_rejects_path_escape(tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    assert resolve_inside(isolated, isolated / "results", "results") == (
        isolated / "results"
    ).resolve()
    with pytest.raises(ValueError, match="outside isolated root"):
        resolve_inside(isolated, tmp_path / "escape", "results")


def test_foreground_command_names_only_scheduled_trainer(tmp_path: Path) -> None:
    command = build_foreground_command(tmp_path)
    assert command == [
        "bash",
        str(tmp_path / "scripts" / "run_microbt_pf_scheduled_exact_fold0.sh"),
    ]


def test_launch_manifest_is_complete_and_frozen(tmp_path: Path) -> None:
    manifest = build_launch_manifest(
        pid=4242,
        git_head="abc123",
        admission_artifact=tmp_path / "gate.json",
        admission_sha256="f" * 64,
        results_root=tmp_path / "results",
        log_path=tmp_path / "train.log",
        command=["bash", "runner.sh"],
        launched_at="2026-08-05T12:00:00+00:00",
        monitor_manifest=tmp_path / "fixed.json",
        monitor_sha256="e" * 64,
        monitor_output=tmp_path / "results" / "fold_0" / "monitor",
    )
    assert manifest == {
        "schema_version": 1,
        "status": "launched",
        "pid": 4242,
        "git_head": "abc123",
        "trainer": "MicroBTPFTrainerA3ScheduledExact",
        "num_epochs": 200,
        "admission_artifact": str(tmp_path / "gate.json"),
        "admission_sha256": "f" * 64,
        "results_root": str(tmp_path / "results"),
        "log_path": str(tmp_path / "train.log"),
        "monitor_manifest": str(tmp_path / "fixed.json"),
        "monitor_sha256": "e" * 64,
        "monitor_output": str(tmp_path / "results" / "fold_0" / "monitor"),
        "monitor_interval_epochs": 25,
        "command": ["bash", "runner.sh"],
        "launched_at": "2026-08-05T12:00:00+00:00",
    }


def test_foreground_script_has_guard_lock_and_exact_training_command() -> None:
    repository = Path(__file__).resolve().parents[3]
    text = (
        repository / "scripts" / "run_microbt_pf_scheduled_exact_fold0.sh"
    ).read_text(encoding="utf-8")
    assert "launch_microbt_pf_scheduled_exact --verify-only" in text
    assert "microbt_pf_guard.py" in text
    assert "microbt_pf_scheduled_exact_train.lock" in text
    assert (
        '1 3d_fullres 0 -tr "${trainer}" -p nnUNetResEncUNetMPlans'
        in text
    )
    assert 'trainer="MicroBTPFTrainerA3ScheduledExact"' in text
    assert 'nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"' in text
    assert 'MICROBT_PF_MONITOR_MANIFEST' in text
    assert 'MICROBT_PF_MONITOR_OUTPUT' in text
    assert 'nnUNet_raw=' in text
    assert DEFAULT_AUGMENTATION_PROCESSES == 8


def test_fold_zero_identifiers_are_complete_and_unique(tmp_path: Path) -> None:
    split = tmp_path / "splits_final.json"
    split.write_text(
        '[{"train": ["case_b", "case_a"], "val": ["case_c", "case_b"]}]',
        encoding="utf-8",
    )
    assert fold_zero_identifiers(split) == ("case_a", "case_b", "case_c")


def test_metadata_cache_preflight_fails_before_launch_when_any_case_is_missing(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    first = cache / "case_a"
    second = cache / "case_b"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "metadata.json").write_text("{}", encoding="utf-8")
    (first / "instances.npz").write_bytes(b"instances-a")
    (second / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata cache is incomplete.*case_b"):
        validate_metadata_cache(cache, ("case_a", "case_b"))

    (second / "instances.npz").write_bytes(b"instances-b")
    assert validate_metadata_cache(cache, ("case_a", "case_b")) == {
        "case_count": 2,
        "cache_directory": str(cache.resolve()),
    }


def test_monitor_manifest_preflight_binds_fold_split_and_decoder(tmp_path: Path) -> None:
    split = tmp_path / "splits_final.json"
    split.write_text(
        json.dumps([{"train": ["train"], "val": ["case-a"]}]),
        encoding="utf-8",
    )
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    manifest = tmp_path / "fixed.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "Dataset001_Example",
                "fold": 0,
                "split_file": "splits_final.json",
                "split_sha256": split_sha,
                "connectivity": 26,
                "small_diameter_mm": 3.0,
                "decoder": {"field_min": 0.25, "foreground_level": 0.0},
                "inference": {
                    "tile_step_size": 0.5,
                    "use_gaussian": True,
                    "use_mirroring": True,
                },
                "milestone_every_epochs": 25,
                "cases": [
                    {
                        "subject_id": "case-a",
                        "stratum": "one/small",
                        "expected_count": 1,
                        "expected_small_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    observed = validate_monitor_manifest(manifest, split)
    assert observed["case_count"] == 1
    assert observed["interval_epochs"] == 25
    assert observed["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()

    split.write_text("[]", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="split"):
        validate_monitor_manifest(manifest, split)
