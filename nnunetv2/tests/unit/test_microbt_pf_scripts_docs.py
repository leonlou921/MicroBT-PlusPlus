from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def sandbox():
    isolated = Path(os.environ.get("MICROBT_PF_TEST_ISOLATED_ROOT", "/path/to/example"))
    isolated.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="script_contract_", dir=isolated))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _base_environment(sandbox: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MICROBT_PF_ISOLATED_ROOT": "/path/to/example",
            "MICROBT_PF_CACHE": str(sandbox / "cache"),
            "nnUNet_results": str(sandbox / "results"),
            "nnUNet_preprocessed": "/path/to/example",
            "MICROBT_PF_PYTHON": "python3",
        }
    )
    return environment


def test_fold0_exits_two_before_training_when_guard_rejects(sandbox: Path) -> None:
    marker = sandbox / "training.marker"
    guard = _executable(sandbox / "guard", "exit 2\n")
    trainer = _executable(sandbox / "train", f"touch {marker!s}\n")
    environment = _base_environment(sandbox)
    environment.update(
        {"MICROBT_PF_GUARD": str(guard), "MICROBT_PF_TRAIN_BIN": str(trainer)}
    )

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_microbt_pf_fold0.sh"), "A3"],
        env=environment,
        cwd=REPO,
    )

    assert completed.returncode == 2
    assert not marker.exists()


def test_fold0_rejects_results_outside_isolated_root_before_marker(
    sandbox: Path,
) -> None:
    marker = sandbox / "training.marker"
    guard = _executable(sandbox / "guard", "exit 0\n")
    trainer = _executable(sandbox / "train", f"touch {marker!s}\n")
    environment = _base_environment(sandbox)
    environment.update(
        {
            "MICROBT_PF_GUARD": str(guard),
            "MICROBT_PF_TRAIN_BIN": str(trainer),
            "nnUNet_results": "/tmp/microbt_pf_forbidden",
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_microbt_pf_fold0.sh"), "A0"],
        env=environment,
        cwd=REPO,
    )
    assert completed.returncode == 2
    assert not marker.exists()


@pytest.mark.parametrize(
    ("variant", "trainer_name"),
    [
        ("A0", "MicroBTPFTrainerA0"),
        ("A1", "MicroBTPFTrainerA1"),
        ("A2", "MicroBTPFTrainerA2"),
        ("A3", "MicroBTPFTrainerA3"),
    ],
)
def test_fold0_invokes_literal_variant_trainer_once(
    sandbox: Path, variant: str, trainer_name: str
) -> None:
    record = sandbox / f"{variant}.args"
    guard = _executable(sandbox / "guard", "exit 0\n")
    trainer = _executable(
        sandbox / "train", f"printf '%s\\n' \"$*\" >> {record!s}\n"
    )
    environment = _base_environment(sandbox)
    environment.update(
        {"MICROBT_PF_GUARD": str(guard), "MICROBT_PF_TRAIN_BIN": str(trainer)}
    )

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_microbt_pf_fold0.sh"), variant],
        env=environment,
        cwd=REPO,
    )

    assert completed.returncode == 0
    lines = record.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"1 3d_fullres 0 -tr {trainer_name} -p nnUNetResEncUNetMPlans"
    ]


def test_a0d_reuses_a0_checkpoint_and_never_invokes_training(sandbox: Path) -> None:
    training_marker = sandbox / "training.marker"
    prediction_record = sandbox / "prediction.args"
    guard = _executable(sandbox / "guard", "exit 0\n")
    trainer = _executable(sandbox / "train", f"touch {training_marker!s}\n")
    predictor = _executable(
        sandbox / "predict", f"printf '%s\\n' \"$*\" > {prediction_record!s}\n"
    )
    a0_model = sandbox / "results" / "A0_model"
    a0_model.mkdir(parents=True)
    (a0_model / "fold_0").mkdir()
    (a0_model / "fold_0" / "checkpoint_final.pth").touch()
    input_folder = sandbox / "input"
    input_folder.mkdir()
    output_folder = sandbox / "a0d_output"
    environment = _base_environment(sandbox)
    environment.update(
        {
            "MICROBT_PF_GUARD": str(guard),
            "MICROBT_PF_TRAIN_BIN": str(trainer),
            "MICROBT_PF_PREDICT_BIN": str(predictor),
            "MICROBT_PF_A0_MODEL_FOLDER": str(a0_model),
            "MICROBT_PF_INFERENCE_INPUT": str(input_folder),
            "MICROBT_PF_INFERENCE_OUTPUT": str(output_folder),
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_microbt_pf_fold0.sh"), "A0-D"],
        env=environment,
        cwd=REPO,
    )

    assert completed.returncode == 0
    assert not training_marker.exists()
    arguments = prediction_record.read_text(encoding="utf-8")
    assert f"-m {a0_model}" in arguments
    assert "--checkpoint checkpoint_final.pth" in arguments
    assert "--a0-decoder" in arguments


def test_cpu_script_disables_cuda_and_runs_pytest_after_guard(sandbox: Path) -> None:
    record = sandbox / "pytest.environment"
    guard = _executable(sandbox / "guard", "exit 0\n")
    pytest_recorder = _executable(
        sandbox / "pytest",
        f"printf 'cuda=%s isolated=%s args=%s\\n' \"${{CUDA_VISIBLE_DEVICES-}}\" \"${{MICROBT_PF_TEST_ISOLATED_ROOT-}}\" \"$*\" > {record!s}\n",
    )
    environment = _base_environment(sandbox)
    environment.pop("MICROBT_PF_TEST_ISOLATED_ROOT", None)
    environment.update(
        {"MICROBT_PF_GUARD": str(guard), "MICROBT_PF_PYTEST_BIN": str(pytest_recorder)}
    )
    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_microbt_pf_cpu_tests.sh")],
        env=environment,
        cwd=REPO,
    )
    assert completed.returncode == 0
    observed = record.read_text(encoding="utf-8")
    assert observed.startswith("cuda= isolated=/path/to/example args=")
    assert "nnunetv2/tests" in observed


def test_causal_matrix_launches_frozen_order_through_fold0(sandbox: Path) -> None:
    record = sandbox / "variants"
    guard = _executable(sandbox / "guard", "exit 0\n")
    fold0 = _executable(
        sandbox / "fold0", f"printf '%s\\n' \"$1\" >> {record!s}\n"
    )
    environment = _base_environment(sandbox)
    environment.update(
        {"MICROBT_PF_GUARD": str(guard), "MICROBT_PF_FOLD0_SCRIPT": str(fold0)}
    )
    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_microbt_pf_causal_matrix.sh")],
        env=environment,
        cwd=REPO,
    )
    assert completed.returncode == 0
    assert record.read_text(encoding="utf-8").splitlines() == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A0-D",
    ]
