from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "run_microbt_pf_gpu_feasibility.sh"


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def sandbox():
    root = Path(os.environ.get("MICROBT_PF_TEST_ISOLATED_ROOT", "/path/to/example"))
    directory = Path(tempfile.mkdtemp(prefix="gpu_contract_", dir=root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _environment(sandbox: Path, guard: Path, runner: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MICROBT_PF_ISOLATED_ROOT": "/path/to/example",
            "MICROBT_PF_GUARD": str(guard),
            "MICROBT_PF_GPU_RUNNER": str(runner),
            "MICROBT_PF_CACHE": str(sandbox / "cache"),
            "nnUNet_results": str(sandbox / "results"),
            "nnUNet_preprocessed": "/path/to/example",
            "MICROBT_PF_GPU_ARTIFACT": str(sandbox / "gpu.json"),
            "MICROBT_PF_GPU_RUN_ROOT": str(sandbox / "runs"),
            "MICROBT_PF_RUNLOCK_ROOT": str(sandbox / "runlocks"),
        }
    )
    return environment


def test_guard_rejection_exits_before_gpu_runner_or_marker(sandbox: Path) -> None:
    marker = sandbox / "runner.marker"
    guard = _executable(sandbox / "guard", "exit 2\n")
    runner = _executable(sandbox / "runner", f"touch {marker!s}\n")
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env=_environment(sandbox, guard, runner),
    )
    assert completed.returncode == 2
    assert not marker.exists()


def test_guard_acceptance_records_exact_matched_gpu_contract(sandbox: Path) -> None:
    record = sandbox / "runner.args"
    guard = _executable(sandbox / "guard", "exit 0\n")
    runner = _executable(
        sandbox / "runner", f"printf '%s\\n' \"$*\" > {record!s}\n"
    )
    environment = _environment(sandbox, guard, runner)
    completed = subprocess.run(["bash", str(SCRIPT)], cwd=REPO, env=environment)
    assert completed.returncode == 0
    arguments = record.read_text(encoding="utf-8")
    assert "--dataset 1" in arguments
    assert "--configuration 3d_fullres" in arguments
    assert "--plans nnUNetResEncUNetMPlans" in arguments
    assert "--variants A0 A3" in arguments
    assert "--seed 42" in arguments
    assert "--case-count 5" in arguments
    assert f"--a0-output {sandbox / 'runs' / 'A0'}" in arguments
    assert f"--a3-output {sandbox / 'runs' / 'A3'}" in arguments
    assert str(sandbox / "gpu.json") in arguments
    assert (sandbox / "runs" / "A0") != (sandbox / "runs" / "A3")
