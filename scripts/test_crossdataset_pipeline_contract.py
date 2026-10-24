from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolver_supports_four_and_one_channel_cases(tmp_path: Path) -> None:
    module = _load("cross_eval", ROOT / "crossdataset_case_files.py")
    for index in range(4):
        (tmp_path / f"MEN__case_{index:04d}.nii.gz").touch()
    assert len(module.resolve_case_files(tmp_path, "MEN__case")) == 4

    one = tmp_path / "MOLAB__case_0000.nii.gz"
    one.touch()
    assert module.resolve_case_files(tmp_path, "MOLAB__case") == (one,)


def test_resolver_rejects_channel_gaps(tmp_path: Path) -> None:
    module = _load("cross_eval_gap", ROOT / "crossdataset_case_files.py")
    (tmp_path / "GLI__case_0000.nii.gz").touch()
    (tmp_path / "GLI__case_0002.nii.gz").touch()
    try:
        module.resolve_case_files(tmp_path, "GLI__case")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("channel gaps must fail the frozen input contract")


def test_train_and_test_shells_are_dataset_dynamic() -> None:
    train = (ROOT / "run_microbt_crossdataset_c2_final300.sh").read_text()
    test = (ROOT / "run_microbt_crossdataset_test.sh").read_text()
    assert "--dataset \"${dataset_id}\"" in train
    assert "--fold all" in train
    assert "MICROBT_UMCTF_TARGET_EPOCHS=300" in train
    assert "MicroBTUnitMassSACLFinal300Trainer" in train
    assert "total_expected_count" in test
    assert "--total-expected-count \"${total_expected_count}\"" in test
    assert "--expected-count 251" not in test


def test_queue_wait_and_output_audit_are_present() -> None:
    assert (ROOT / "wait_for_queue_complete.py").is_file()
    assert (ROOT / "validate_crossdataset_output.py").is_file()
