from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from nnunetv2.nets.microbt_pf.model import MicroBTPFNetwork
from nnunetv2.topology.persistent_field_0d import (
    InducedMatching0D,
    PersistencePairs0D,
)
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    PersistentTargetTopology,
)
from nnunetv2.training.loss import microbt_pf_losses
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA0,
    MicroBTPFTrainerA1,
    MicroBTPFTrainerA2,
    MicroBTPFTrainerA3,
    MicroBTPFTrainerA3ExactFast,
    MicroBTPFTrainerA3ScheduledExact,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


DATASET_NAME = "Dataset001_Example"
SHAPE = (3, 3, 5)


def _plans_and_dataset() -> tuple[dict, dict]:
    preprocessed = os.environ.get("nnUNet_preprocessed")
    if not preprocessed:
        pytest.skip("nnUNet_preprocessed is unavailable")
    dataset = Path(preprocessed) / DATASET_NAME
    plans_file = dataset / "nnUNetResEncUNetMPlans.json"
    dataset_file = dataset / "dataset.json"
    if not plans_file.is_file() or not dataset_file.is_file():
        pytest.skip("Dataset001 plans are unavailable")
    return (
        json.loads(plans_file.read_text(encoding="utf-8")),
        json.loads(dataset_file.read_text(encoding="utf-8")),
    )


def _flat(z: int, y: int, x: int) -> int:
    return int(np.ravel_multi_index((z, y, x), SHAPE))


def _matching() -> InducedMatching0D:
    prediction = PersistencePairs0D(
        birth_flat=np.asarray([_flat(1, 1, 1)], dtype=np.uint32),
        death_flat=np.asarray([_flat(1, 1, 2)], dtype=np.int64),
        essential=np.asarray([False], dtype=bool),
    )
    target = PersistencePairs0D(
        birth_flat=np.asarray([_flat(1, 1, 2)], dtype=np.uint32),
        death_flat=np.asarray([_flat(1, 1, 3)], dtype=np.int64),
        essential=np.asarray([False], dtype=bool),
    )
    return InducedMatching0D(
        prediction=prediction,
        target=target,
        matched_prediction=np.asarray([0], dtype=np.int64),
        matched_target=np.asarray([0], dtype=np.int64),
        unmatched_prediction=np.empty(0, dtype=np.int64),
        unmatched_target=np.empty(0, dtype=np.int64),
    )


def _persistent_target() -> PersistentFieldTarget:
    field = torch.full(SHAPE, -1.0)
    field[1, 1, 2] = 1.0
    field[1, 1, 3] = -0.4
    instance_ids = torch.zeros(SHAPE, dtype=torch.int32)
    instance_ids[1, 1, 2] = 1
    pairs = _matching().target
    flat = field.numpy().ravel()
    birth_values = flat[pairs.birth_flat.astype(np.int64)].copy()
    death_values = flat[pairs.death_flat].copy()
    return PersistentFieldTarget(
        field=field,
        instance_ids=instance_ids,
        dense_valid_mask=torch.ones(SHAPE, dtype=torch.bool),
        valid_instance_ids=(1,),
        centers_zyx=((1, 1, 2),),
        topology=PersistentTargetTopology(
            pairs=pairs,
            birth_values=birth_values,
            death_values=death_values,
            branch_valid=np.asarray([True], dtype=bool),
        ),
    )


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Conv3d(1, 3, 1, bias=False)
        nn.init.constant_(self.output.weight, 0.15)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output(image)


def _step_trainer(trainer_class):
    trainer = object.__new__(trainer_class)
    trainer.device = torch.device("cpu")
    unet = _TinyUNet()
    trainer.network = unet if trainer_class is MicroBTPFTrainerA0 else MicroBTPFNetwork(unet)
    trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.05)
    trainer.grad_scaler = None
    trainer.loss = trainer._build_loss()
    return trainer, unet


def _step_batch() -> dict:
    data = torch.linspace(-1.0, 1.0, int(np.prod(SHAPE))).reshape(1, 1, *SHAPE)
    target = torch.zeros((1, 3, *SHAPE), dtype=torch.float32)
    target[:, 0, 1, 1, 1:4] = 1.0
    target[:, 1, 1, 1, 2:4] = 1.0
    target[:, 2, 1, 1, 2] = 1.0
    return {
        "data": data,
        "target": target,
        "persistent_targets": (_persistent_target(),),
    }


def test_exact_inheritance_flags_and_official_training_defaults() -> None:
    assert MicroBTPFTrainerA0.__bases__ == (nnUNetTrainer,)
    assert MicroBTPFTrainerA1.__bases__ == (nnUNetTrainer,)
    assert MicroBTPFTrainerA2.__bases__ == (MicroBTPFTrainerA1,)
    assert MicroBTPFTrainerA3.__bases__ == (MicroBTPFTrainerA2,)
    assert MicroBTPFTrainerA3ExactFast.__bases__ == (MicroBTPFTrainerA3,)
    assert MicroBTPFTrainerA3ScheduledExact.__bases__ == (
        MicroBTPFTrainerA3ExactFast,
    )
    assert MicroBTPFTrainerA3ExactFast.autocast_dtype is torch.bfloat16
    assert MicroBTPFTrainerA3ScheduledExact.autocast_dtype is torch.bfloat16
    assert (
        MicroBTPFTrainerA0.conditional_field,
        MicroBTPFTrainerA0.equal_object_field,
        MicroBTPFTrainerA0.h0_matching,
        MicroBTPFTrainerA0.component_sampling,
    ) == (False, False, False, False)
    assert (
        MicroBTPFTrainerA1.conditional_field,
        MicroBTPFTrainerA1.equal_object_field,
        MicroBTPFTrainerA1.h0_matching,
        MicroBTPFTrainerA1.component_sampling,
    ) == (True, True, False, False)
    assert (
        MicroBTPFTrainerA2.conditional_field,
        MicroBTPFTrainerA2.equal_object_field,
        MicroBTPFTrainerA2.h0_matching,
        MicroBTPFTrainerA2.component_sampling,
    ) == (True, True, True, False)
    assert (
        MicroBTPFTrainerA3.conditional_field,
        MicroBTPFTrainerA3.equal_object_field,
        MicroBTPFTrainerA3.h0_matching,
        MicroBTPFTrainerA3.component_sampling,
    ) == (True, True, True, True)
    assert MicroBTPFTrainerA3.configure_optimizers is nnUNetTrainer.configure_optimizers


@pytest.mark.parametrize(
    "trainer_class",
    [
        MicroBTPFTrainerA0,
        MicroBTPFTrainerA1,
        MicroBTPFTrainerA2,
        MicroBTPFTrainerA3,
        MicroBTPFTrainerA3ExactFast,
        MicroBTPFTrainerA3ScheduledExact,
    ],
)
def test_constructor_disables_deep_supervision_before_initialize(trainer_class) -> None:
    plans, dataset_json = _plans_and_dataset()
    trainer = trainer_class(plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu"))
    assert trainer.enable_deep_supervision is False
    assert trainer.num_epochs == (
        200 if trainer_class is MicroBTPFTrainerA3ScheduledExact else 1000
    )
    if trainer_class is MicroBTPFTrainerA3ScheduledExact:
        assert trainer.save_every == 25
    assert trainer.initial_lr == 1e-2
    assert trainer.weight_decay == 3e-5
    assert trainer.num_iterations_per_epoch == 250
    assert trainer.num_val_iterations_per_epoch == 50


def test_nnunet_discovery_and_import_have_no_legacy_runtime_dependency(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    environment["nnUNet_results"] = str(tmp_path / "results")
    environment["nnUNet_raw"] = str(tmp_path / "raw")
    code = f"""
import json, os, sys, torch
from pathlib import Path
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as package
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
dataset = Path(os.environ['nnUNet_preprocessed']) / '{DATASET_NAME}'
plans = json.loads((dataset / 'nnUNetResEncUNetMPlans.json').read_text())
dataset_json = json.loads((dataset / 'dataset.json').read_text())
folder = str(Path(package.__file__).parent)
for name in ('MicroBTPFTrainerA0', 'MicroBTPFTrainerA1', 'MicroBTPFTrainerA2', 'MicroBTPFTrainerA3', 'MicroBTPFTrainerA3ExactFast', 'MicroBTPFTrainerA3ScheduledExact'):
    cls = recursive_find_python_class(folder, name, 'nnunetv2.training.nnUNetTrainer')
    assert cls is not None
    instance = cls(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    assert instance.enable_deep_supervision is False
assert not [name for name in sys.modules if 'microbtpp' in name.lower()]
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "trainer_class",
    [MicroBTPFTrainerA0, MicroBTPFTrainerA1, MicroBTPFTrainerA2, MicroBTPFTrainerA3],
)
def test_one_cpu_optimizer_step_has_exact_loss_contract_and_head_gradients(
    trainer_class,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: _matching(),
    )
    trainer, unet = _step_trainer(trainer_class)
    before = unet.output.weight.detach().clone()

    observed = trainer.train_step(_step_batch())

    assert set(observed) == {"loss", "region", "persistent_field", "field", "betti_0"}
    assert all(np.isfinite(value).all() for value in observed.values())
    assert not torch.equal(before, unet.output.weight.detach())
    assert unet.output.weight.grad is not None
    assert torch.count_nonzero(unet.output.weight.grad[0]) > 0
    assert torch.count_nonzero(unet.output.weight.grad[1]) > 0
    assert torch.count_nonzero(unet.output.weight.grad[2]) > 0
    if trainer_class is MicroBTPFTrainerA0:
        assert observed["persistent_field"] == 0
    elif trainer_class is MicroBTPFTrainerA1:
        assert observed["field"] > 0 and observed["betti_0"] == 0
    else:
        assert observed["field"] > 0 and observed["betti_0"] >= 0


def test_a2_h0_term_backpropagates_only_through_live_critical_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: _matching(),
    )
    trainer, _ = _step_trainer(MicroBTPFTrainerA2)
    logits = torch.zeros((1, 3, *SHAPE), requires_grad=True)
    output = {
        "field_logits": logits,
        "region_probabilities": torch.zeros((1, 3, *SHAPE)),
    }
    values = trainer.loss(output, torch.zeros((1, 3, *SHAPE)), (_persistent_target(),))
    values["betti_0"].backward()
    assert logits.grad is not None
    assert logits.grad[0, 0, 1, 1, 1] != 0
    assert logits.grad[0, 0, 1, 1, 2] != 0
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0


def test_exact_fast_training_matches_a3_and_records_every_exact_h0_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def matching(prediction, target, target_pairs):
        calls.append(int(target_pairs.birth_flat.size))
        return _matching()

    monkeypatch.setattr(
        microbt_pf_losses, "induced_matching_0d_precomputed_target", matching
    )
    a3, _ = _step_trainer(MicroBTPFTrainerA3)
    exact_fast, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    exact_fast.grad_scaler = object()
    exact_fast._initialize_exact_fast_state()
    assert exact_fast.grad_scaler is None

    reference = a3.train_step(_step_batch())
    observed = exact_fast.train_step(_step_batch())

    for key in reference:
        np.testing.assert_array_equal(observed[key], reference[key])
    assert calls == [1, 1]
    telemetry = exact_fast.exact_fast_telemetry
    assert telemetry["exact_h0_train_calls"] == 1
    assert telemetry["nonfinite_steps"] == 0
    assert telemetry["skipped_steps"] == 0
    assert telemetry["target_precompute_wait_seconds"] >= 0
    assert telemetry["topology_wall_seconds"] >= 0
    assert np.isfinite(telemetry["last_preclip_gradient_norm"])


def test_exact_fast_validation_runs_exact_h0_for_first_five_of_fifty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def matching(prediction, target, target_pairs):
        calls.append(1)
        return _matching()

    monkeypatch.setattr(
        microbt_pf_losses, "induced_matching_0d_precomputed_target", matching
    )
    trainer, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    trainer._initialize_exact_fast_state()
    trainer.label_manager = SimpleNamespace(has_ignore_label=False)
    trainer.on_validation_epoch_start()

    with torch.no_grad():
        for _ in range(50):
            trainer.validation_step(_step_batch())

    assert len(calls) == 5
    assert trainer.exact_fast_telemetry["validation_exact_h0_calls"] == 5


def test_exact_fast_validation_precomputes_only_an_audit_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def matching(prediction, target, target_pairs, floor=-1.0):
        calls.append(int(target_pairs.birth_flat.size))
        return _matching()

    monkeypatch.setattr(
        microbt_pf_losses, "induced_matching_0d_precomputed_target", matching
    )
    trainer, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    trainer._initialize_exact_fast_state()
    trainer.label_manager = SimpleNamespace(has_ignore_label=False)
    trainer.on_validation_epoch_start()
    batch = _step_batch()
    object.__setattr__(batch["persistent_targets"][0], "topology", None)

    with torch.no_grad():
        trainer.validation_step(batch)

    assert calls == [1]
    assert batch["persistent_targets"][0].topology is None
    assert trainer.exact_fast_telemetry["validation_exact_h0_calls"] == 1


def test_exact_fast_monitor_validation_accepts_dense_target_without_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_matching(*args, **kwargs):
        raise AssertionError("monitor-only validation must not execute H0")

    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        unexpected_matching,
    )
    trainer, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    trainer._initialize_exact_fast_state()
    trainer.label_manager = SimpleNamespace(has_ignore_label=False)
    trainer.on_validation_epoch_start()
    trainer._exact_fast_validation_index = trainer.validation_h0_audit_iterations
    batch = _step_batch()
    object.__setattr__(batch["persistent_targets"][0], "topology", None)

    with torch.no_grad():
        output = trainer.validation_step(batch)

    assert np.isfinite(output["loss"]).all()
    assert trainer.exact_fast_telemetry["validation_exact_h0_calls"] == 0


def test_exact_fast_fails_closed_when_target_topology_is_missing() -> None:
    trainer, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    trainer._initialize_exact_fast_state()
    batch = _step_batch()
    object.__setattr__(batch["persistent_targets"][0], "topology", None)

    with pytest.raises(RuntimeError, match="precomputed target topology"):
        trainer.train_step(batch)

    assert trainer.exact_fast_telemetry["skipped_steps"] == 1


def test_scheduled_exact_policy_has_25_sparse_calls_and_full_exact_tail() -> None:
    for epoch in (0, 196):
        decisions = [
            MicroBTPFTrainerA3ScheduledExact.exact_h0_schedule(epoch, step)
            for step in range(250)
        ]
        assert [step for step, decision in enumerate(decisions) if decision[1]] == list(
            range(0, 250, 10)
        )
        assert {decision[0] for decision in decisions} == {"sparse_exact"}
        assert {decision[2] for decision in decisions if decision[1]} == {10.0}
        assert {decision[2] for decision in decisions if not decision[1]} == {0.0}

    for epoch in (197, 198, 199):
        decisions = [
            MicroBTPFTrainerA3ScheduledExact.exact_h0_schedule(epoch, step)
            for step in range(250)
        ]
        assert all(decision == ("full_exact", True, 1.0) for decision in decisions)


def test_scheduled_exact_sparse_due_step_weights_h0_by_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: _matching(),
    )
    reference, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    reference._initialize_exact_fast_state()
    scheduled, _ = _step_trainer(MicroBTPFTrainerA3ScheduledExact)
    scheduled.current_epoch = 0
    scheduled._initialize_exact_fast_state()
    scheduled._initialize_scheduled_exact_state()

    exact_values = reference.train_step(_step_batch())
    scheduled_values = scheduled.train_step(_step_batch())

    np.testing.assert_allclose(scheduled_values["region"], exact_values["region"])
    np.testing.assert_allclose(scheduled_values["field"], exact_values["field"])
    np.testing.assert_allclose(
        scheduled_values["betti_0"], exact_values["betti_0"] * 10.0
    )
    np.testing.assert_allclose(
        scheduled_values["persistent_field"],
        scheduled_values["field"] + scheduled_values["betti_0"],
    )
    np.testing.assert_allclose(
        scheduled_values["loss"],
        scheduled_values["region"] + scheduled_values["persistent_field"],
    )
    assert scheduled.exact_fast_telemetry["exact_h0_train_calls"] == 1
    assert scheduled.scheduled_exact_telemetry["schedule_skips"] == 0
    assert scheduled.scheduled_exact_telemetry["effective_h0_weight"] == 10.0
    assert scheduled_values["h0_active"] == 1
    np.testing.assert_allclose(
        scheduled_values["betti_0_raw"], exact_values["betti_0"]
    )


def test_scheduled_exact_sparse_off_step_uses_dense_losses_without_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("off-schedule step must not execute H0")
        ),
    )
    trainer, _ = _step_trainer(MicroBTPFTrainerA3ScheduledExact)
    trainer.current_epoch = 0
    trainer._initialize_exact_fast_state()
    trainer._initialize_scheduled_exact_state()
    trainer._scheduled_train_step_index = 1
    batch = _step_batch()
    object.__setattr__(batch["persistent_targets"][0], "topology", None)

    values = trainer.train_step(batch)

    assert values["betti_0"] == 0
    np.testing.assert_allclose(values["loss"], values["region"] + values["field"])
    assert trainer.exact_fast_telemetry["exact_h0_train_calls"] == 0
    assert trainer.exact_fast_telemetry["skipped_steps"] == 0
    assert trainer.scheduled_exact_telemetry["schedule_skips"] == 1
    assert trainer.scheduled_exact_telemetry["training_steps"] == 1
    assert values["h0_active"] == 0
    assert values["betti_0_raw"] == 0


def test_scheduled_exact_due_step_precomputes_missing_topology_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def matching(prediction, target, target_pairs):
        calls.append(int(target_pairs.birth_flat.size))
        return _matching()

    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        matching,
    )
    trainer, _ = _step_trainer(MicroBTPFTrainerA3ScheduledExact)
    trainer.current_epoch = 0
    trainer._initialize_exact_fast_state()
    trainer._initialize_scheduled_exact_state()
    batch = _step_batch()
    object.__setattr__(batch["persistent_targets"][0], "topology", None)

    values = trainer.train_step(batch)

    assert np.isfinite(values["loss"])
    assert calls == [1]
    assert batch["persistent_targets"][0].topology is None
    assert trainer.exact_fast_telemetry["exact_h0_train_calls"] == 1
    assert trainer.exact_fast_telemetry["skipped_steps"] == 0
    assert trainer.exact_fast_telemetry["target_precompute_wait_seconds"] > 0
    assert trainer.scheduled_exact_telemetry["training_steps"] == 1


def test_scheduled_exact_dataloader_disables_unused_training_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake_builder(trainer, **kwargs):
        observed.update(kwargs)
        return "train", "validation"

    trainer_module = sys.modules[MicroBTPFTrainerA3ScheduledExact.__module__]
    monkeypatch.setattr(
        trainer_module,
        "build_microbt_pf_dataloaders",
        fake_builder,
    )
    trainer = object.__new__(MicroBTPFTrainerA3ScheduledExact)

    assert trainer.get_dataloaders() == ("train", "validation")
    assert observed == {
        "component_uniform": True,
        "precompute_training_topology": False,
    }


def test_scheduled_exact_tail_step_matches_exact_fast_weight_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        microbt_pf_losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: _matching(),
    )
    reference, _ = _step_trainer(MicroBTPFTrainerA3ExactFast)
    reference._initialize_exact_fast_state()
    scheduled, _ = _step_trainer(MicroBTPFTrainerA3ScheduledExact)
    scheduled.current_epoch = 197
    scheduled._initialize_exact_fast_state()
    scheduled._initialize_scheduled_exact_state()

    exact_values = reference.train_step(_step_batch())
    scheduled_values = scheduled.train_step(_step_batch())

    for key in exact_values:
        np.testing.assert_array_equal(scheduled_values[key], exact_values[key])
    assert scheduled.scheduled_exact_telemetry["schedule_mode"] == "full_exact"
    assert scheduled.scheduled_exact_telemetry["effective_h0_weight"] == 1.0


def test_scheduled_exact_epoch_start_resets_local_schedule_for_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MicroBTPFTrainerA3ExactFast,
        "on_train_epoch_start",
        lambda self: "started",
    )
    trainer = object.__new__(MicroBTPFTrainerA3ScheduledExact)
    trainer.current_epoch = 197
    trainer._initialize_scheduled_exact_state()
    trainer._scheduled_train_step_index = 173
    trainer.scheduled_exact_telemetry["training_steps"] = 173
    trainer.scheduled_exact_telemetry["schedule_skips"] = 156

    result = trainer.on_train_epoch_start()

    assert result == "started"
    assert trainer._scheduled_train_step_index == 0
    assert trainer.scheduled_exact_telemetry == {
        "schedule_mode": "full_exact",
        "interval": 1,
        "effective_h0_weight": 1.0,
        "tail_start_epoch": 197,
        "training_steps": 0,
        "schedule_skips": 0,
    }


def test_scheduled_exact_epoch_end_writes_component_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Break caught: nnU-Net's base epoch logger discards the already-computed
    # region, field, and Betti components.
    monkeypatch.setattr(
        MicroBTPFTrainerA3ExactFast,
        "on_train_epoch_end",
        lambda self, outputs: "base-result",
    )
    trainer = object.__new__(MicroBTPFTrainerA3ScheduledExact)
    trainer.output_folder = str(tmp_path)
    trainer.current_epoch = 4
    trainer.scheduled_exact_telemetry = {
        "schedule_mode": "sparse_exact",
        "effective_h0_weight": 10.0,
    }
    messages: list[str] = []
    trainer.print_to_log_file = messages.append
    outputs = [
        {
            "loss": np.asarray(2.6),
            "region": np.asarray(0.4),
            "field": np.asarray(0.2),
            "betti_0": np.asarray(2.0),
            "persistent_field": np.asarray(2.2),
            "betti_0_raw": np.asarray(0.2),
            "h0_active": np.asarray(1),
        },
        {
            "loss": np.asarray(0.8),
            "region": np.asarray(0.5),
            "field": np.asarray(0.3),
            "betti_0": np.asarray(0.0),
            "persistent_field": np.asarray(0.3),
            "betti_0_raw": np.asarray(0.0),
            "h0_active": np.asarray(0),
        },
    ]

    result = trainer.on_train_epoch_end(outputs)

    assert result == "base-result"
    rows = [
        json.loads(line)
        for line in (tmp_path / "monitor" / "learning_components.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["phase"] == "train"
    assert rows[0]["epoch"] == 4
    assert rows[0]["h0_active_steps"] == 1
    assert rows[0]["betti_0_raw_active_mean"] == pytest.approx(0.2)
    assert any("component telemetry" in message for message in messages)


def test_scheduled_exact_train_start_preflights_explicit_fixed_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MicroBTPFTrainerA3ExactFast,
        "on_train_start",
        lambda self: "started",
    )
    trainer_module = sys.modules[MicroBTPFTrainerA3ScheduledExact.__module__]
    sentinel = object()
    observed: dict = {}

    def fake_from_trainer(trainer, manifest_path, output_root):
        observed.update(
            trainer=trainer,
            manifest_path=Path(manifest_path),
            output_root=Path(output_root),
        )
        return sentinel

    monkeypatch.setattr(
        trainer_module.FixedCaseMonitor,
        "from_trainer",
        fake_from_trainer,
    )
    manifest = tmp_path / "fixed.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MICROBT_PF_MONITOR_MANIFEST", str(manifest))
    monkeypatch.setenv("MICROBT_PF_MONITOR_OUTPUT", str(tmp_path / "sentinel"))
    trainer = object.__new__(MicroBTPFTrainerA3ScheduledExact)
    trainer.print_to_log_file = lambda *_: None

    result = trainer.on_train_start()

    assert result == "started"
    assert trainer._fixed_case_monitor is sentinel
    assert observed == {
        "trainer": trainer,
        "manifest_path": manifest,
        "output_root": tmp_path / "sentinel",
    }


def test_scheduled_exact_epoch_and_final_hooks_use_completed_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    class Monitor:
        def run_if_due(self, epoch, *, final=False):
            calls.append((int(epoch), bool(final)))

    def base_epoch_end(trainer):
        trainer.current_epoch += 1
        return "epoch-ended"

    monkeypatch.setattr(
        MicroBTPFTrainerA3ExactFast,
        "on_epoch_end",
        base_epoch_end,
    )
    monkeypatch.setattr(
        MicroBTPFTrainerA3ExactFast,
        "on_train_end",
        lambda self: "training-ended",
    )
    trainer = object.__new__(MicroBTPFTrainerA3ScheduledExact)
    trainer.current_epoch = 24
    trainer.num_epochs = 200
    trainer._fixed_case_monitor = Monitor()

    assert trainer.on_epoch_end() == "epoch-ended"
    assert calls == [(25, False)]
    trainer.current_epoch = 200
    assert trainer.on_train_end() == "training-ended"
    assert calls[-1] == (200, True)


def test_conditional_trainer_routes_deep_supervision_to_wrapped_unet() -> None:
    unet = nn.Module()
    unet.decoder = SimpleNamespace(deep_supervision=True)
    wrapped = MicroBTPFNetwork(unet)
    trainer = object.__new__(MicroBTPFTrainerA3ScheduledExact)
    trainer.is_ddp = False
    trainer.network = SimpleNamespace(_orig_mod=wrapped)

    trainer.set_deep_supervision_enabled(True)

    assert unet.decoder.deep_supervision is False


def test_validation_uses_region_probabilities_not_conditional_mark_logits() -> None:
    trainer = object.__new__(MicroBTPFTrainerA2)
    trainer.device = torch.device("cpu")
    region_probability = torch.zeros((1, 3, *SHAPE))
    region_probability[:, 0] = 1.0

    class FixedNetwork(nn.Module):
        def forward(self, data):
            return {
                "field_logits": torch.full((1, 3, *SHAPE), 20.0),
                "region_probabilities": region_probability,
            }

    class FixedLoss:
        def __call__(self, output, target, persistent_targets):
            zero = output["field_logits"].sum() * 0.0
            return {
                "loss": zero,
                "region": zero,
                "persistent_field": zero,
                "field": zero,
                "betti_0": zero,
            }

    trainer.network = FixedNetwork()
    trainer.loss = FixedLoss()
    trainer.label_manager = SimpleNamespace(has_ignore_label=False, has_regions=True)
    target = torch.zeros((1, 3, *SHAPE))
    target[:, 0] = 1.0
    observed = trainer.validation_step(
        {"data": torch.zeros((1, 1, *SHAPE)), "target": target, "persistent_targets": ()}
    )
    np.testing.assert_array_equal(observed["tp_hard"], [45, 0, 0])
    np.testing.assert_array_equal(observed["fp_hard"], [0, 0, 0])
    np.testing.assert_array_equal(observed["fn_hard"], [0, 0, 0])


def _tiny_plans(plans: dict) -> dict:
    plans = deepcopy(plans)
    configuration = plans["configurations"]["3d_fullres"]
    configuration["patch_size"] = [8, 8, 8]
    configuration["batch_size"] = 1
    architecture = configuration["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 2
    architecture["features_per_stage"] = [4, 8]
    architecture["kernel_sizes"] = [[3, 3, 3], [3, 3, 3]]
    architecture["strides"] = [[1, 1, 1], [2, 2, 2]]
    architecture["n_blocks_per_stage"] = [1, 1]
    architecture["n_conv_per_stage_decoder"] = [1]
    return plans


def test_a3_nnunet_checkpoint_roundtrip_is_bitwise_equal(tmp_path: Path) -> None:
    plans, dataset_json = _plans_and_dataset()
    plans = _tiny_plans(plans)
    trainer_folder = Path(sys.modules[nnUNetTrainer.__module__].__file__).parent
    discovered = recursive_find_python_class(
        str(trainer_folder), "MicroBTPFTrainerA3", "nnunetv2.training.nnUNetTrainer"
    )
    assert discovered is MicroBTPFTrainerA3
    os.environ["nnUNet_compile"] = "false"
    torch.manual_seed(7)
    source = discovered(plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu"))
    source.initialize()
    source.network.eval()
    fixed_input = torch.linspace(-1.0, 1.0, 4 * 8 * 8 * 8).reshape(1, 4, 8, 8, 8)
    with torch.no_grad():
        expected = source.network(fixed_input)
    checkpoint = tmp_path / "checkpoint.pth"
    source.save_checkpoint(str(checkpoint))

    torch.manual_seed(99)
    restored = discovered(plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu"))
    restored.load_checkpoint(str(checkpoint))
    restored.network.eval()
    with torch.no_grad():
        observed = restored.network(fixed_input)
    assert set(observed) == set(expected)
    assert all(torch.equal(observed[key], expected[key]) for key in expected)
