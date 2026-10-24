from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nnunetv2.nets.microbt_spp.model import MicroBTSPPNetwork
from nnunetv2.training.nnUNetTrainer.MicroBTSPPTrainer import (
    MicroBTSPPTrainerGate5,
    MicroBTSPPTrainerScheduledExact,
    _MicroBTSPPVariantCriterion,
)
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTSBPFTrainerScheduledExact,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def test_spp_builder_uses_four_raw_channels_without_deep_supervision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}
    unet = nn.Conv3d(1, 4, 1)

    def fake_builder(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return unet

    monkeypatch.setattr(nnUNetTrainer, "build_network_architecture", fake_builder)
    network = MicroBTSPPTrainerScheduledExact.build_network_architecture(
        "architecture", {}, (), 1, 3, enable_deep_supervision=True
    )

    assert isinstance(network, MicroBTSPPNetwork)
    assert observed["args"][4] == 4
    assert observed["kwargs"]["enable_deep_supervision"] is False


def test_spp_training_horizon_ramp_and_sparse_exact_schedule_respect_six_minute_gate() -> None:
    assert MicroBTSPPTrainerScheduledExact.training_epochs == 200
    assert MicroBTSPPTrainerScheduledExact.initial_learning_rate == pytest.approx(3e-4)
    assert MicroBTSPPTrainerScheduledExact.object_morse_weight(0) == pytest.approx(0.25)
    assert MicroBTSPPTrainerScheduledExact.object_morse_weight(9) == pytest.approx(1.0)
    assert MicroBTSPPTrainerScheduledExact.object_morse_weight(199) == pytest.approx(1.0)
    assert MicroBTSPPTrainerScheduledExact.validation_h0_audit_iterations == 0
    assert MicroBTSPPTrainerScheduledExact.exact_h0_schedule(199, 0) == (
        "sparse_exact",
        True,
        10.0,
    )
    assert MicroBTSPPTrainerScheduledExact.exact_h0_schedule(199, 1) == (
        "sparse_exact",
        False,
        0.0,
    )


def test_spp_limits_dense_validation_without_changing_training_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_parent_init(self, *args, **kwargs) -> None:
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

    monkeypatch.setattr(MicroBTSBPFTrainerScheduledExact, "__init__", fake_parent_init)
    trainer = MicroBTSPPTrainerScheduledExact({}, "3d_fullres", 0, {})

    assert trainer.num_iterations_per_epoch == 250
    assert trainer.num_val_iterations_per_epoch == 2


def test_spp_validation_reuses_training_graph_without_backward_and_restores_mode() -> None:
    class PerfectSemanticNetwork(nn.Module):
        def forward(self, data):
            assert self.training is True
            assert torch.is_grad_enabled() is True
            probabilities = torch.ones(
                (data.shape[0], 3, *data.shape[2:]), dtype=data.dtype
            )
            return {"region_probabilities": probabilities}

    compiled_network = PerfectSemanticNetwork()
    compiled_network.eval()
    trainer = object.__new__(MicroBTSPPTrainerScheduledExact)
    trainer.is_ddp = False
    trainer.network = compiled_network
    trainer.device = torch.device("cpu")
    trainer.label_manager = SimpleNamespace(has_ignore_label=False)
    trainer._validation_monitor_loss = SimpleNamespace(
        region_fidelity=lambda probabilities, target: probabilities.sum() * 0.0
    )
    with torch.no_grad():
        result = trainer.validation_step(
            {
                "data": torch.zeros((1, 1, 2, 2, 2)),
                "target": torch.ones((1, 3, 2, 2, 2), dtype=torch.bool),
            }
        )

    assert result["loss"] == 0
    assert result["object_morse"] == 0
    assert result["tp_hard"].tolist() == [8, 8, 8]
    assert result["fp_hard"].tolist() == [0, 0, 0]
    assert result["fn_hard"].tolist() == [0, 0, 0]
    assert trainer.network is compiled_network
    assert trainer.network.training is False


def test_spp_requests_semantic_only_validation_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "nnunetv2.training.nnUNetTrainer.MicroBTSPPTrainer"
    )
    observed = {}

    def fake_builder(trainer, **kwargs):
        observed.update(kwargs)
        return "train-loader", "validation-loader"

    monkeypatch.setattr(
        module, "build_microbt_pf_dataloaders", fake_builder, raising=False
    )
    trainer = object.__new__(MicroBTSPPTrainerScheduledExact)

    loaders = trainer.get_dataloaders()

    assert loaders == ("train-loader", "validation-loader")
    assert observed == {
        "component_uniform": True,
        "precompute_training_topology": False,
        "semantic_validation": True,
    }


def test_spp_freezes_two_validation_batches_before_epoch_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    class ValidationLoader:
        def __init__(self):
            self.items = iter(({"batch": 1}, {"batch": 2}))
            self.finished = False

        def __next__(self):
            return next(self.items)

        def _finish(self):
            self.finished = True

    def fake_parent_start(self):
        self.dataloader_val = ValidationLoader()
        return sentinel

    monkeypatch.setattr(
        MicroBTSBPFTrainerScheduledExact, "on_train_start", fake_parent_start
    )
    trainer = object.__new__(MicroBTSPPTrainerScheduledExact)

    result = trainer.on_train_start()
    source_loader = trainer._validation_source_loader
    observed = [next(trainer.dataloader_val)["batch"] for _ in range(5)]

    assert result is sentinel
    assert observed == [1, 2, 1, 2, 1]
    assert len(trainer._fixed_validation_batches) == 2
    assert source_loader.finished is False
    assert callable(trainer.dataloader_val._finish)


def test_spp_criterion_scales_false_persistence_once_inside_betti_family() -> None:
    values = {
        "loss": torch.tensor(10.0),
        "region": torch.tensor(1.0),
        "object_morse": torch.tensor(9.0),
        "persistent_field": torch.tensor(9.0),
        "field": torch.tensor(2.0),
        "betti_0": torch.tensor(3.0),
        "false_persistence": torch.tensor(1.0),
        "birth_completeness": torch.tensor(4.0),
    }

    ramped = _MicroBTSPPVariantCriterion._apply_object_morse_weight(values, 0.5)
    sparse = MicroBTSPPTrainerScheduledExact._apply_betti_weight(ramped, 10.0)

    assert ramped["false_persistence"] == 0.5
    assert sparse["false_persistence"] == 5
    assert sparse["betti_0"] == 15
    assert sparse["object_morse"] == 36
    assert sparse["loss"] == 37


def test_spp_gate_is_five_epochs_with_formal_lr_horizon() -> None:
    trainer = object.__new__(MicroBTSPPTrainerGate5)
    trainer.network = nn.Linear(2, 2)
    trainer.initial_lr = 3e-4
    trainer.weight_decay = 3e-5
    _, scheduler = trainer.configure_optimizers()

    assert trainer.gate_epochs == 5
    assert trainer.training_epochs == 200
    assert scheduler.max_steps == 200
