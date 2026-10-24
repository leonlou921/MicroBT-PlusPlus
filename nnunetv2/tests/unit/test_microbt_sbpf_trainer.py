from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nnunetv2.nets.microbt_pf.model import MicroBTSBPFNetwork
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ExactFast,
    MicroBTPFTrainerA3ScheduledExact,
    MicroBTSBPFTrainerScheduledExact,
    MicroBTSBPFTrainerStableGate5,
    MicroBTSBPFTrainerStableScheduledExact,
    MicroBTSBPFTrainerSmoke,
    _MicroBTSBPFVariantCriterion,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def test_sbpf_builder_maps_three_region_contract_to_four_channel_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    unet = nn.Conv3d(1, 4, 1)

    def fake_builder(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return unet

    monkeypatch.setattr(nnUNetTrainer, "build_network_architecture", fake_builder)
    network = MicroBTSBPFTrainerScheduledExact.build_network_architecture(
        "architecture", {}, (), 1, 3, enable_deep_supervision=True
    )

    assert isinstance(network, MicroBTSBPFNetwork)
    assert network.unet is unet
    assert observed["args"][4] == 4
    assert observed["kwargs"]["enable_deep_supervision"] is False

    with pytest.raises(RuntimeError, match="three-region"):
        MicroBTSBPFTrainerScheduledExact.build_network_architecture(
            "architecture", {}, (), 1, 4
        )


def test_sbpf_criterion_and_schedule_preserve_a3_execution_policy() -> None:
    trainer = object.__new__(MicroBTSBPFTrainerScheduledExact)
    criterion = trainer._build_loss()
    assert isinstance(criterion, _MicroBTSBPFVariantCriterion)
    assert criterion.h0_matching_enabled is True
    assert MicroBTSBPFTrainerScheduledExact.training_epochs == 200
    for epoch in (0, 196, 197, 199):
        for step in (0, 1, 10, 249):
            assert MicroBTSBPFTrainerScheduledExact.exact_h0_schedule(
                epoch, step
            ) == MicroBTPFTrainerA3ScheduledExact.exact_h0_schedule(
                epoch, step
            )


def test_sbpf_sparse_weight_scales_both_exact_topology_terms() -> None:
    values = {
        "loss": torch.tensor(0.0),
        "region": torch.tensor(1.0),
        "object_morse": torch.tensor(9.0),
        "field": torch.tensor(2.0),
        "betti_0": torch.tensor(3.0),
        "birth_completeness": torch.tensor(4.0),
    }

    weighted = MicroBTSBPFTrainerScheduledExact._apply_betti_weight(values, 10.0)

    assert weighted["betti_0"] == 30
    assert weighted["birth_completeness"] == 40
    assert weighted["object_morse"] == 72
    assert weighted["loss"] == 73


def test_sbpf_deep_supervision_routes_to_its_wrapped_unet() -> None:
    unet = nn.Module()
    unet.decoder = SimpleNamespace(deep_supervision=True)
    trainer = object.__new__(MicroBTSBPFTrainerScheduledExact)
    trainer.is_ddp = False
    trainer.network = MicroBTSBPFNetwork(unet)

    trainer.set_deep_supervision_enabled(True)

    assert unet.decoder.deep_supervision is False


def test_sbpf_train_end_skips_terminal_fixed_case_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        MicroBTPFTrainerA3ExactFast,
        "on_train_end",
        lambda self: calls.append("base") or "done",
    )
    trainer = object.__new__(MicroBTSBPFTrainerScheduledExact)
    trainer._fixed_case_monitor = SimpleNamespace(
        run_if_due=lambda *args, **kwargs: calls.append("monitor")
    )

    assert trainer.on_train_end() == "done"
    assert calls == ["base"]


def test_sbpf_smoke_trainer_is_a_bounded_execution_gate() -> None:
    assert issubclass(MicroBTSBPFTrainerSmoke, MicroBTSBPFTrainerScheduledExact)
    assert MicroBTSBPFTrainerSmoke.smoke_epochs == 2
    assert MicroBTSBPFTrainerSmoke.smoke_train_iterations == 20
    assert MicroBTSBPFTrainerSmoke.smoke_val_iterations == 2


def test_sbpf_skips_incompatible_duplicate_standard_posttraining_validation() -> None:
    trainer = object.__new__(MicroBTSBPFTrainerScheduledExact)
    messages: list[str] = []
    trainer.print_to_log_file = messages.append

    assert trainer.perform_actual_validation(save_probabilities=True) is None
    assert messages == [
        "SB-PF standard post-training validation skipped; external paired full203 owns evaluation"
    ]


def test_stable_sbpf_uses_low_lr_and_linear_object_morse_ramp() -> None:
    assert MicroBTSBPFTrainerStableScheduledExact.initial_learning_rate == pytest.approx(
        1e-3
    )
    assert MicroBTSBPFTrainerStableScheduledExact.object_morse_weight(0) == pytest.approx(
        0.05
    )
    assert MicroBTSBPFTrainerStableScheduledExact.object_morse_weight(9) == pytest.approx(
        0.5
    )
    assert MicroBTSBPFTrainerStableScheduledExact.object_morse_weight(19) == pytest.approx(
        1.0
    )
    assert MicroBTSBPFTrainerStableScheduledExact.object_morse_weight(199) == pytest.approx(
        1.0
    )


def test_object_morse_ramp_scales_the_complete_family() -> None:
    values = {
        "loss": torch.tensor(10.0),
        "region": torch.tensor(1.0),
        "object_morse": torch.tensor(9.0),
        "persistent_field": torch.tensor(9.0),
        "field": torch.tensor(2.0),
        "betti_0": torch.tensor(3.0),
        "birth_completeness": torch.tensor(4.0),
    }

    weighted = _MicroBTSBPFVariantCriterion._apply_object_morse_weight(values, 0.5)

    assert weighted["field"] == 1
    assert weighted["betti_0"] == 1.5
    assert weighted["birth_completeness"] == 2
    assert weighted["object_morse"] == 4.5
    assert weighted["persistent_field"] == 4.5
    assert weighted["loss"] == 5.5
    assert weighted["object_morse_weight"] == 0.5


def test_stable_sbpf_preserves_formal_exact_h0_schedule() -> None:
    assert MicroBTSBPFTrainerStableScheduledExact.training_epochs == 200
    for epoch in (0, 19, 196, 197, 199):
        for step in (0, 1, 10, 249):
            assert MicroBTSBPFTrainerStableScheduledExact.exact_h0_schedule(
                epoch, step
            ) == MicroBTSBPFTrainerScheduledExact.exact_h0_schedule(epoch, step)


def test_stable_gate_runs_five_epochs_with_formal_lr_horizon() -> None:
    trainer = object.__new__(MicroBTSBPFTrainerStableGate5)
    trainer.network = nn.Linear(2, 2)
    trainer.initial_lr = 1e-3
    trainer.weight_decay = 3e-5

    _, scheduler = trainer.configure_optimizers()

    assert MicroBTSBPFTrainerStableGate5.gate_epochs == 5
    assert MicroBTSBPFTrainerStableGate5.training_epochs == 200
    assert scheduler.max_steps == 200


def test_stable_trainer_sets_both_criteria_to_current_ramp_weight() -> None:
    trainer = object.__new__(MicroBTSBPFTrainerStableScheduledExact)
    trainer.current_epoch = 9
    trainer.loss = _MicroBTSBPFVariantCriterion(h0_matching=True)
    trainer._validation_monitor_loss = _MicroBTSBPFVariantCriterion(
        h0_matching=False
    )

    weight = trainer._set_object_morse_weight_for_current_epoch()

    assert weight == pytest.approx(0.5)
    assert trainer.loss.object_morse_weight == pytest.approx(0.5)
    assert trainer._validation_monitor_loss.object_morse_weight == pytest.approx(0.5)
