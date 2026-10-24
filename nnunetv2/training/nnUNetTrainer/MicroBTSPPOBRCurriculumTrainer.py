from __future__ import annotations

import os
from typing import NamedTuple

import torch
from torch import nn

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ExactFast,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPTrainer import (
    MicroBTSPPOBRTrainerScratch200,
    _MicroBTSPPOBRVariantCriterion,
)


class CurriculumWeights(NamedTuple):
    field: float
    birth: float
    h0: float


def _linear_phase(
    epoch: int,
    start_epoch: int,
    end_epoch: int,
    start_value: float,
    end_value: float,
) -> float:
    progress = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(start_value) + progress * (float(end_value) - float(start_value))


def curriculum_weights(epoch: int) -> CurriculumWeights:
    """Return the pre-registered field, OBR, and exact-H0 weights."""
    epoch = int(epoch)
    if not 0 <= epoch < 200:
        raise ValueError("epoch must be in [0, 200)")
    if epoch <= 4:
        return CurriculumWeights(0.0, 0.0, 0.0)
    if epoch <= 14:
        return CurriculumWeights(
            _linear_phase(epoch, 5, 14, 0.025, 0.25),
            _linear_phase(epoch, 5, 14, 0.005, 0.05),
            0.0,
        )
    if epoch <= 29:
        return CurriculumWeights(
            _linear_phase(epoch, 14, 29, 0.25, 0.50),
            _linear_phase(epoch, 14, 29, 0.05, 0.20),
            0.0,
        )
    if epoch <= 59:
        return CurriculumWeights(
            _linear_phase(epoch, 29, 59, 0.50, 1.0),
            _linear_phase(epoch, 29, 59, 0.20, 1.0),
            _linear_phase(epoch, 29, 59, 0.0, 1.0),
        )
    return CurriculumWeights(1.0, 1.0, 1.0)


class _MicroBTSPPOBRCurriculumCriterion(_MicroBTSPPOBRVariantCriterion):
    """Apply independent curriculum weights without changing loss formulas."""

    def __init__(self, h0_matching: bool) -> None:
        super().__init__(h0_matching)
        self.curriculum = CurriculumWeights(0.0, 0.0, 0.0)

    def set_curriculum_weights(self, weights: CurriculumWeights) -> None:
        if any(not 0.0 <= float(value) <= 1.0 for value in weights):
            raise ValueError("curriculum weights must be in [0, 1]")
        self.curriculum = CurriculumWeights(*(float(value) for value in weights))

    @staticmethod
    def _apply_curriculum_weights(
        values: dict[str, torch.Tensor], weights: CurriculumWeights
    ) -> dict[str, torch.Tensor]:
        weighted = dict(values)
        raw_field = values.get("field_raw", values["field"])
        raw_birth = values.get("birth_reservation_raw", values["birth_reservation"])
        raw_existence = values.get("birth_existence_raw", values["birth_existence"])
        raw_prominence = values.get("birth_prominence_raw", values["birth_prominence"])

        weighted["field_raw"] = raw_field
        weighted["field"] = raw_field * float(weights.field)
        weighted["birth_reservation_raw"] = raw_birth
        weighted["birth_existence_raw"] = raw_existence
        weighted["birth_prominence_raw"] = raw_prominence
        weighted["birth_reservation"] = raw_birth * float(weights.birth)
        weighted["birth_existence"] = raw_existence * float(weights.birth)
        weighted["birth_prominence"] = raw_prominence * float(weights.birth)
        weighted["object_morse"] = (
            weighted["field"]
            + values["betti_0"]
            + weighted["birth_reservation"]
        )
        weighted["persistent_field"] = weighted["object_morse"]
        weighted["loss"] = values["region"].float() + weighted["object_morse"]
        for name, value in (
            ("field_weight", weights.field),
            ("birth_weight", weights.birth),
            ("h0_curriculum_weight", weights.h0),
        ):
            weighted[name] = torch.as_tensor(
                value, device=values["loss"].device, dtype=values["loss"].dtype
            )
        return weighted

    def forward(self, output, region_target, persistent_targets):
        values = super().forward(output, region_target, persistent_targets)
        return self._apply_curriculum_weights(values, self.curriculum)


class MicroBTSPPOBRCurriculumTrainerScratch200(MicroBTSPPOBRTrainerScratch200):
    """Complete SPP-OBR from scratch with a causal three-level curriculum."""

    gate_epochs = 30

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        requested_epochs = int(
            os.getenv("MICROBT_CURRICULUM_GATE_EPOCHS", str(self.training_epochs))
        )
        if requested_epochs not in (self.gate_epochs, self.training_epochs):
            raise ValueError("curriculum run length must be 30 or 200 epochs")
        self.num_epochs = requested_epochs
        self.save_every = 5 if requested_epochs == self.gate_epochs else 25

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(
            optimizer, self.initial_lr, self.training_epochs
        )
        return optimizer, scheduler

    def _build_loss(self) -> nn.Module:
        return _MicroBTSPPOBRCurriculumCriterion(h0_matching=True)

    def _initialize_exact_fast_state(self) -> None:
        MicroBTPFTrainerA3ExactFast._initialize_exact_fast_state(self)
        self._validation_monitor_loss = _MicroBTSPPOBRCurriculumCriterion(
            h0_matching=False
        )

    @classmethod
    def exact_h0_schedule(
        cls, epoch: int, step: int
    ) -> tuple[str, bool, float]:
        epoch = int(epoch)
        step = int(step)
        weights = curriculum_weights(epoch)
        if not 0 <= step < 250:
            raise ValueError("training step must be in [0, 250)")
        if weights.h0 == 0.0:
            return "curriculum_dense", False, 0.0
        use_exact_h0 = step % cls.sparse_exact_interval == 0
        return (
            "sparse_exact",
            use_exact_h0,
            cls.sparse_h0_weight * weights.h0 if use_exact_h0 else 0.0,
        )

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weights = curriculum_weights(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, _MicroBTSPPOBRCurriculumCriterion):
                raise TypeError("curriculum SPP-OBR requires its criterion")
            criterion.set_curriculum_weights(weights)
        self._current_curriculum_weights = weights
        return max(weights)

    def on_train_epoch_start(self):
        weights = curriculum_weights(int(self.current_epoch))
        self.print_to_log_file(
            "SPP-OBR curriculum weights: "
            f"field={weights.field:.6f}, birth={weights.birth:.6f}, "
            f"h0={weights.h0:.6f}"
        )
        return super().on_train_epoch_start()

