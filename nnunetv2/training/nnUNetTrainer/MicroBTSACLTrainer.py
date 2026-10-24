from __future__ import annotations

import json

import numpy as np

from nnunetv2.training.loss.microbt_sacl_losses import MicroBTSACLLoss
from nnunetv2.training.nnUNetTrainer.MicroBTCTFTrainer import ctf_object_weight
from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassSurvivalTrainer import (
    MicroBTUnitMassSurvivalTrainerScratch60,
)


def sacl_continuation_lambda(epoch: int) -> float:
    """Reuse B0's warm-up, then keep its reached endpoint through epoch 299."""
    epoch = int(epoch)
    if not 0 <= epoch < 300:
        raise ValueError("epoch must be in [0, 300)")
    # B0's scientific schedule reaches its frozen endpoint at epoch 29.  Its
    # helper retains a legacy 200-epoch domain guard, so final300 must clamp
    # only the helper's domain while preserving the exact endpoint value.
    return ctf_object_weight(min(epoch, 199)) / 0.25


def sacl_weights(epoch: int) -> tuple[float, float]:
    """Continue support -> measure -> transported atoms with one scalar path."""
    continuation = sacl_continuation_lambda(epoch)
    return 0.25 * continuation, 0.25 * continuation**2


class MicroBTUnitMassSACLTrainerScratch60(
    MicroBTUnitMassSurvivalTrainerScratch60
):
    """B0 with Support-to-Atom Continuation Learning as the sole intervention."""

    sacl_schedule = "b0_linear_measure_quadratic_transport"

    def _build_loss(self):
        return MicroBTSACLLoss(
            measure_weight=0.0,
            transport_weight=0.0,
            hard_survival_mass=self.hard_survival_mass,
        )

    def _initialize_exact_fast_state(self) -> None:
        super()._initialize_exact_fast_state()
        self._validation_monitor_loss = MicroBTSACLLoss(
            measure_weight=0.0,
            transport_weight=0.0,
            hard_survival_mass=self.hard_survival_mass,
        )

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        measure_weight, transport_weight = sacl_weights(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, MicroBTSACLLoss):
                raise TypeError("SACL trainer requires MicroBTSACLLoss")
            criterion.set_sacl_weights(measure_weight, transport_weight)
        self._current_ctf_weight = measure_weight
        self._current_sacl_lambda = sacl_continuation_lambda(self.current_epoch)
        self._current_sacl_measure_weight = measure_weight
        self._current_sacl_transport_weight = transport_weight
        return measure_weight

    def on_train_start(self):
        result = super().on_train_start()
        self.print_to_log_file(
            "MicroBT-SACL C2-Full frozen protocol: "
            + json.dumps(
                {
                    "architecture_carrier": "B0",
                    "causal_change": "support_to_atom_continuation_learning",
                    "decoder_min_atom_mass": 0.25,
                    "hard_survival_mass": self.hard_survival_mass,
                    "measure_weight": "0.25*lambda",
                    "object_endpoint_epoch": 29,
                    "schema_version": 1,
                    "semantic_support_weight": 1.0,
                    "support_only_epochs": [0, 4],
                    "transport_weight": "0.25*lambda^2",
                },
                sort_keys=True,
            )
        )
        return result


class MicroBTUnitMassSACLFinal300Trainer(MicroBTUnitMassSACLTrainerScratch60):
    """Final C2 protocol with a 300-epoch PolyLR horizon and formal snapshots."""

    training_epochs = 300
    continuation_snapshot_epochs = (100, 150, 200, 250, 300)

    def on_train_epoch_start(self):
        result = super().on_train_epoch_start()
        self.print_to_log_file(
            "MicroBT-SACL continuation: "
            + json.dumps(
                {
                    "epoch": int(self.current_epoch),
                    "lambda": self._current_sacl_lambda,
                    "measure_weight": self._current_sacl_measure_weight,
                    "schema_version": 1,
                    "transport_weight": self._current_sacl_transport_weight,
                },
                sort_keys=True,
            )
        )
        return result

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        summary = {"epoch": int(self.current_epoch), "schema_version": 1}
        for key in (
            "sacl_measure_weight",
            "sacl_transport_weight",
            "mass_conservation_weighted",
            "transport_assignment_weighted",
            "transport_assignment_raw",
        ):
            values = np.asarray(
                [np.asarray(item[key]).item() for item in train_outputs]
            )
            summary[f"{key}_mean"] = float(values.mean())
        self.print_to_log_file(
            "MicroBT-SACL component telemetry: "
            + json.dumps(summary, sort_keys=True)
        )
        return result
