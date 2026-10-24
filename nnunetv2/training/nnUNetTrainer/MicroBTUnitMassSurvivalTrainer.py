from __future__ import annotations

import json

from nnunetv2.training.loss.microbt_umctf_losses import MicroBTUnitMassLoss
from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassTrainer import (
    MicroBTUnitMassTrainerScratch60,
)


class MicroBTUnitMassSurvivalTrainerScratch60(MicroBTUnitMassTrainerScratch60):
    """UMCTF with one decoder-aligned hard-object survival term."""

    hard_survival_mass = 0.25

    def _build_loss(self):
        return MicroBTUnitMassLoss(
            object_weight=0.0,
            hard_survival_mass=self.hard_survival_mass,
        )

    def _initialize_exact_fast_state(self) -> None:
        super()._initialize_exact_fast_state()
        self._validation_monitor_loss = MicroBTUnitMassLoss(
            object_weight=0.0,
            hard_survival_mass=self.hard_survival_mass,
        )

    def on_train_start(self):
        result = super().on_train_start()
        self.print_to_log_file(
            "MicroBT-UMCTF-HS frozen protocol: "
            + json.dumps(
                {
                    "causal_change": "decoder_aligned_hard_object_survival",
                    "hard_survival_mass": self.hard_survival_mass,
                    "loss_aggregation": "mean_unit_mass_plus_mean_hard_deficit",
                    "object_density": "sigmoid_wt_times_sigmoid_density_logit",
                    "schema_version": 1,
                },
                sort_keys=True,
            )
        )
        return result
