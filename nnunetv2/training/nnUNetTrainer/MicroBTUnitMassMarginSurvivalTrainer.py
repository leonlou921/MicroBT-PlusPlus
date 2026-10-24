from __future__ import annotations

import json

from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassSurvivalTrainer import (
    MicroBTUnitMassSurvivalTrainerScratch60,
)


class MicroBTUnitMassMarginSurvivalTrainerScratch60(
    MicroBTUnitMassSurvivalTrainerScratch60
):
    """UMCTF-HS with one training-only survival margin above decoder cutoff."""

    hard_survival_mass = 0.50
    decoder_min_atom_mass = 0.25

    def on_train_start(self):
        result = super().on_train_start()
        self.print_to_log_file(
            "MicroBT-UMCTF-HSM05 frozen causal change: "
            + json.dumps(
                {
                    "causal_change": "decoder_aligned_survival_margin",
                    "decoder_min_atom_mass": self.decoder_min_atom_mass,
                    "scientific_parent": "UMCTF-HS",
                    "training_survival_mass": self.hard_survival_mass,
                    "schema_version": 1,
                },
                sort_keys=True,
            )
        )
        return result

