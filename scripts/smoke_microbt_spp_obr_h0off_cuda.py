from __future__ import annotations

import json

import torch

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRH0OffCausalTrainer import (
    MicroBTSPPOBRH0OffCausalTrainerScratch60,
    h0off_curriculum_weights,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    trainer = get_trainer_from_args(
        "1",
        "3d_fullres",
        0,
        "MicroBTSPPOBRH0OffCausalTrainerScratch60",
        "nnUNetResEncUNetMPlans",
        device=torch.device("cuda"),
    )
    if not isinstance(trainer, MicroBTSPPOBRH0OffCausalTrainerScratch60):
        raise RuntimeError(type(trainer).__name__)
    trainer.initialize()
    if trainer.num_epochs != 60:
        raise RuntimeError(f"unexpected num_epochs={trainer.num_epochs}")
    if next(trainer.network.parameters()).device.type != "cuda":
        raise RuntimeError("network was not initialized on CUDA")
    for epoch in (0, 15, 30, 59):
        for step in (0, 249):
            mode, active, weight = trainer.exact_h0_schedule(epoch, step)
            if (mode, active, weight) != ("h0_off_causal", False, 0.0):
                raise RuntimeError(f"H0 contract violated at {epoch}:{step}")
    trainer.current_epoch = 30
    trainer._set_object_morse_weight_for_current_epoch()
    weights = trainer._current_curriculum_weights
    if weights != h0off_curriculum_weights(30) or weights.h0 != 0.0:
        raise RuntimeError(f"invalid curriculum weights: {weights}")
    print(
        json.dumps(
            {
                "status": "passed",
                "trainer": trainer.__class__.__name__,
                "num_epochs": trainer.num_epochs,
                "device": next(trainer.network.parameters()).device.type,
                "parameter_count": sum(p.numel() for p in trainer.network.parameters()),
                "epoch30_weights": weights._asdict(),
                "exact_h0": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
