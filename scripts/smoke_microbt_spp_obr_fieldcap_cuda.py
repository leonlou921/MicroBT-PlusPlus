from __future__ import annotations

import json

import torch

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRFieldCapCausalTrainer import (
    MicroBTSPPOBRFieldCapCausalTrainerScratch60,
    field_cap_curriculum_weights,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    trainer = get_trainer_from_args(
        "1",
        "3d_fullres",
        0,
        "MicroBTSPPOBRFieldCapCausalTrainerScratch60",
        "nnUNetResEncUNetMPlans",
        device=torch.device("cuda"),
    )
    if not isinstance(trainer, MicroBTSPPOBRFieldCapCausalTrainerScratch60):
        raise RuntimeError(type(trainer).__name__)
    trainer.initialize()
    if trainer.num_epochs != 60:
        raise RuntimeError(f"unexpected num_epochs={trainer.num_epochs}")
    if trainer.optimizer.state:
        raise RuntimeError("scratch optimizer state is not empty")
    if next(trainer.network.parameters()).device.type != "cuda":
        raise RuntimeError("network was not initialized on CUDA")
    for epoch in (0, 29, 30, 45, 59):
        weights = trainer.effective_curriculum_weights(epoch)
        if weights != field_cap_curriculum_weights(epoch):
            raise RuntimeError(f"weight dispatch failed at epoch {epoch}")
        if trainer.exact_h0_schedule(epoch, 249) != (
            "h0_off_causal",
            False,
            0.0,
        ):
            raise RuntimeError(f"H0 contract violated at epoch {epoch}")
    epoch30 = trainer.effective_curriculum_weights(30)
    print(
        json.dumps(
            {
                "status": "passed",
                "trainer": trainer.__class__.__name__,
                "num_epochs": trainer.num_epochs,
                "device": next(trainer.network.parameters()).device.type,
                "optimizer_state_entries": len(trainer.optimizer.state),
                "epoch30_weights": epoch30._asdict(),
                "exact_h0": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
