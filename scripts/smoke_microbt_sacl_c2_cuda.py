from __future__ import annotations

import json

import numpy as np
import torch

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.MicroBTSACLTrainer import (
    MicroBTUnitMassSACLTrainerScratch60,
    sacl_weights,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    trainer = get_trainer_from_args(
        "1",
        "3d_fullres",
        0,
        "MicroBTUnitMassSACLTrainerScratch60",
        "nnUNetResEncUNetMPlans",
        device=torch.device("cuda"),
    )
    if not isinstance(trainer, MicroBTUnitMassSACLTrainerScratch60):
        raise RuntimeError(type(trainer).__name__)
    trainer.initialize()
    if trainer.optimizer.state:
        raise RuntimeError("scratch optimizer state is not empty")
    train_loader, validation_loader = trainer.get_dataloaders()
    results = []
    required = (
        "loss",
        "region",
        "object_transport",
        "transport_assignment_raw",
        "transport_assignment_weighted",
        "mass_conservation_raw",
        "mass_conservation_weighted",
        "sacl_measure_weight",
        "sacl_transport_weight",
    )
    for epoch in (5, 17):
        trainer.current_epoch = epoch
        trainer._set_object_morse_weight_for_current_epoch()
        result = trainer.train_step(next(train_loader))
        missing = set(required).difference(result)
        if missing:
            raise RuntimeError(f"missing SACL smoke metrics: {sorted(missing)}")
        if not all(np.isfinite(np.asarray(result[key])).all() for key in required):
            raise FloatingPointError("nonfinite SACL smoke result")
        measure, transport = sacl_weights(epoch)
        if abs(float(np.asarray(result["sacl_measure_weight"])) - measure) > 1e-7:
            raise RuntimeError("measure schedule drift")
        if abs(float(np.asarray(result["sacl_transport_weight"])) - transport) > 1e-7:
            raise RuntimeError("transport schedule drift")
        results.append(
            {
                "epoch": epoch,
                "loss": float(np.asarray(result["loss"])),
                "measure_weight": measure,
                "transport_weight": transport,
            }
        )
    for loader in (train_loader, validation_loader):
        finish = getattr(loader, "_finish", None)
        if finish is not None:
            finish()
    if not trainer.optimizer.state:
        raise RuntimeError("optimizer did not update")
    print(
        json.dumps(
            {
                "device": next(trainer.network.parameters()).device.type,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "num_epochs": trainer.num_epochs,
                "optimizer_state_entries_after_step": len(trainer.optimizer.state),
                "results": results,
                "status": "passed",
                "trainer": trainer.__class__.__name__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
