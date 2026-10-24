from __future__ import annotations

import json

import numpy as np
import torch

from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.MicroBTCTFTrainer import (
    MicroBTCTFTrainerScratch60,
    ctf_object_weight,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    trainer = get_trainer_from_args(
        "1", "3d_fullres", 0, "MicroBTCTFTrainerScratch60",
        "nnUNetResEncUNetMPlans", device=torch.device("cuda")
    )
    if not isinstance(trainer, MicroBTCTFTrainerScratch60):
        raise RuntimeError(type(trainer).__name__)
    trainer.initialize()
    if trainer.optimizer.state:
        raise RuntimeError("scratch optimizer state is not empty")
    trainer.current_epoch = 5
    trainer._set_object_morse_weight_for_current_epoch()
    train_loader, validation_loader = trainer.get_dataloaders()
    batch = next(train_loader)
    result = trainer.train_step(batch)
    for loader in (train_loader, validation_loader):
        finish = getattr(loader, "_finish", None)
        if finish is not None:
            finish()
    required = ("loss", "region", "object_transport", "transport_offset", "center_anchor")
    if any(key not in result for key in required):
        raise RuntimeError(f"missing smoke metrics: {sorted(set(required).difference(result))}")
    if not all(np.isfinite(np.asarray(result[key])).all() for key in required):
        raise FloatingPointError("nonfinite CTF smoke result")
    print(json.dumps({
        "status": "passed",
        "trainer": trainer.__class__.__name__,
        "device": next(trainer.network.parameters()).device.type,
        "num_epochs": trainer.num_epochs,
        "optimizer_state_entries_after_step": len(trainer.optimizer.state),
        "epoch5_object_weight": ctf_object_weight(5),
        "exact_h0": False,
        "loss": float(np.asarray(result["loss"])),
        "region": float(np.asarray(result["region"])),
        "object_transport": float(np.asarray(result["object_transport"])),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
