from __future__ import annotations

import json

import numpy as np
import torch

from nnunetv2.inference.microbt_umctf_decode import decode_umctf_instances
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.training.nnUNetTrainer.MicroBTUnitMassSurvivalTrainer import (
    MicroBTUnitMassSurvivalTrainerScratch60,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    trainer = get_trainer_from_args(
        "1", "3d_fullres", 0, "MicroBTUnitMassSurvivalTrainerScratch60",
        "nnUNetResEncUNetMPlans", device=torch.device("cuda"),
    )
    if not isinstance(trainer, MicroBTUnitMassSurvivalTrainerScratch60):
        raise RuntimeError(type(trainer).__name__)
    if trainer.hard_survival_mass != 0.25:
        raise RuntimeError("hard survival threshold drift")
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
    required = (
        "loss", "region", "object_transport", "object_transport_raw",
        "transport_offset", "endpoint_error_vox", "mass_conservation_raw",
        "object_mass_error", "background_mass_error",
        "mean_predicted_object_mass",
    )
    missing = set(required).difference(result)
    if missing:
        raise RuntimeError(f"missing smoke metrics: {sorted(missing)}")
    if not all(np.isfinite(np.asarray(result[key])).all() for key in required):
        raise FloatingPointError("nonfinite Unit-Mass CTF smoke result")
    if not trainer.optimizer.state:
        raise RuntimeError("optimizer did not update")

    density = np.zeros((5, 5, 5), dtype=np.float32)
    density[1, 1, 1] = 0.5
    density[1, 1, 2] = 0.5
    transport = np.zeros((3, 5, 5, 5), dtype=np.float32)
    transport[:, 1, 1, 1] = np.asarray((0.5, 0.5, 0.5))
    transport[:, 1, 1, 2] = np.asarray((0.5, 0.5, -0.5))
    decoded = decode_umctf_instances(
        (density > 0).astype(np.float32), density, transport,
        offset_scale_zyx=(1, 1, 1), density_threshold=0,
        min_atom_mass=0.2, nms_radius=1, max_centers=8,
    )
    if decoded.count != 1 or abs(decoded.total_atom_mass - 1.0) > 1e-6:
        raise RuntimeError("CUDA smoke decoder contract failed")
    print(json.dumps({
        "status": "passed",
        "trainer": trainer.__class__.__name__,
        "num_epochs": trainer.num_epochs,
        "device": next(trainer.network.parameters()).device.type,
        "optimizer_state_entries_after_step": len(trainer.optimizer.state),
        "loss": float(np.asarray(result["loss"])),
        "region": float(np.asarray(result["region"])),
        "object_transport_raw": float(np.asarray(result["object_transport_raw"])),
        "mass_conservation_raw": float(np.asarray(result["mass_conservation_raw"])),
        "object_mass_error": float(np.asarray(result["object_mass_error"])),
        "background_mass_error": float(np.asarray(result["background_mass_error"])),
        "mean_predicted_object_mass": float(np.asarray(result["mean_predicted_object_mass"])),
        "endpoint_error_mm": float(np.asarray(result["endpoint_error_vox"])),
        "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "decoder_count": decoded.count,
        "decoder_mass": decoded.total_atom_mass,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
