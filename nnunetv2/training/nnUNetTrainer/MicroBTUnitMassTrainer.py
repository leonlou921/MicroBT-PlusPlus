from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nnunetv2.nets.microbt_umctf.model import MicroBTUnitMassNetwork
from nnunetv2.training.loss.microbt_umctf_losses import MicroBTUnitMassLoss
from nnunetv2.training.nnUNetTrainer.MicroBTCTFTrainer import (
    MicroBTCTFAssignmentTrainerScratch60,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


ALLOWED_UMCTF_TARGET_EPOCHS = (60, 200, 300)


class MicroBTUnitMassTrainerScratch60(MicroBTCTFAssignmentTrainerScratch60):
    """Scratch Unit-Mass CTF gate; can continue the same trajectory to epoch 200."""

    snapshot_epochs = (15, 30, 40, 50, 60)
    continuation_snapshot_epochs = (100, 150, 200)

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        target_epochs = int(os.getenv("MICROBT_UMCTF_TARGET_EPOCHS", str(self.causal_epochs)))
        if target_epochs not in ALLOWED_UMCTF_TARGET_EPOCHS:
            allowed = ", ".join(str(value) for value in ALLOWED_UMCTF_TARGET_EPOCHS)
            raise ValueError(f"MICROBT_UMCTF_TARGET_EPOCHS must be one of: {allowed}")
        self.num_epochs = target_epochs

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: list[str] | tuple[str, ...],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if num_output_channels != 3:
            raise RuntimeError("Unit-Mass CTF requires the three-region label contract")
        unet = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            7,
            enable_deep_supervision=False,
        )
        return MicroBTUnitMassNetwork(unet)

    def _build_loss(self) -> nn.Module:
        return MicroBTUnitMassLoss(object_weight=0.0)

    def _initialize_exact_fast_state(self) -> None:
        super()._initialize_exact_fast_state()
        self._validation_monitor_loss = MicroBTUnitMassLoss(object_weight=0.0)

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weight = super()._set_object_morse_weight_for_current_epoch()
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, MicroBTUnitMassLoss):
                raise TypeError("Unit-Mass trainer requires MicroBTUnitMassLoss")
        return weight

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        module = self.network.module if self.is_ddp else self.network
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        if not isinstance(module, MicroBTUnitMassNetwork):
            raise TypeError("Unit-Mass CTF requires MicroBTUnitMassNetwork")
        unet = module.unet
        while hasattr(unet, "_orig_mod"):
            unet = unet._orig_mod
        unet.decoder.deep_supervision = False

    def on_train_epoch_end(self, train_outputs):
        # Call the parent chain once; add mass telemetry without duplicating CTF logs.
        result = super().on_train_epoch_end(train_outputs)
        summary = {"epoch": int(self.current_epoch), "schema_version": 1}
        for key in (
            "mass_conservation_raw", "object_mass_error",
            "background_mass_error", "mean_predicted_object_mass",
        ):
            values = np.asarray([np.asarray(item[key]).item() for item in train_outputs])
            summary[f"{key}_mean"] = float(values.mean())
        self.print_to_log_file(
            "MicroBT-UMCTF mass telemetry: " + json.dumps(summary, sort_keys=True)
        )
        return result

    def on_epoch_end(self):
        completed_epoch = int(self.current_epoch) + 1
        result = super().on_epoch_end()
        if completed_epoch in self.continuation_snapshot_epochs:
            self.current_epoch -= 1
            try:
                destination = Path(self.output_folder) / f"checkpoint_phase_{completed_epoch:03d}.pth"
                self.save_checkpoint(str(destination))
                self.print_to_log_file(f"MicroBT-UMCTF snapshot saved: {destination.name}")
            finally:
                self.current_epoch += 1
        return result
