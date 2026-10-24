from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nnunetv2.nets.microbt_ctf.model import MicroBTCTFNetwork
from nnunetv2.training.dataloading.microbt_ctf_targets import (
    build_centroid_transport_target,
)
from nnunetv2.training.dataloading.microbt_pf_loader import (
    build_microbt_pf_dataloaders,
)
from nnunetv2.training.loss.microbt_ctf_losses import MicroBTCTFAssignmentLoss, MicroBTCTFLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ExactFast,
)
from nnunetv2.training.nnUNetTrainer.MicroBTSPPTrainer import (
    MicroBTSPPTrainerScheduledExact,
    _ScratchSPPInitializationMixin,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def ctf_object_weight(epoch: int) -> float:
    """Ramp the single object family from zero to a bounded 0.25."""
    epoch = int(epoch)
    if not 0 <= epoch < 200:
        raise ValueError("epoch must be in [0, 200)")
    if epoch <= 4:
        return 0.0
    if epoch >= 29:
        return 0.25
    return 0.25 * float(epoch - 4) / 25.0


class MicroBTCTFTrainerScratch60(
    _ScratchSPPInitializationMixin, MicroBTSPPTrainerScheduledExact
):
    """Sixty-epoch scratch gate for the H0-free centroid transport paradigm."""

    scratch_seed = 42
    causal_epochs = 60
    training_epochs = 200
    snapshot_epochs = (15, 30, 40, 50, 60)
    initial_learning_rate = 1e-2
    full_exact_tail_epochs = 0
    validation_h0_audit_iterations = 0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.causal_epochs
        self.save_every = 5
        self.initial_lr = self.initial_learning_rate

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
            raise RuntimeError("MicroBT-CTF requires the three-region label contract")
        unet = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            6,
            enable_deep_supervision=False,
        )
        return MicroBTCTFNetwork(unet)

    def _build_loss(self) -> nn.Module:
        return MicroBTCTFLoss(object_weight=0.0)

    def get_dataloaders(self):
        return build_microbt_pf_dataloaders(
            self,
            component_uniform=True,
            precompute_training_topology=False,
            persistent_target_builder=build_centroid_transport_target,
        )

    def _initialize_exact_fast_state(self) -> None:
        MicroBTPFTrainerA3ExactFast._initialize_exact_fast_state(self)
        self._validation_monitor_loss = MicroBTCTFLoss(object_weight=0.0)

    @classmethod
    def exact_h0_schedule(
        cls, epoch: int, step: int
    ) -> tuple[str, bool, float]:
        if not 0 <= int(epoch) < cls.training_epochs:
            raise ValueError(f"epoch must be in [0, {cls.training_epochs})")
        if not 0 <= int(step) < 250:
            raise ValueError("training step must be in [0, 250)")
        return "ctf_no_h0", False, 0.0

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.training_epochs)
        return optimizer, scheduler

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weight = ctf_object_weight(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, MicroBTCTFLoss):
                raise TypeError("MicroBT-CTF requires MicroBTCTFLoss")
            criterion.set_object_weight(weight)
        self._current_ctf_weight = weight
        return weight

    def on_train_epoch_start(self):
        weight = self._set_object_morse_weight_for_current_epoch()
        self.print_to_log_file(
            f"MicroBT-CTF contract: epoch={int(self.current_epoch)}, "
            f"object_weight={weight:.6f}, exact_h0=disabled, checkpoint=none"
        )
        return super().on_train_epoch_start()

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        summary = {"epoch": int(self.current_epoch), "schema_version": 1}
        for key in (
            "region",
            "object_transport",
            "object_transport_raw",
            "transport_offset",
            "center_anchor",
            "endpoint_error_vox",
            "object_weight",
        ):
            values = np.asarray([np.asarray(item[key]).item() for item in train_outputs])
            summary[f"{key}_mean"] = float(values.mean())
        self.print_to_log_file("MicroBT-CTF component telemetry: " + json.dumps(summary, sort_keys=True))
        return result
    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        module = self.network.module if self.is_ddp else self.network
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        if not isinstance(module, MicroBTCTFNetwork):
            raise TypeError("MicroBT-CTF requires MicroBTCTFNetwork")
        unet = module.unet
        while hasattr(unet, "_orig_mod"):
            unet = unet._orig_mod
        if not hasattr(unet, "decoder"):
            raise AttributeError("wrapped nnU-Net has no decoder")
        unet.decoder.deep_supervision = False

    def on_epoch_end(self):
        completed_epoch = int(self.current_epoch) + 1
        result = super().on_epoch_end()
        if completed_epoch in self.snapshot_epochs:
            self.current_epoch -= 1
            try:
                destination = Path(self.output_folder) / f"checkpoint_phase_{completed_epoch:03d}.pth"
                self.save_checkpoint(str(destination))
                self.print_to_log_file(f"MicroBT-CTF snapshot saved: {destination.name}")
            finally:
                self.current_epoch += 1
        return result


class MicroBTCTFAssignmentTrainerScratch60(MicroBTCTFTrainerScratch60):
    """Scratch CTF gate with equal-instance assignment consistency."""

    def _build_loss(self) -> nn.Module:
        return MicroBTCTFAssignmentLoss(object_weight=0.0)

    def _initialize_exact_fast_state(self) -> None:
        super()._initialize_exact_fast_state()
        self._validation_monitor_loss = MicroBTCTFAssignmentLoss(object_weight=0.0)
