from __future__ import annotations

import random
import time
from itertools import cycle

import numpy as np
import torch
from torch import nn

from nnunetv2.nets.microbt_spp.model import MicroBTSPPNetwork
from nnunetv2.training.dataloading.microbt_pf_loader import (
    build_microbt_pf_dataloaders,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.microbt_spp_losses import (
    MicroBTSPPLoss,
    MicroBTSPPOBRLoss,
)
from nnunetv2.training.loss.microbt_spp_obr import asymmetric_birth_fields
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ExactFast,
    MicroBTSBPFTrainerScheduledExact,
    _MicroBTSBPFVariantCriterion,
    _trainer_autocast,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def set_microbt_spp_scratch_seed(seed: int) -> None:
    """Seed all RNGs before network construction; no checkpoint is consulted."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _ScratchSPPInitializationMixin:
    scratch_seed = 42

    def initialize(self) -> None:
        set_microbt_spp_scratch_seed(self.scratch_seed)
        super().initialize()
        if self.optimizer.state:
            raise RuntimeError("scratch SPP initialization requires empty optimizer state")
        self.print_to_log_file(
            "SPP scratch initialization: "
            f"seed={self.scratch_seed}, optimizer_state_entries=0, checkpoint=none"
        )


class _MicroBTSPPVariantCriterion(_MicroBTSBPFVariantCriterion):
    def __init__(self, h0_matching: bool) -> None:
        super().__init__(h0_matching)
        self.full = MicroBTSPPLoss()

    @staticmethod
    def _apply_object_morse_weight(
        values: dict[str, torch.Tensor], weight: float
    ) -> dict[str, torch.Tensor]:
        weighted = _MicroBTSBPFVariantCriterion._apply_object_morse_weight(
            values, weight
        )
        zero = values["loss"] * 0.0
        weighted["false_persistence"] = values.get(
            "false_persistence", zero
        ) * float(weight)
        return weighted


class _MicroBTSPPOBRVariantCriterion(_MicroBTSBPFVariantCriterion):
    """SPP-OBR criterion with every-step reservation and optional exact H0."""

    def __init__(self, h0_matching: bool) -> None:
        super().__init__(h0_matching)
        self.full = MicroBTSPPOBRLoss()
        self.birth_reservation = self.full.birth_reservation

    @staticmethod
    def _apply_object_morse_weight(
        values: dict[str, torch.Tensor], weight: float
    ) -> dict[str, torch.Tensor]:
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Object-Morse weight must be in [0, 1]")
        weighted = dict(values)
        for key in (
            "field",
            "betti_0",
            "false_persistence",
            "birth_reservation",
            "birth_existence",
            "birth_prominence",
            "object_morse",
            "persistent_field",
        ):
            weighted[key] = values[key] * weight
        for key in (
            "birth_reservation",
            "birth_existence",
            "birth_prominence",
        ):
            weighted[f"{key}_raw"] = values[key]
        weighted["loss"] = values["region"].float() + weighted["object_morse"]
        weighted["object_morse_weight"] = torch.as_tensor(
            weight, device=values["loss"].device, dtype=values["loss"].dtype
        )
        return weighted

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets,
    ) -> dict[str, torch.Tensor]:
        if self.h0_matching_enabled:
            topology_started = time.perf_counter()
            result = self.full(output, region_target, persistent_targets)
            self.last_betti_wall_seconds = time.perf_counter() - topology_started
        else:
            region = self.region_fidelity(
                output["region_probabilities"], region_target
            )
            field_logits = output["field_logits"]
            with torch.autocast(
                device_type=field_logits.device.type, enabled=False
            ):
                positive_field, negative_field = asymmetric_birth_fields(
                    field_logits, persistent_targets
                )
                field = self.equal_object_field(
                    negative_field, persistent_targets
                )
                birth_reservation, birth_existence, birth_prominence = (
                    self.birth_reservation(
                        positive_field, negative_field, persistent_targets
                    )
                )
                zero = field_logits.sum().float() * 0.0
                object_morse = field + birth_reservation
                total = region.float() + object_morse
            result = {
                "loss": total,
                "region": region,
                "object_morse": object_morse,
                "field": field,
                "betti_0": zero,
                "false_persistence": zero,
                "birth_reservation": birth_reservation,
                "birth_existence": birth_existence,
                "birth_prominence": birth_prominence,
            }
            self.last_betti_wall_seconds = 0.0
        result = {**result, "persistent_field": result["object_morse"]}
        return self._apply_object_morse_weight(result, self.object_morse_weight)


class _FixedValidationIterator:
    def __init__(self, batches: tuple[dict, ...]) -> None:
        if not batches:
            raise ValueError("fixed validation iterator requires at least one batch")
        self._iterator = cycle(batches)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        return next(self._iterator)

    def _finish(self) -> None:
        return None


class MicroBTSPPTrainerScheduledExact(MicroBTSBPFTrainerScheduledExact):
    """200-epoch support-preserving persistence training under the six-minute gate."""

    # Validation remains a dense, inference-only audit. Rebuilding cubical H0
    # there is redundant with the scheduled training audits and dominates time.
    validation_h0_audit_iterations = 0
    validation_iterations_per_epoch = 2
    training_epochs = 200
    full_exact_tail_epochs = 0
    initial_learning_rate = 3e-4
    object_morse_ramp_epochs = 10

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = self.initial_learning_rate
        self.num_val_iterations_per_epoch = self.validation_iterations_per_epoch

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
            raise RuntimeError("MicroBT-SPP requires the three-region label contract")
        unet = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            4,
            enable_deep_supervision=False,
        )
        return MicroBTSPPNetwork(unet)

    def _build_loss(self) -> nn.Module:
        return _MicroBTSPPVariantCriterion(h0_matching=True)

    def _initialize_exact_fast_state(self) -> None:
        MicroBTPFTrainerA3ExactFast._initialize_exact_fast_state(self)
        self._validation_monitor_loss = _MicroBTSPPVariantCriterion(
            h0_matching=False
        )

    @staticmethod
    def _apply_betti_weight(
        loss_values: dict[str, torch.Tensor], betti_weight: float
    ) -> dict[str, torch.Tensor]:
        if float(betti_weight) == 1.0:
            return loss_values
        weighted = dict(loss_values)
        weighted["betti_0"] = loss_values["betti_0"] * float(betti_weight)
        weighted["false_persistence"] = (
            loss_values["false_persistence"] * float(betti_weight)
        )
        weighted["birth_completeness"] = (
            loss_values["birth_completeness"] * float(betti_weight)
        )
        weighted["object_morse"] = (
            loss_values["field"]
            + weighted["betti_0"]
            + weighted["birth_completeness"]
        )
        weighted["persistent_field"] = weighted["object_morse"]
        weighted["loss"] = loss_values["region"].float() + weighted["object_morse"]
        return weighted

    @classmethod
    def object_morse_weight(cls, epoch: int) -> float:
        epoch = int(epoch)
        if not 0 <= epoch < cls.training_epochs:
            raise ValueError(f"epoch must be in [0, {cls.training_epochs})")
        if cls.object_morse_ramp_epochs <= 1:
            return 1.0
        progress = min(1.0, float(epoch) / float(cls.object_morse_ramp_epochs - 1))
        return 0.25 + 0.75 * progress

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weight = self.object_morse_weight(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, _MicroBTSPPVariantCriterion):
                raise TypeError("MicroBT-SPP requires its variant criterion")
            criterion.set_object_morse_weight(weight)
        return weight

    def on_train_epoch_start(self):
        weight = self._set_object_morse_weight_for_current_epoch()
        self.print_to_log_file(f"SPP object-family weight: {weight:.6f}")
        return super().on_train_epoch_start()

    def get_dataloaders(self):
        return build_microbt_pf_dataloaders(
            self,
            component_uniform=True,
            precompute_training_topology=False,
            semantic_validation=True,
        )

    def on_train_start(self):
        result = super().on_train_start()
        self._validation_source_loader = self.dataloader_val
        self._fixed_validation_batches = tuple(
            next(self._validation_source_loader)
            for _ in range(self.validation_iterations_per_epoch)
        )
        self.dataloader_val = _FixedValidationIterator(
            self._fixed_validation_batches
        )
        return result

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, (list, tuple)):
            raise RuntimeError("MicroBT-SPP validation requires one semantic target")
        target = target.to(self.device, non_blocking=True)
        validation_network = self.network
        was_training = bool(validation_network.training)
        validation_network.train(True)
        try:
            # Reuse the already compiled training forward. The surrounding
            # nnU-Net validation loop is no_grad, so re-enable autograd only
            # for the forward graph; no backward or optimizer step is called.
            with torch.enable_grad(), _trainer_autocast(self):
                output = validation_network(data)
                region = self._validation_monitor_loss.region_fidelity(
                    output["region_probabilities"], target
                )
        finally:
            validation_network.train(was_training)

        region_probabilities = self._region_probabilities(output)
        predicted = (region_probabilities > 0.5).long()
        axes = [0] + list(range(2, predicted.ndim))
        if self.label_manager.has_ignore_label:
            mask = ~target[:, -1:] if target.dtype == torch.bool else 1 - target[:, -1:]
            target_for_metrics = target[:, :-1]
        else:
            mask = None
            target_for_metrics = target
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted,
            target_for_metrics.to(dtype=torch.bool),
            axes=axes,
            mask=mask,
        )
        zero = region.float() * 0.0
        return {
            "loss": region.detach().cpu().numpy(),
            "region": region.detach().cpu().numpy(),
            "object_morse": zero.detach().cpu().numpy(),
            "persistent_field": zero.detach().cpu().numpy(),
            "field": zero.detach().cpu().numpy(),
            "betti_0": zero.detach().cpu().numpy(),
            "betti_0_raw": zero.detach().cpu().numpy(),
            "false_persistence": zero.detach().cpu().numpy(),
            "birth_completeness": zero.detach().cpu().numpy(),
            "birth_completeness_raw": zero.detach().cpu().numpy(),
            "h0_active": zero.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy(),
            "fp_hard": fp.detach().cpu().numpy(),
            "fn_hard": fn.detach().cpu().numpy(),
        }

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        module = self.network.module if self.is_ddp else self.network
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        if not isinstance(module, MicroBTSPPNetwork):
            raise TypeError("MicroBT-SPP requires MicroBTSPPNetwork")
        unet = module.unet
        while hasattr(unet, "_orig_mod"):
            unet = unet._orig_mod
        if not hasattr(unet, "decoder"):
            raise AttributeError("wrapped nnU-Net has no decoder")
        unet.decoder.deep_supervision = False


class MicroBTSPPOBRTrainerScheduledExact(MicroBTSPPTrainerScheduledExact):
    """SPP with every-step one-birth reservation and scheduled exact H0."""

    def _build_loss(self) -> nn.Module:
        return _MicroBTSPPOBRVariantCriterion(h0_matching=True)

    def _initialize_exact_fast_state(self) -> None:
        MicroBTPFTrainerA3ExactFast._initialize_exact_fast_state(self)
        self._validation_monitor_loss = _MicroBTSPPOBRVariantCriterion(
            h0_matching=False
        )

    @staticmethod
    def _apply_betti_weight(
        loss_values: dict[str, torch.Tensor], betti_weight: float
    ) -> dict[str, torch.Tensor]:
        if float(betti_weight) == 1.0:
            return loss_values
        weighted = dict(loss_values)
        weighted["betti_0"] = loss_values["betti_0"] * float(betti_weight)
        weighted["false_persistence"] = (
            loss_values["false_persistence"] * float(betti_weight)
        )
        weighted["object_morse"] = (
            loss_values["field"]
            + weighted["betti_0"]
            + loss_values["birth_reservation"]
        )
        weighted["persistent_field"] = weighted["object_morse"]
        weighted["loss"] = loss_values["region"].float() + weighted["object_morse"]
        return weighted

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weight = self.object_morse_weight(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, _MicroBTSPPOBRVariantCriterion):
                raise TypeError("MicroBT-SPP-OBR requires its variant criterion")
            criterion.set_object_morse_weight(weight)
        return weight

    def validation_step(self, batch: dict) -> dict:
        result = MicroBTSPPTrainerScheduledExact.validation_step(self, batch)
        result.pop("birth_completeness", None)
        result.pop("birth_completeness_raw", None)
        zero = result["object_morse"]
        result.update(
            {
                "birth_reservation": zero,
                "birth_reservation_raw": zero,
                "birth_existence": zero,
                "birth_existence_raw": zero,
                "birth_prominence": zero,
                "birth_prominence_raw": zero,
            }
        )
        return result


class MicroBTSPPOBRTrainerScratch200(
    _ScratchSPPInitializationMixin, MicroBTSPPOBRTrainerScheduledExact
):
    """Full 200-epoch SPP-OBR training from deterministic random initialization."""

    training_epochs = 200
    initial_learning_rate = 1e-2
    full_exact_tail_epochs = 0
    object_morse_ramp_epochs = 20

    @classmethod
    def object_morse_weight(cls, epoch: int) -> float:
        epoch = int(epoch)
        if not 0 <= epoch < cls.training_epochs:
            raise ValueError(f"epoch must be in [0, {cls.training_epochs})")
        return min(1.0, float(epoch) / float(cls.object_morse_ramp_epochs - 1))


class MicroBTSPPOBRTrainerScratchGate5(MicroBTSPPOBRTrainerScratch200):
    """Five scratch epochs while retaining the formal 200-epoch LR horizon."""

    gate_epochs = 5

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.gate_epochs
        self.save_every = 1

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = PolyLRScheduler(
            optimizer, self.initial_lr, self.training_epochs
        )
        return optimizer, scheduler


class MicroBTSPPTrainerGate5(MicroBTSPPTrainerScheduledExact):
    gate_epochs = 5

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.gate_epochs
        self.save_every = 1

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


class MicroBTSPPTrainerSmoke(MicroBTSPPTrainerScheduledExact):
    smoke_epochs = 2
    smoke_train_iterations = 20
    smoke_val_iterations = 2

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.smoke_epochs
        self.num_iterations_per_epoch = self.smoke_train_iterations
        self.num_val_iterations_per_epoch = self.smoke_val_iterations
        self.save_every = 1
