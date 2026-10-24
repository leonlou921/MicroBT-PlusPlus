from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import autocast, nn

from nnunetv2.evaluation.microbt_pf_training_monitor import (
    summarize_component_outputs,
    upsert_jsonl,
)
from nnunetv2.evaluation.microbt_pf_live_monitor import FixedCaseMonitor
from nnunetv2.nets.microbt_pf.model import MicroBTPFNetwork, MicroBTSBPFNetwork
from nnunetv2.training.dataloading.microbt_pf_loader import (
    build_microbt_pf_dataloaders,
)
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    precompute_persistent_target_topology,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.microbt_pf_losses import (
    BettiMatching0Loss,
    EqualObjectFieldLoss,
    MicroBTSBPFLoss,
    RegionFidelityLoss,
)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class _MicroBTPFVariantCriterion(nn.Module):
    def __init__(
        self,
        conditional_field: bool,
        equal_object_field: bool,
        h0_matching: bool,
    ) -> None:
        super().__init__()
        self.conditional_field = bool(conditional_field)
        self.equal_object_field_enabled = bool(equal_object_field)
        self.h0_matching_enabled = bool(h0_matching)
        self.region_fidelity = RegionFidelityLoss()
        self.equal_object_field = EqualObjectFieldLoss()
        self.betti_matching_0 = BettiMatching0Loss()
        self.last_betti_wall_seconds = 0.0

    def forward(
        self,
        output: torch.Tensor | dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget] = (),
    ) -> dict[str, torch.Tensor]:
        if self.conditional_field:
            if not isinstance(output, dict):
                raise TypeError("conditional MicroBT-PF output must be a dictionary")
            field_logits = output["field_logits"]
            region_probabilities = output["region_probabilities"]
        else:
            if not isinstance(output, torch.Tensor):
                raise TypeError("A0 output must be the official nnU-Net tensor")
            field_logits = output
            region_probabilities = torch.sigmoid(output)

        region = self.region_fidelity(region_probabilities, region_target)
        zero = field_logits.sum().float() * 0.0
        field = zero
        betti_0 = zero
        if self.equal_object_field_enabled or self.h0_matching_enabled:
            if len(persistent_targets) != field_logits.shape[0]:
                raise ValueError("persistent targets are required for A1-A3")
            with torch.autocast(device_type=field_logits.device.type, enabled=False):
                bounded_field = torch.tanh(field_logits[:, 0:1].float())
                if self.equal_object_field_enabled:
                    field = self.equal_object_field(
                        bounded_field, persistent_targets
                    )
                if self.h0_matching_enabled:
                    topology_started = time.perf_counter()
                    betti_0 = self.betti_matching_0(
                        bounded_field, persistent_targets
                    )
                    self.last_betti_wall_seconds = (
                        time.perf_counter() - topology_started
                    )
        persistent_field = field + betti_0
        total = region.float() + persistent_field
        return {
            "loss": total,
            "region": region,
            "persistent_field": persistent_field,
            "field": field,
            "betti_0": betti_0,
        }


class _MicroBTSBPFVariantCriterion(nn.Module):
    """Scheduled SB-PF criterion with an exact and a dense-only path."""

    def __init__(self, h0_matching: bool) -> None:
        super().__init__()
        self.h0_matching_enabled = bool(h0_matching)
        self.full = MicroBTSBPFLoss()
        self.region_fidelity = RegionFidelityLoss()
        self.equal_object_field = EqualObjectFieldLoss()
        self.last_betti_wall_seconds = 0.0
        self.object_morse_weight = 1.0

    def set_object_morse_weight(self, weight: float) -> None:
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Object-Morse weight must be in [0, 1]")
        self.object_morse_weight = weight

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
            "birth_completeness",
            "object_morse",
            "persistent_field",
        ):
            weighted[key] = values[key] * weight
        weighted["loss"] = values["region"].float() + weighted["object_morse"]
        weighted["object_morse_weight"] = torch.as_tensor(
            weight, device=values["loss"].device, dtype=values["loss"].dtype
        )
        return weighted

    def forward(
        self,
        output: dict[str, torch.Tensor],
        region_target: torch.Tensor,
        persistent_targets: Sequence[PersistentFieldTarget],
    ) -> dict[str, torch.Tensor]:
        if self.h0_matching_enabled:
            topology_started = time.perf_counter()
            result = self.full(output, region_target, persistent_targets)
            self.last_betti_wall_seconds = time.perf_counter() - topology_started
        else:
            region = self.region_fidelity(output["region_probabilities"], region_target)
            field_logits = output["field_logits"]
            with torch.autocast(device_type=field_logits.device.type, enabled=False):
                field = self.equal_object_field(
                    output["birth_field"].float(), persistent_targets
                )
                zero = field_logits.sum().float() * 0.0
                object_morse = field
                total = region.float() + object_morse
            result = {
                "loss": total,
                "region": region,
                "object_morse": object_morse,
                "field": field,
                "betti_0": zero,
                "birth_completeness": zero,
            }
            self.last_betti_wall_seconds = 0.0
        # The legacy monitor consumes this alias; optimization uses object_morse.
        result = {**result, "persistent_field": result["object_morse"]}
        return self._apply_object_morse_weight(result, self.object_morse_weight)


def _require_three_region_contract(trainer: nnUNetTrainer) -> None:
    if not trainer.label_manager.has_regions:
        raise RuntimeError("MicroBT-PF requires nnU-Net region training")
    if trainer.label_manager.num_segmentation_heads != 3:
        raise RuntimeError("MicroBT-PF requires exactly WT, TC, and ET regions")


def _trainer_autocast(trainer: nnUNetTrainer):
    if trainer.device.type != "cuda":
        return dummy_context()
    dtype = getattr(trainer, "autocast_dtype", None)
    if dtype is None:
        return autocast(trainer.device.type, enabled=True)
    return autocast(trainer.device.type, enabled=True, dtype=dtype)


class MicroBTPFTrainerA0(nnUNetTrainer):
    conditional_field = False
    equal_object_field = False
    h0_matching = False
    component_sampling = False

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        _require_three_region_contract(self)
        self.enable_deep_supervision = False

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
            raise RuntimeError("MicroBT-PF requires exactly three network outputs")
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False,
        )

    def _build_loss(self) -> nn.Module:
        return _MicroBTPFVariantCriterion(
            conditional_field=self.conditional_field,
            equal_object_field=self.equal_object_field,
            h0_matching=self.h0_matching,
        )

    def _move_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, (list, tuple)):
            raise RuntimeError("MicroBT-PF does not accept deep-supervision targets")
        target = target.to(self.device, non_blocking=True)
        persistent_targets = tuple(batch.get("persistent_targets", ()))
        return data, target, persistent_targets

    def train_step(self, batch: dict) -> dict:
        data, target, persistent_targets = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with _trainer_autocast(self):
            output = self.network(data)
            loss_values = self.loss(output, target, persistent_targets)
            scalar_loss = loss_values["loss"]

        if self.grad_scaler is not None:
            self.grad_scaler.scale(scalar_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            scalar_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            key: value.detach().cpu().numpy() for key, value in loss_values.items()
        }

    @staticmethod
    def _region_probabilities(
        output: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(output, dict):
            return output["region_probabilities"]
        return torch.sigmoid(output)

    def validation_step(self, batch: dict) -> dict:
        data, target, persistent_targets = self._move_batch(batch)
        with _trainer_autocast(self):
            output = self.network(data)
            loss_values = self.loss(output, target, persistent_targets)
        region_probabilities = self._region_probabilities(output)
        predicted = (region_probabilities > 0.5).long()
        axes = [0] + list(range(2, predicted.ndim))

        if self.label_manager.has_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = 1 - target[:, -1:]
            target_for_metrics = target[:, :-1]
        else:
            mask = None
            target_for_metrics = target
        target_for_metrics = target_for_metrics.to(dtype=torch.bool)
        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted, target_for_metrics, axes=axes, mask=mask
        )
        result = {
            key: value.detach().cpu().numpy() for key, value in loss_values.items()
        }
        result.update(
            {
                "tp_hard": tp.detach().cpu().numpy(),
                "fp_hard": fp.detach().cpu().numpy(),
                "fn_hard": fn.detach().cpu().numpy(),
            }
        )
        return result


class MicroBTPFTrainerA1(nnUNetTrainer):
    conditional_field = True
    equal_object_field = True
    h0_matching = False
    component_sampling = False

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        _require_three_region_contract(self)
        self.enable_deep_supervision = False

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
            raise RuntimeError("MicroBT-PF requires exactly three network outputs")
        unet = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False,
        )
        return MicroBTPFNetwork(unet)

    def _build_loss(self) -> nn.Module:
        return _MicroBTPFVariantCriterion(
            conditional_field=self.conditional_field,
            equal_object_field=self.equal_object_field,
            h0_matching=self.h0_matching,
        )

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        module = self.network.module if self.is_ddp else self.network
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        if not isinstance(module, MicroBTPFNetwork):
            raise TypeError("conditional MicroBT-PF requires MicroBTPFNetwork")
        unet = module.unet
        while hasattr(unet, "_orig_mod"):
            unet = unet._orig_mod
        if not hasattr(unet, "decoder"):
            raise AttributeError("wrapped nnU-Net has no decoder")
        unet.decoder.deep_supervision = False

    _move_batch = MicroBTPFTrainerA0._move_batch
    train_step = MicroBTPFTrainerA0.train_step
    _region_probabilities = staticmethod(MicroBTPFTrainerA0._region_probabilities)
    validation_step = MicroBTPFTrainerA0.validation_step

    def get_dataloaders(self):
        return build_microbt_pf_dataloaders(
            self, component_uniform=self.component_sampling
        )


class MicroBTPFTrainerA2(MicroBTPFTrainerA1):
    conditional_field = True
    equal_object_field = True
    h0_matching = True
    component_sampling = False


class MicroBTPFTrainerA3(MicroBTPFTrainerA2):
    conditional_field = True
    equal_object_field = True
    h0_matching = True
    component_sampling = True


class MicroBTPFTrainerA3ExactFast(MicroBTPFTrainerA3):
    """Exact A3 execution path with prefetched target topology and audit telemetry."""

    validation_h0_audit_iterations = 5
    autocast_dtype = torch.bfloat16

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._initialize_exact_fast_state()

    def _initialize_exact_fast_state(self) -> None:
        self.grad_scaler = None
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self._exact_fast_validation_index = 0
        self._exact_fast_epoch_started: float | None = None
        self._validation_monitor_loss = _MicroBTPFVariantCriterion(
            conditional_field=True,
            equal_object_field=True,
            h0_matching=False,
        )
        self.exact_fast_telemetry = {
            "exact_h0_train_calls": 0,
            "validation_exact_h0_calls": 0,
            "target_precompute_wait_seconds": 0.0,
            "topology_wall_seconds": 0.0,
            "nonfinite_steps": 0,
            "skipped_steps": 0,
            "last_preclip_gradient_norm": 0.0,
            "epoch_wall_seconds": 0.0,
        }

    def _ensure_exact_fast_state(self) -> None:
        if not hasattr(self, "exact_fast_telemetry"):
            self._initialize_exact_fast_state()

    def _require_precomputed_topology(
        self, persistent_targets: Sequence[PersistentFieldTarget]
    ) -> None:
        self._ensure_exact_fast_state()
        started = time.perf_counter()
        try:
            for target in persistent_targets:
                if target.topology is None:
                    self.exact_fast_telemetry["skipped_steps"] += 1
                    raise RuntimeError(
                        "MicroBTPFTrainerA3ExactFast requires precomputed target topology"
                    )
        finally:
            self.exact_fast_telemetry["target_precompute_wait_seconds"] += (
                time.perf_counter() - started
            )

    @staticmethod
    def _apply_betti_weight(
        loss_values: dict[str, torch.Tensor], betti_weight: float
    ) -> dict[str, torch.Tensor]:
        if float(betti_weight) == 1.0:
            return loss_values
        weighted = dict(loss_values)
        weighted["betti_0"] = loss_values["betti_0"] * float(betti_weight)
        weighted["persistent_field"] = loss_values["field"] + weighted["betti_0"]
        weighted["loss"] = (
            loss_values["region"].float() + weighted["persistent_field"]
        )
        return weighted

    def _train_step_with_criterion(
        self,
        batch: dict,
        criterion: _MicroBTPFVariantCriterion,
        *,
        use_exact_h0: bool,
        betti_weight: float,
    ) -> dict:
        self._ensure_exact_fast_state()
        data, target, persistent_targets = self._move_batch(batch)
        if use_exact_h0:
            self._require_precomputed_topology(persistent_targets)
        self.optimizer.zero_grad(set_to_none=True)
        with _trainer_autocast(self):
            output = self.network(data)
            loss_values = criterion(output, target, persistent_targets)
            raw_betti_0 = loss_values["betti_0"]
            raw_birth_completeness = loss_values.get("birth_completeness")
            if use_exact_h0:
                loss_values = self._apply_betti_weight(loss_values, betti_weight)
            scalar_loss = loss_values["loss"]
        if use_exact_h0:
            self.exact_fast_telemetry["exact_h0_train_calls"] += 1
            self.exact_fast_telemetry["topology_wall_seconds"] += float(
                criterion.last_betti_wall_seconds
            )
        if not bool(torch.isfinite(scalar_loss.detach()).all()):
            self.exact_fast_telemetry["nonfinite_steps"] += 1
            self.exact_fast_telemetry["skipped_steps"] += 1
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite exact-fast training loss")

        if self.grad_scaler is not None:
            self.grad_scaler.scale(scalar_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 12
            )
        else:
            scalar_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 12
            )
        gradient_norm_value = float(gradient_norm.detach().cpu())
        self.exact_fast_telemetry["last_preclip_gradient_norm"] = gradient_norm_value
        if not np.isfinite(gradient_norm_value):
            self.exact_fast_telemetry["nonfinite_steps"] += 1
            self.exact_fast_telemetry["skipped_steps"] += 1
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite exact-fast gradient norm")
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        result = {
            key: value.detach().cpu().numpy() for key, value in loss_values.items()
        }
        result["betti_0_raw"] = raw_betti_0.detach().cpu().numpy()
        if raw_birth_completeness is not None:
            result["birth_completeness_raw"] = (
                raw_birth_completeness.detach().cpu().numpy()
            )
        result["h0_active"] = np.asarray(int(use_exact_h0), dtype=np.int64)
        return result

    def train_step(self, batch: dict) -> dict:
        return self._train_step_with_criterion(
            batch,
            self.loss,
            use_exact_h0=True,
            betti_weight=1.0,
        )

    def on_train_epoch_start(self):
        self._ensure_exact_fast_state()
        for key in (
            "exact_h0_train_calls",
            "target_precompute_wait_seconds",
            "topology_wall_seconds",
            "nonfinite_steps",
            "skipped_steps",
        ):
            self.exact_fast_telemetry[key] = 0 if key.endswith("calls") or key.endswith("steps") else 0.0
        self._exact_fast_epoch_started = time.perf_counter()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        if self._exact_fast_epoch_started is not None:
            self.exact_fast_telemetry["epoch_wall_seconds"] = (
                time.perf_counter() - self._exact_fast_epoch_started
            )
        self.print_to_log_file(
            "MicroBT-PF exact-fast telemetry: "
            + json.dumps(self.exact_fast_telemetry, sort_keys=True)
        )
        return result

    def on_validation_epoch_start(self):
        self._ensure_exact_fast_state()
        self._exact_fast_validation_index = 0
        self.exact_fast_telemetry["validation_exact_h0_calls"] = 0
        return super().on_validation_epoch_start()

    def validation_step(self, batch: dict) -> dict:
        self._ensure_exact_fast_state()
        use_exact_h0 = (
            self._exact_fast_validation_index
            < self.validation_h0_audit_iterations
        )
        self._exact_fast_validation_index += 1
        if use_exact_h0:
            audit_batch = dict(batch)
            audit_batch["persistent_targets"] = tuple(
                precompute_persistent_target_topology(target)
                for target in batch["persistent_targets"]
            )
            self.exact_fast_telemetry["validation_exact_h0_calls"] += 1
            result = super().validation_step(audit_batch)
            result["betti_0_raw"] = np.asarray(result["betti_0"])
            if "birth_completeness" in result:
                result["birth_completeness_raw"] = np.asarray(
                    result["birth_completeness"]
                )
            result["h0_active"] = np.asarray(1, dtype=np.int64)
            return result

        training_loss = self.loss
        self.loss = self._validation_monitor_loss
        try:
            result = super().validation_step(batch)
            result["betti_0_raw"] = np.asarray(0.0)
            if "birth_completeness" in result:
                result["birth_completeness_raw"] = np.asarray(0.0)
            result["h0_active"] = np.asarray(0, dtype=np.int64)
            return result
        finally:
            self.loss = training_loss


class MicroBTPFTrainerA3ScheduledExact(MicroBTPFTrainerA3ExactFast):
    """A3 with deterministic sparse exact H0 and a full-exact tail."""

    training_epochs = 200
    sparse_exact_interval = 10
    full_exact_tail_epochs = 3
    sparse_h0_weight = 10.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._initialize_scheduled_exact_state()

    @classmethod
    def exact_h0_schedule(
        cls, epoch: int, step: int
    ) -> tuple[str, bool, float]:
        epoch = int(epoch)
        step = int(step)
        if not 0 <= epoch < cls.training_epochs:
            raise ValueError(f"epoch must be in [0, {cls.training_epochs})")
        if not 0 <= step < 250:
            raise ValueError("training step must be in [0, 250)")
        tail_start = cls.training_epochs - cls.full_exact_tail_epochs
        if epoch >= tail_start:
            return "full_exact", True, 1.0
        use_exact_h0 = step % cls.sparse_exact_interval == 0
        return (
            "sparse_exact",
            use_exact_h0,
            cls.sparse_h0_weight if use_exact_h0 else 0.0,
        )

    def _new_scheduled_exact_telemetry(self) -> dict:
        epoch = int(getattr(self, "current_epoch", 0))
        mode, _, active_weight = self.exact_h0_schedule(epoch, 0)
        return {
            "schedule_mode": mode,
            "interval": 1 if mode == "full_exact" else self.sparse_exact_interval,
            "effective_h0_weight": active_weight,
            "tail_start_epoch": self.training_epochs - self.full_exact_tail_epochs,
            "training_steps": 0,
            "schedule_skips": 0,
        }

    def _initialize_scheduled_exact_state(self) -> None:
        self.num_epochs = self.training_epochs
        self.save_every = 25
        self._scheduled_train_step_index = 0
        self._fixed_case_monitor = None
        self.scheduled_exact_telemetry = self._new_scheduled_exact_telemetry()

    def _ensure_scheduled_exact_state(self) -> None:
        if not hasattr(self, "scheduled_exact_telemetry"):
            self._initialize_scheduled_exact_state()

    def train_step(self, batch: dict) -> dict:
        self._ensure_scheduled_exact_state()
        mode, use_exact_h0, betti_weight = self.exact_h0_schedule(
            self.current_epoch, self._scheduled_train_step_index
        )
        criterion = self.loss if use_exact_h0 else self._validation_monitor_loss
        training_batch = batch
        if use_exact_h0:
            topology_started = time.perf_counter()
            training_batch = dict(batch)
            training_batch["persistent_targets"] = tuple(
                precompute_persistent_target_topology(target)
                for target in batch["persistent_targets"]
            )
            self.exact_fast_telemetry["target_precompute_wait_seconds"] += (
                time.perf_counter() - topology_started
            )
        values = self._train_step_with_criterion(
            training_batch,
            criterion,
            use_exact_h0=use_exact_h0,
            betti_weight=betti_weight,
        )
        self._scheduled_train_step_index += 1
        self.scheduled_exact_telemetry["schedule_mode"] = mode
        self.scheduled_exact_telemetry["training_steps"] += 1
        if not use_exact_h0:
            self.scheduled_exact_telemetry["schedule_skips"] += 1
        return values

    def get_dataloaders(self):
        return build_microbt_pf_dataloaders(
            self,
            component_uniform=True,
            precompute_training_topology=False,
        )

    def on_train_start(self):
        result = super().on_train_start()
        manifest_path = os.getenv("MICROBT_PF_MONITOR_MANIFEST")
        output_root = os.getenv("MICROBT_PF_MONITOR_OUTPUT")
        if bool(manifest_path) != bool(output_root):
            raise RuntimeError(
                "MICROBT_PF_MONITOR_MANIFEST and MICROBT_PF_MONITOR_OUTPUT "
                "must be set together"
            )
        self._fixed_case_monitor = (
            FixedCaseMonitor.from_trainer(self, manifest_path, output_root)
            if manifest_path is not None
            else None
        )
        if self._fixed_case_monitor is not None:
            self.print_to_log_file(
                "MicroBT-PF fixed monitor preflight passed: "
                f"manifest={manifest_path}, output={output_root}"
            )
        return result

    def on_train_epoch_start(self):
        self._ensure_scheduled_exact_state()
        self._scheduled_train_step_index = 0
        self.scheduled_exact_telemetry = self._new_scheduled_exact_telemetry()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        component_summary = summarize_component_outputs(
            train_outputs,
            phase="train",
            epoch=int(self.current_epoch),
            schedule=self.scheduled_exact_telemetry,
        )
        upsert_jsonl(
            Path(self.output_folder) / "monitor" / "learning_components.jsonl",
            component_summary,
            key_fields=("epoch", "phase"),
        )
        self.print_to_log_file(
            "MicroBT-PF component telemetry: "
            + json.dumps(component_summary, sort_keys=True)
        )
        self.print_to_log_file(
            "MicroBT-PF scheduled-exact telemetry: "
            + json.dumps(self.scheduled_exact_telemetry, sort_keys=True)
        )
        return result

    def on_validation_epoch_end(self, val_outputs):
        result = super().on_validation_epoch_end(val_outputs)
        component_summary = summarize_component_outputs(
            val_outputs,
            phase="val",
            epoch=int(self.current_epoch),
            schedule=self.scheduled_exact_telemetry,
        )
        upsert_jsonl(
            Path(self.output_folder) / "monitor" / "learning_components.jsonl",
            component_summary,
            key_fields=("epoch", "phase"),
        )
        self.print_to_log_file(
            "MicroBT-PF component telemetry: "
            + json.dumps(component_summary, sort_keys=True)
        )
        return result

    def on_epoch_end(self):
        result = super().on_epoch_end()
        monitor = getattr(self, "_fixed_case_monitor", None)
        if monitor is not None:
            monitor.run_if_due(int(self.current_epoch), final=False)
        return result

    def on_train_end(self):
        result = super().on_train_end()
        monitor = getattr(self, "_fixed_case_monitor", None)
        if monitor is not None:
            monitor.run_if_due(int(self.current_epoch), final=True)
        return result


class MicroBTSBPFTrainerScheduledExact(MicroBTPFTrainerA3ScheduledExact):
    """Minimal support-birth PF trainer with the unchanged A3 exact-H0 schedule."""

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
            raise RuntimeError("MicroBT SB-PF requires the three-region label contract")
        unet = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            4,
            enable_deep_supervision=False,
        )
        return MicroBTSBPFNetwork(unet)

    def _build_loss(self) -> nn.Module:
        return _MicroBTSBPFVariantCriterion(h0_matching=True)

    def _initialize_exact_fast_state(self) -> None:
        MicroBTPFTrainerA3ExactFast._initialize_exact_fast_state(self)
        self._validation_monitor_loss = _MicroBTSBPFVariantCriterion(
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

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        module = self.network.module if self.is_ddp else self.network
        while hasattr(module, "_orig_mod"):
            module = module._orig_mod
        if not isinstance(module, MicroBTSBPFNetwork):
            raise TypeError("MicroBT SB-PF requires MicroBTSBPFNetwork")
        unet = module.unet
        while hasattr(unet, "_orig_mod"):
            unet = unet._orig_mod
        if not hasattr(unet, "decoder"):
            raise AttributeError("wrapped nnU-Net has no decoder")
        unet.decoder.deep_supervision = False

    def on_train_end(self):
        # The launcher immediately starts the 203-case evaluation. Do not insert
        # the legacy terminal fixed-case inference between training and evaluation.
        return MicroBTPFTrainerA3ExactFast.on_train_end(self)

    def perform_actual_validation(self, save_probabilities: bool = False) -> None:
        self.print_to_log_file(
            "SB-PF standard post-training validation skipped; "
            "external paired full203 owns evaluation"
        )


class MicroBTSBPFTrainerStableScheduledExact(MicroBTSBPFTrainerScheduledExact):
    """Conservative fine-tuning of the mature A3-equivalent initialization."""

    initial_learning_rate = 1e-3
    object_morse_ramp_epochs = 20

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

    @classmethod
    def object_morse_weight(cls, epoch: int) -> float:
        epoch = int(epoch)
        if not 0 <= epoch < cls.training_epochs:
            raise ValueError(f"epoch must be in [0, {cls.training_epochs})")
        return min(1.0, float(epoch + 1) / float(cls.object_morse_ramp_epochs))

    def _set_object_morse_weight_for_current_epoch(self) -> float:
        weight = self.object_morse_weight(int(self.current_epoch))
        for criterion in (self.loss, self._validation_monitor_loss):
            if not isinstance(criterion, _MicroBTSBPFVariantCriterion):
                raise TypeError("stable SB-PF requires its variant criterion")
            criterion.set_object_morse_weight(weight)
        return weight

    def on_train_epoch_start(self):
        weight = self._set_object_morse_weight_for_current_epoch()
        self.print_to_log_file(f"Object-Morse weight: {weight:.6f}")
        return super().on_train_epoch_start()


class MicroBTSBPFTrainerStableGate5(MicroBTSBPFTrainerStableScheduledExact):
    """Five full epochs with the formal 200-epoch LR and topology horizons."""

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


class MicroBTSBPFTrainerSmoke(MicroBTSBPFTrainerScheduledExact):
    """One-epoch real-data CUDA gate; not a scientific training variant."""

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
