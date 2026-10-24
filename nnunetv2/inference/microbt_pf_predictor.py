from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from nnunetv2.inference.microbt_pf_export import export_microbt_pf_prediction
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.nets.microbt_pf.model import MicroBTPFNetwork, MicroBTSBPFNetwork
from nnunetv2.nets.microbt_spp.model import MicroBTSPPNetwork


def raw_logits_to_microbt_pf(
    raw_logits: torch.Tensor | np.ndarray, a0: bool = False
) -> dict[str, torch.Tensor]:
    logits = torch.as_tensor(raw_logits).float()
    if logits.ndim not in (4, 5):
        raise ValueError("raw logits must have shape [C,D,H,W] or [B,C,D,H,W]")
    channel_axis = 0 if logits.ndim == 4 else 1
    if logits.shape[channel_axis] != 3:
        raise ValueError("MicroBT-PF inference requires exactly three raw-logit channels")
    if logits.ndim == 4:
        whole_tumor_logit, core_logit, enhancing_logit = logits
    else:
        whole_tumor_logit = logits[:, 0]
        core_logit = logits[:, 1]
        enhancing_logit = logits[:, 2]

    if a0:
        field = 2.0 * torch.sigmoid(whole_tumor_logit) - 1.0
        region_probabilities = torch.sigmoid(logits)
    else:
        field = torch.tanh(whole_tumor_logit)
        whole_tumor = (field + 1.0) * 0.5
        tumor_core = whole_tumor * torch.sigmoid(core_logit)
        enhancing_tumor = tumor_core * torch.sigmoid(enhancing_logit)
        region_probabilities = torch.stack(
            (whole_tumor, tumor_core, enhancing_tumor), dim=channel_axis
        )
    return {
        "field": field,
        "region_probabilities": region_probabilities,
    }


def raw_logits_to_microbt_sbpf(
    raw_logits: torch.Tensor | np.ndarray,
) -> dict[str, torch.Tensor]:
    """Convert raw four-channel SB-PF logits without coupling support and births."""
    logits = torch.as_tensor(raw_logits).float()
    if logits.ndim not in (4, 5):
        raise ValueError("raw logits must have shape [C,D,H,W] or [B,C,D,H,W]")
    channel_axis = 0 if logits.ndim == 4 else 1
    if logits.shape[channel_axis] != 4:
        raise ValueError("MicroBT SB-PF inference requires exactly four raw-logit channels")
    if logits.ndim == 4:
        support_logit, birth_logit, core_logit, enhancing_logit = logits
    else:
        support_logit = logits[:, 0]
        birth_logit = logits[:, 1]
        core_logit = logits[:, 2]
        enhancing_logit = logits[:, 3]

    whole_tumor = torch.sigmoid(support_logit)
    birth_field = torch.tanh(birth_logit)
    tumor_core = whole_tumor * torch.sigmoid(core_logit)
    enhancing_tumor = tumor_core * torch.sigmoid(enhancing_logit)
    return {
        "birth_field": birth_field,
        "region_probabilities": torch.stack(
            (whole_tumor, tumor_core, enhancing_tumor), dim=channel_axis
        ),
    }


def raw_logits_to_microbt_spp(
    raw_logits: torch.Tensor | np.ndarray,
) -> dict[str, torch.Tensor]:
    """Convert four SPP logits after restoration so containment stays exact."""
    logits = torch.as_tensor(raw_logits).float()
    if logits.ndim not in (4, 5):
        raise ValueError("raw logits must have shape [C,D,H,W] or [B,C,D,H,W]")
    channel_axis = 0 if logits.ndim == 4 else 1
    if logits.shape[channel_axis] != 4:
        raise ValueError("MicroBT-SPP inference requires exactly four raw-logit channels")
    if logits.ndim == 4:
        support_logit, birth_logit, core_logit, enhancing_logit = logits
    else:
        support_logit = logits[:, 0]
        birth_logit = logits[:, 1]
        core_logit = logits[:, 2]
        enhancing_logit = logits[:, 3]
    whole_tumor = torch.sigmoid(support_logit)
    birth_field = -1.0 + whole_tumor * (1.0 + torch.tanh(birth_logit))
    tumor_core = whole_tumor * torch.sigmoid(core_logit)
    enhancing_tumor = tumor_core * torch.sigmoid(enhancing_logit)
    return {
        "birth_field": birth_field,
        "region_probabilities": torch.stack(
            (whole_tumor, tumor_core, enhancing_tumor), dim=channel_axis
        ),
    }


class MicroBTPFPredictor(nnUNetPredictor):
    def _get_num_output_channels_for_sliding_window(self) -> int:
        module = self.network
        while True:
            if hasattr(module, "module"):
                module = module.module
                continue
            if hasattr(module, "_orig_mod"):
                module = module._orig_mod
                continue
            break
        if isinstance(module, (MicroBTSBPFNetwork, MicroBTSPPNetwork)):
            return 4
        if isinstance(module, MicroBTPFNetwork):
            return 3
        return super()._get_num_output_channels_for_sliding_window()

    def uses_support_preserving_persistence(self) -> bool:
        module = self.network
        while True:
            if hasattr(module, "module"):
                module = module.module
                continue
            if hasattr(module, "_orig_mod"):
                module = module._orig_mod
                continue
            break
        return isinstance(module, MicroBTSPPNetwork)

    @staticmethod
    def _unwrap_raw_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(output, dict):
            output = output["field_logits"]
        if not isinstance(output, torch.Tensor):
            raise TypeError("MicroBT-PF network output must contain tensor field_logits")
        return output

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self._unwrap_raw_logits(self.network(x)).clone()
        if mirror_axes is not None:
            if max(mirror_axes) > x.ndim - 3:
                raise ValueError("mirror axes do not match the input dimensionality")
            tensor_axes = [axis + 2 for axis in mirror_axes]
            combinations = [
                combination
                for count in range(1, len(tensor_axes) + 1)
                for combination in itertools.combinations(tensor_axes, count)
            ]
            for axes in combinations:
                mirrored = self.network(torch.flip(x, axes))
                prediction += torch.flip(self._unwrap_raw_logits(mirrored), axes)
            prediction /= len(combinations) + 1
        return prediction

    def predict_microbt_pf_from_files(
        self,
        source: str | Sequence[Sequence[str]],
        output_folder: str,
        *,
        save_field: bool = False,
        a0: bool = False,
        overwrite: bool = True,
    ) -> list[dict]:
        if self.configuration_manager.previous_stage_name is not None:
            raise RuntimeError("MicroBT-PF inference does not support cascaded configurations")
        inputs, outputs, previous_stage = self._manage_input_and_output_lists(
            source,
            output_folder,
            None,
            overwrite,
            0,
            1,
            save_field,
        )
        if previous_stage is not None and any(item is not None for item in previous_stage):
            raise RuntimeError("MicroBT-PF inference does not accept previous-stage segmentations")
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
        results: list[dict] = []
        for case_files, output_truncated in zip(inputs, outputs):
            data, _, properties = preprocessor.run_case(
                case_files,
                None,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json,
            )
            raw_logits = self.predict_logits_from_preprocessed_data(
                torch.from_numpy(data)
            ).cpu()
            result = export_microbt_pf_prediction(
                raw_logits,
                properties,
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                output_truncated,
                save_field=save_field,
                a0=a0,
                spp=self.uses_support_preserving_persistence(),
            )
            results.append(result)
        return results
