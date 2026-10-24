from __future__ import annotations

import numpy as np
import torch

from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor
from nnunetv2.nets.microbt_umctf.model import MicroBTUnitMassNetwork


def raw_logits_to_umctf(raw_logits: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
    logits = torch.as_tensor(raw_logits).float()
    if logits.ndim not in (4, 5):
        raise ValueError("Unit-Mass CTF logits require [C,D,H,W] or [B,C,D,H,W]")
    channel_axis = 0 if logits.ndim == 4 else 1
    if logits.shape[channel_axis] != 7:
        raise ValueError("Unit-Mass CTF inference requires exactly seven channels")
    if logits.ndim == 4:
        whole_tumor = torch.sigmoid(logits[0:1])
        transport = torch.tanh(logits[1:4])
        tumor_core = whole_tumor * torch.sigmoid(logits[4:5])
        enhancing = tumor_core * torch.sigmoid(logits[5:6])
        density = whole_tumor * torch.sigmoid(logits[6:7])
    else:
        whole_tumor = torch.sigmoid(logits[:, 0:1])
        transport = torch.tanh(logits[:, 1:4])
        tumor_core = whole_tumor * torch.sigmoid(logits[:, 4:5])
        enhancing = tumor_core * torch.sigmoid(logits[:, 5:6])
        density = whole_tumor * torch.sigmoid(logits[:, 6:7])
    return {
        "transport": transport,
        "object_density": density,
        "region_probabilities": torch.cat(
            (whole_tumor, tumor_core, enhancing), dim=channel_axis
        ),
    }


class MicroBTUnitMassPredictor(MicroBTPFPredictor):
    def _get_num_output_channels_for_sliding_window(self) -> int:
        module = self.network
        while hasattr(module, "module") or hasattr(module, "_orig_mod"):
            module = module.module if hasattr(module, "module") else module._orig_mod
        if isinstance(module, MicroBTUnitMassNetwork):
            return 7
        return super()._get_num_output_channels_for_sliding_window()

