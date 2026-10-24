from __future__ import annotations

import numpy as np
import torch

from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor
from nnunetv2.nets.microbt_ctf.model import MicroBTCTFNetwork


def raw_logits_to_ctf(raw_logits: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor]:
    logits = torch.as_tensor(raw_logits).float()
    if logits.ndim not in (4, 5):
        raise ValueError("CTF raw logits must have shape [C,D,H,W] or [B,C,D,H,W]")
    channel_axis = 0 if logits.ndim == 4 else 1
    if logits.shape[channel_axis] != 6:
        raise ValueError("CTF inference requires exactly six raw-logit channels")
    if logits.ndim == 4:
        support_logit = logits[0:1]
        transport = torch.tanh(logits[1:4])
        core_logit = logits[4:5]
        enhancing_logit = logits[5:6]
    else:
        support_logit = logits[:, 0:1]
        transport = torch.tanh(logits[:, 1:4])
        core_logit = logits[:, 4:5]
        enhancing_logit = logits[:, 5:6]
    whole_tumor = torch.sigmoid(support_logit)
    tumor_core = whole_tumor * torch.sigmoid(core_logit)
    enhancing_tumor = tumor_core * torch.sigmoid(enhancing_logit)
    return {
        "transport": transport,
        "region_probabilities": torch.cat(
            (whole_tumor, tumor_core, enhancing_tumor), dim=channel_axis
        ),
    }


class MicroBTCTFPredictor(MicroBTPFPredictor):
    def _get_num_output_channels_for_sliding_window(self) -> int:
        module = self.network
        while hasattr(module, "module") or hasattr(module, "_orig_mod"):
            module = module.module if hasattr(module, "module") else module._orig_mod
        if isinstance(module, MicroBTCTFNetwork):
            return 6
        return super()._get_num_output_channels_for_sliding_window()
