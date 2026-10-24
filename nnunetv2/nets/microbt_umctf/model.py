from __future__ import annotations

import torch
from torch import nn

from nnunetv2.nets.microbt_ctf.model import MicroBTCTFNetwork


def umctf_logits_to_outputs(field_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode nested regions, centroid transport, and WT-gated object density."""
    if field_logits.ndim < 2 or field_logits.shape[1] != 7:
        raise ValueError("Unit-Mass CTF logits must have exactly seven channels")
    whole_tumor = torch.sigmoid(field_logits[:, 0:1])
    transport = torch.tanh(field_logits[:, 1:4])
    tumor_core = whole_tumor * torch.sigmoid(field_logits[:, 4:5])
    enhancing_tumor = tumor_core * torch.sigmoid(field_logits[:, 5:6])
    object_density = whole_tumor * torch.sigmoid(field_logits[:, 6:7])
    return {
        "transport": transport,
        "object_density": object_density,
        "region_probabilities": torch.cat(
            (whole_tumor, tumor_core, enhancing_tumor), dim=1
        ),
    }


class MicroBTUnitMassNetwork(MicroBTCTFNetwork):
    """nnU-Net wrapper for a conserved lesion measure transported to centroids."""

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        field_logits = self.unet(image)
        if isinstance(field_logits, (list, tuple)):
            raise RuntimeError("Deep supervision is not supported by Unit-Mass CTF")
        if not isinstance(field_logits, torch.Tensor):
            raise TypeError("The wrapped nnU-Net must return a tensor")
        return {"field_logits": field_logits, **umctf_logits_to_outputs(field_logits)}

