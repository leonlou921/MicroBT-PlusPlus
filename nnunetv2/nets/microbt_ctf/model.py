from __future__ import annotations

import torch
from torch import nn


def ctf_logits_to_outputs(field_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode support, centroid displacement, and nested TC/ET probabilities."""
    if field_logits.ndim < 2 or field_logits.shape[1] != 6:
        raise ValueError("CTF logits must have exactly six channels")
    whole_tumor = torch.sigmoid(field_logits[:, 0:1])
    transport = torch.tanh(field_logits[:, 1:4])
    tumor_core = whole_tumor * torch.sigmoid(field_logits[:, 4:5])
    enhancing_tumor = tumor_core * torch.sigmoid(field_logits[:, 5:6])
    return {
        "transport": transport,
        "region_probabilities": torch.cat(
            (whole_tumor, tumor_core, enhancing_tumor), dim=1
        ),
    }


class MicroBTCTFNetwork(nn.Module):
    """nnU-Net wrapper whose object representation is a centroid transport field."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        field_logits = self.unet(image)
        if isinstance(field_logits, (list, tuple)):
            raise RuntimeError("Deep supervision is not supported by MicroBT-CTF")
        if not isinstance(field_logits, torch.Tensor):
            raise TypeError("The wrapped nnU-Net must return a tensor")
        return {"field_logits": field_logits, **ctf_logits_to_outputs(field_logits)}
