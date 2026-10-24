from __future__ import annotations

import torch
from torch import nn


def spp_logits_to_outputs(field_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map four logits to support-preserving semantic and persistent outputs.

    Topology receives a one-way conditioning signal: the support probability is
    detached only for the persistent field, while region fidelity retains its
    normal live gradient through the semantic hierarchy.
    """
    if field_logits.ndim < 2 or field_logits.shape[1] != 4:
        raise ValueError("SPP logits must have exactly four channels")

    whole_tumor = torch.sigmoid(field_logits[:, 0:1])
    raw_birth = torch.tanh(field_logits[:, 1:2])
    birth_field = -1.0 + whole_tumor.detach() * (1.0 + raw_birth)
    tumor_core = whole_tumor * torch.sigmoid(field_logits[:, 2:3])
    enhancing_tumor = tumor_core * torch.sigmoid(field_logits[:, 3:4])
    return {
        "birth_field": birth_field,
        "region_probabilities": torch.cat(
            (whole_tumor, tumor_core, enhancing_tumor), dim=1
        ),
    }


class MicroBTSPPNetwork(nn.Module):
    """Parameter-free wrapper implementing support-preserving persistence."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        field_logits = self.unet(image)
        if isinstance(field_logits, (list, tuple)):
            raise RuntimeError("Deep supervision is not supported by MicroBT-SPP")
        if not isinstance(field_logits, torch.Tensor):
            raise TypeError("The wrapped nnU-Net must return a tensor")
        return {"field_logits": field_logits, **spp_logits_to_outputs(field_logits)}
