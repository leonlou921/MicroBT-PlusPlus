from __future__ import annotations

import torch
from torch import nn


def field_logits_to_region_probabilities(field_logits: torch.Tensor) -> torch.Tensor:
    """Map the one persistent field and optional conditional marks to regions."""
    if field_logits.ndim < 2 or field_logits.shape[1] not in (1, 3):
        raise ValueError("Persistent-field logits must have one or three channels")

    field = torch.tanh(field_logits[:, 0:1])
    whole_tumor = (field + 1.0) * 0.5
    if field_logits.shape[1] == 1:
        return whole_tumor

    tumor_core = whole_tumor * torch.sigmoid(field_logits[:, 1:2])
    enhancing_tumor = tumor_core * torch.sigmoid(field_logits[:, 2:3])
    return torch.cat((whole_tumor, tumor_core, enhancing_tumor), dim=1)


class MicroBTPFNetwork(nn.Module):
    """A parameter-free contract wrapper around the official nnU-Net network."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        field_logits = self.unet(image)
        if isinstance(field_logits, (list, tuple)):
            raise RuntimeError("Deep supervision is not supported by MicroBT-PF")
        if not isinstance(field_logits, torch.Tensor):
            raise TypeError("The wrapped nnU-Net must return a tensor")
        return {
            "field_logits": field_logits,
            "region_probabilities": field_logits_to_region_probabilities(field_logits),
        }


def sbpf_logits_to_outputs(field_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Transform the four independent SB-PF logits into its public outputs.

    Channel order is semantic support, Morse birth, conditional tumour core,
    and conditional enhancing tumour. The transform is parameter free.
    """
    if field_logits.ndim < 2 or field_logits.shape[1] != 4:
        raise ValueError("SB-PF logits must have exactly four channels")

    whole_tumor = torch.sigmoid(field_logits[:, 0:1])
    birth_field = torch.tanh(field_logits[:, 1:2])
    tumor_core = whole_tumor * torch.sigmoid(field_logits[:, 2:3])
    enhancing_tumor = tumor_core * torch.sigmoid(field_logits[:, 3:4])
    return {
        "birth_field": birth_field,
        "region_probabilities": torch.cat(
            (whole_tumor, tumor_core, enhancing_tumor), dim=1
        ),
    }


class MicroBTSBPFNetwork(nn.Module):
    """Parameter-free wrapper for the four-channel support-birth contract."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        field_logits = self.unet(image)
        if isinstance(field_logits, (list, tuple)):
            raise RuntimeError("Deep supervision is not supported by MicroBT SB-PF")
        if not isinstance(field_logits, torch.Tensor):
            raise TypeError("The wrapped nnU-Net must return a tensor")
        return {"field_logits": field_logits, **sbpf_logits_to_outputs(field_logits)}
