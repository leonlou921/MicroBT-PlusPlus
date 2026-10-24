from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from nnunetv2.nets.microbt_pf.model import (
    field_logits_to_region_probabilities,
    sbpf_logits_to_outputs,
)
from nnunetv2.training.nnUNetTrainer.microbt_sbpf_checkpoint import (
    map_three_channel_head_tensor,
    migrate_a3_network_weights,
)


def test_three_to_four_channel_head_mapping_is_functionally_exact() -> None:
    torch.manual_seed(7)
    old_head = nn.Conv3d(5, 3, kernel_size=1)
    new_head = nn.Conv3d(5, 4, kernel_size=1)
    with torch.no_grad():
        new_head.weight.copy_(map_three_channel_head_tensor(old_head.weight))
        new_head.bias.copy_(map_three_channel_head_tensor(old_head.bias))
    features = torch.randn(2, 5, 3, 4, 5)

    old_logits = old_head(features)
    new_logits = new_head(features)
    old_regions = field_logits_to_region_probabilities(old_logits)
    new_output = sbpf_logits_to_outputs(new_logits)

    torch.testing.assert_close(new_output["region_probabilities"], old_regions)
    torch.testing.assert_close(new_output["birth_field"], torch.tanh(old_logits[:, 0:1]))


def test_state_dict_migration_maps_only_all_segmentation_heads() -> None:
    weights = OrderedDict(
        {
            "unet.encoder.weight": torch.randn(8, 4, 3, 3, 3),
            "unet.decoder.seg_layers.0.weight": torch.randn(3, 8, 1, 1, 1),
            "unet.decoder.seg_layers.0.bias": torch.randn(3),
            "unet.decoder.seg_layers.1.weight": torch.randn(3, 4, 1, 1, 1),
            "unet.decoder.seg_layers.1.bias": torch.randn(3),
        }
    )

    migrated, mapped_keys = migrate_a3_network_weights(weights)

    assert mapped_keys == (
        "unet.decoder.seg_layers.0.bias",
        "unet.decoder.seg_layers.0.weight",
        "unet.decoder.seg_layers.1.bias",
        "unet.decoder.seg_layers.1.weight",
    )
    assert migrated["unet.decoder.seg_layers.0.weight"].shape[0] == 4
    assert migrated["unet.decoder.seg_layers.1.bias"].shape == (4,)
    assert migrated["unet.encoder.weight"] is weights["unet.encoder.weight"]


def test_migration_fails_closed_for_missing_or_already_migrated_heads() -> None:
    with pytest.raises(ValueError, match="no three-channel segmentation heads"):
        migrate_a3_network_weights({"unet.encoder.weight": torch.randn(8, 4)})

    with pytest.raises(ValueError, match="already has four channels"):
        migrate_a3_network_weights(
            {"unet.decoder.seg_layers.0.weight": torch.randn(4, 8, 1, 1, 1)}
        )
