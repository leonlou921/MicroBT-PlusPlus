from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.microbt_spp_checkpoint import (
    birth_initial_logit,
    map_three_channel_head_tensor,
)


def test_spp_checkpoint_mapping_preserves_region_heads_and_nearly_preserves_pf_field() -> None:
    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    bias = torch.tensor([-1.0, 0.5, 1.5])

    mapped_weight = map_three_channel_head_tensor(weight, is_bias=False)
    mapped_bias = map_three_channel_head_tensor(bias, is_bias=True)

    torch.testing.assert_close(mapped_weight[0], 2.0 * weight[0])
    torch.testing.assert_close(mapped_weight[1], torch.zeros_like(weight[0]))
    torch.testing.assert_close(mapped_weight[2:], weight[1:])
    torch.testing.assert_close(mapped_bias[[0, 2, 3]], torch.tensor([-2.0, 0.5, 1.5]))
    torch.testing.assert_close(mapped_bias[1], torch.tensor(birth_initial_logit))

    q = torch.linspace(-8.0, 8.0, 1001)
    pf_field = torch.tanh(q)
    support = torch.sigmoid(2.0 * q)
    spp_field = -1.0 + support * (1.0 + torch.tanh(torch.tensor(birth_initial_logit)))
    assert torch.max(torch.abs(spp_field - pf_field)) < 0.037
