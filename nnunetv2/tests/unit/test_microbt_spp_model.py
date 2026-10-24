from __future__ import annotations

import torch
from torch import nn

from nnunetv2.nets.microbt_spp.model import MicroBTSPPNetwork, spp_logits_to_outputs


def test_spp_field_is_support_conditioned_and_topology_cannot_train_support_head() -> None:
    logits = torch.tensor(
        [[[[[0.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = spp_logits_to_outputs(logits)

    support = torch.sigmoid(logits[:, 0:1])
    expected_field = -1.0 + support * (1.0 + torch.tanh(logits[:, 1:2]))
    torch.testing.assert_close(output["birth_field"], expected_field)
    torch.testing.assert_close(
        output["region_probabilities"],
        torch.tensor([[[[[0.5]]], [[[0.25]]], [[[0.125]]]]]),
    )

    output["birth_field"].sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1]) == 1


def test_spp_threshold_algebra_contains_every_admissible_birth() -> None:
    logits = torch.linspace(-8.0, 8.0, 257).reshape(1, 1, 1, 1, -1)
    four = torch.zeros((1, 4, 1, 1, logits.shape[-1]))
    four[:, 0:1] = logits
    four[:, 1:2] = 20.0

    output = spp_logits_to_outputs(four)
    admitted = output["birth_field"] >= 0.25
    support = output["region_probabilities"][:, 0:1]

    assert torch.all(support[admitted] >= 0.625 - 1e-6)


def test_spp_wrapper_adds_no_parameters() -> None:
    unet = nn.Conv3d(1, 4, 1)
    network = MicroBTSPPNetwork(unet)
    output = network(torch.zeros(2, 1, 3, 4, 5))

    assert set(output) == {"field_logits", "birth_field", "region_probabilities"}
    assert output["field_logits"].shape == (2, 4, 3, 4, 5)
    assert sum(parameter.numel() for parameter in network.parameters()) == sum(
        parameter.numel() for parameter in unet.parameters()
    )
