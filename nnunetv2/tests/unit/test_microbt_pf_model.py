from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from nnunetv2.nets.microbt_pf.model import (
    MicroBTPFNetwork,
    MicroBTSBPFNetwork,
    field_logits_to_region_probabilities,
    sbpf_logits_to_outputs,
)


class TinyUNet(nn.Module):
    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.output = nn.Conv3d(1, output_channels, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output(image)


class DeepSupervisionUNet(nn.Module):
    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        prediction = image[:, :1]
        return [prediction, prediction]


def test_three_channel_transform_is_exact_and_hierarchical() -> None:
    logits = torch.tensor(
        [[[[[0.0]]], [[[0.0]]], [[[0.0]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    probabilities = field_logits_to_region_probabilities(logits)

    expected = torch.tensor([[[[[0.5]]], [[[0.25]]], [[[0.125]]]]])
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.shape == logits.shape
    assert torch.all(probabilities[:, 2] <= probabilities[:, 1])
    assert torch.all(probabilities[:, 1] <= probabilities[:, 0])

    probabilities.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_binary_transform_returns_only_whole_tumor_probability() -> None:
    logits = torch.tensor([[[[[-2.0, 0.0, 2.0]]]]])

    probabilities = field_logits_to_region_probabilities(logits)

    expected = (torch.tanh(logits) + 1.0) * 0.5
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.shape == logits.shape


def test_two_channel_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match="one or three channels"):
        field_logits_to_region_probabilities(torch.zeros(1, 2, 2, 2, 2))


def test_wrapper_returns_exact_contract_without_adding_parameters() -> None:
    unet = TinyUNet(output_channels=3)
    network = MicroBTPFNetwork(unet)
    image = torch.randn(2, 1, 3, 4, 5)

    output = network(image)

    assert set(output) == {"field_logits", "region_probabilities"}
    assert output["field_logits"].shape == (2, 3, 3, 4, 5)
    assert output["region_probabilities"].shape == (2, 3, 3, 4, 5)
    assert list(dict(network.named_children())) == ["unet"]
    assert sum(parameter.numel() for parameter in network.parameters()) == sum(
        parameter.numel() for parameter in unet.parameters()
    )


def test_wrapper_rejects_deep_supervision_output() -> None:
    network = MicroBTPFNetwork(DeepSupervisionUNet())

    with pytest.raises(RuntimeError, match="Deep supervision"):
        network(torch.zeros(1, 1, 2, 2, 2))


def test_sbpf_four_channel_transform_is_exact_and_decoupled() -> None:
    logits = torch.tensor(
        [[[[[0.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = sbpf_logits_to_outputs(logits)

    expected_regions = torch.tensor([[[[[0.5]]], [[[0.25]]], [[[0.125]]]]])
    torch.testing.assert_close(output["region_probabilities"], expected_regions)
    torch.testing.assert_close(output["birth_field"], torch.tanh(logits[:, 1:2]))
    assert output["birth_field"].shape == (1, 1, 1, 1, 1)
    assert torch.all(output["region_probabilities"][:, 2] <= output["region_probabilities"][:, 1])
    assert torch.all(output["region_probabilities"][:, 1] <= output["region_probabilities"][:, 0])

    output["birth_field"].sum().backward(retain_graph=True)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1]) == 1


def test_sbpf_transform_rejects_non_four_channel_logits() -> None:
    with pytest.raises(ValueError, match="exactly four channels"):
        sbpf_logits_to_outputs(torch.zeros(1, 3, 2, 2, 2))


def test_sbpf_wrapper_returns_explicit_birth_contract_without_parameters() -> None:
    unet = TinyUNet(output_channels=4)
    network = MicroBTSBPFNetwork(unet)
    image = torch.randn(2, 1, 3, 4, 5)

    output = network(image)

    assert set(output) == {"field_logits", "birth_field", "region_probabilities"}
    assert output["field_logits"].shape == (2, 4, 3, 4, 5)
    assert output["birth_field"].shape == (2, 1, 3, 4, 5)
    assert output["region_probabilities"].shape == (2, 3, 3, 4, 5)
    assert list(dict(network.named_children())) == ["unet"]
    assert sum(parameter.numel() for parameter in network.parameters()) == sum(
        parameter.numel() for parameter in unet.parameters()
    )


def test_sbpf_wrapper_rejects_deep_supervision_output() -> None:
    network = MicroBTSBPFNetwork(DeepSupervisionUNet())

    with pytest.raises(RuntimeError, match="Deep supervision"):
        network(torch.zeros(1, 1, 2, 2, 2))


@pytest.mark.skipif(
    not os.environ.get("MICROBT_PF_DATASET1_PLANS"),
    reason="Dataset001 plans are only available in the isolated integration environment",
)
def test_dataset001_plans_reconstruct_official_three_channel_resenc_m() -> None:
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    plans = json.loads(
        Path(os.environ["MICROBT_PF_DATASET1_PLANS"]).read_text(encoding="utf-8")
    )
    architecture = plans["configurations"]["3d_fullres"]["architecture"]
    unet = get_network_from_plans(
        arch_class_name=architecture["network_class_name"],
        arch_kwargs=architecture["arch_kwargs"],
        arch_kwargs_req_import=architecture["_kw_requires_import"],
        input_channels=4,
        output_channels=3,
        allow_init=False,
        deep_supervision=False,
    )
    network = MicroBTPFNetwork(unet)

    assert type(network.unet).__name__ == "ResidualEncoderUNet"
    assert network.unet.decoder.seg_layers[-1].out_channels == 3
    assert sum(parameter.numel() for parameter in network.parameters()) == sum(
        parameter.numel() for parameter in unet.parameters()
    )
