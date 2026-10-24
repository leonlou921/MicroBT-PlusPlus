from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nnunetv2.inference.microbt_pf_export import decode_microbt_spp_cropped
from nnunetv2.inference.microbt_pf_predictor import (
    MicroBTPFPredictor,
    raw_logits_to_microbt_spp,
)
from nnunetv2.nets.microbt_spp.model import MicroBTSPPNetwork


def test_spp_inference_couples_after_raw_logit_restoration() -> None:
    logits = torch.tensor(
        [
            [[[0.0]]],
            [[[1.0]]],
            [[[0.0]]],
            [[[0.0]]],
        ]
    )

    output = raw_logits_to_microbt_spp(logits)

    expected_field = -1.0 + 0.5 * (1.0 + torch.tanh(logits[1]))
    torch.testing.assert_close(output["birth_field"], expected_field)
    torch.testing.assert_close(
        output["region_probabilities"],
        torch.tensor([[[[0.5]]], [[[0.25]]], [[[0.125]]]]),
    )


def test_spp_cropped_semantic_union_equals_support_and_instance_union() -> None:
    birth = np.asarray([[[-0.9, -0.2, -1.0, -0.4, -0.8]]], dtype=np.float32)
    regions = np.asarray(
        [
            [[[0.9, 0.8, 0.1, 0.7, 0.9]]],
            [[[0.8, 0.7, 0.0, 0.6, 0.8]]],
            [[[0.7, 0.6, 0.0, 0.5, 0.7]]],
        ],
        dtype=np.float32,
    )

    semantic, decoded = decode_microbt_spp_cropped(birth, regions)

    support = regions[0] >= 0.5
    np.testing.assert_array_equal(semantic > 0, support)
    np.testing.assert_array_equal(decoded["instance_ids"] > 0, support)
    assert decoded["support_conserved"] is True


def test_spp_predictor_reports_four_raw_channels_and_detects_contract() -> None:
    predictor = object.__new__(MicroBTPFPredictor)
    predictor.label_manager = type("Labels", (), {"num_segmentation_heads": 3})()
    predictor.network = MicroBTSPPNetwork(nn.Identity())

    assert predictor._get_num_output_channels_for_sliding_window() == 4
    assert predictor.uses_support_preserving_persistence() is True
