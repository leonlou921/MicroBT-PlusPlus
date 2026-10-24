from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from nnunetv2.inference.microbt_pf_export import decode_microbt_sbpf_cropped
from nnunetv2.inference.microbt_pf_predictor import (
    MicroBTPFPredictor,
    raw_logits_to_microbt_sbpf,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.nets.microbt_pf.model import MicroBTPFNetwork, MicroBTSBPFNetwork


def test_sbpf_raw_logits_conversion_is_exact_for_unbatched_input() -> None:
    logits = torch.tensor(
        [
            [[[0.0]]],
            [[[1.0]]],
            [[[0.0]]],
            [[[0.0]]],
        ]
    )

    output = raw_logits_to_microbt_sbpf(logits)

    torch.testing.assert_close(output["birth_field"], torch.tanh(logits[1]))
    expected = torch.tensor([[[[0.5]]], [[[0.25]]], [[[0.125]]]])
    torch.testing.assert_close(output["region_probabilities"], expected)


def test_sbpf_raw_logits_conversion_supports_batch_and_rejects_wrong_channels() -> None:
    output = raw_logits_to_microbt_sbpf(np.zeros((2, 4, 3, 4, 5), dtype=np.float32))
    assert output["birth_field"].shape == (2, 3, 4, 5)
    assert output["region_probabilities"].shape == (2, 3, 3, 4, 5)

    with pytest.raises(ValueError, match="exactly four"):
        raw_logits_to_microbt_sbpf(torch.zeros(3, 2, 2, 2))


def test_sbpf_semantic_regions_are_clipped_to_marker_decoded_wt() -> None:
    birth = np.asarray([[[1.0, 0.2, -1.0]]], dtype=np.float32)
    regions = np.asarray(
        [
            [[[0.9, 0.9, 0.1]]],
            [[[0.8, 0.8, 0.8]]],
            [[[0.7, 0.7, 0.7]]],
        ],
        dtype=np.float32,
    )

    semantic, decoded = decode_microbt_sbpf_cropped(birth, regions)

    assert decoded["count"] == 1
    assert semantic.tolist() == [[[3, 3, 0]]]
    np.testing.assert_array_equal(semantic > 0, decoded["instance_ids"] > 0)


def test_sliding_window_channel_contract_uses_raw_pf_head_count() -> None:
    stock = object.__new__(nnUNetPredictor)
    stock.label_manager = type("Labels", (), {"num_segmentation_heads": 3})()
    assert stock._get_num_output_channels_for_sliding_window() == 3

    predictor = object.__new__(MicroBTPFPredictor)
    predictor.label_manager = stock.label_manager
    predictor.network = MicroBTPFNetwork(nn.Identity())
    assert predictor._get_num_output_channels_for_sliding_window() == 3

    predictor.network = MicroBTSBPFNetwork(nn.Identity())
    assert predictor._get_num_output_channels_for_sliding_window() == 4
