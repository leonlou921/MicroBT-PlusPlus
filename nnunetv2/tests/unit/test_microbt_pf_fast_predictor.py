from __future__ import annotations

import pytest
import torch
from torch import nn

from nnunetv2.inference.microbt_pf_fast_predictor import (
    mirror_axis_combinations,
    predict_mirrored_microbatches,
)
from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor


class _AsymmetricDictNetwork(nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        ramp = torch.arange(image.shape[-1], device=image.device, dtype=image.dtype)
        return {"field_logits": image[:, :3] + ramp, "ignored": image[:, :1] + 99}


def test_mirror_axis_combinations_are_canonical() -> None:
    assert mirror_axis_combinations((0, 1, 2), 3) == (
        (),
        (2,),
        (3,),
        (4,),
        (2, 3),
        (2, 4),
        (3, 4),
        (2, 3, 4),
    )


def test_microbatch_matches_serial_dict_output_exactly() -> None:
    image = torch.arange(3 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 3, 2, 3, 4)
    network = _AsymmetricDictNetwork()
    unwrap = MicroBTPFPredictor._unwrap_raw_logits
    reference = predict_mirrored_microbatches(network, image, unwrap, (0, 1, 2), 1)
    actual = predict_mirrored_microbatches(network, image, unwrap, (0, 1, 2), 4)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_no_mirroring_is_one_network_call() -> None:
    class CountingNetwork(_AsymmetricDictNetwork):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            self.calls += 1
            return super().forward(image)

    network = CountingNetwork()
    image = torch.zeros((1, 3, 2, 2, 2))
    result = predict_mirrored_microbatches(
        network, image, MicroBTPFPredictor._unwrap_raw_logits, None, 8
    )
    assert network.calls == 1
    assert result.shape == image.shape


@pytest.mark.parametrize("batch_size", (0, -1))
def test_microbatch_size_must_be_positive(batch_size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        predict_mirrored_microbatches(
            _AsymmetricDictNetwork(),
            torch.zeros((1, 3, 2, 2, 2)),
            MicroBTPFPredictor._unwrap_raw_logits,
            (0,),
            batch_size,
        )


def test_mirror_axis_must_fit_spatial_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensionality"):
        mirror_axis_combinations((3,), 3)
