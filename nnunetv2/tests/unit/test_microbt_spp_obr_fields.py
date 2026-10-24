from __future__ import annotations

import numpy as np
import pytest
import torch

from nnunetv2.nets.microbt_spp.model import spp_logits_to_outputs
from nnunetv2.training.dataloading.microbt_pf_targets import (
    build_persistent_field_target,
)
from nnunetv2.training.loss.microbt_spp_obr import asymmetric_birth_fields


SHAPE = (5, 5, 5)
CORE = (2, 2, 2)


def _target():
    instance_ids = np.zeros(SHAPE, dtype=np.int32)
    instance_ids[CORE] = 1
    return build_persistent_field_target(
        instance_ids,
        centers_by_id={1: tuple(float(value) for value in CORE)},
        spacing_zyx=(1.0, 1.0, 1.0),
        precompute_topology=False,
    )


def test_asymmetric_fields_are_forward_equivalent_to_deployed_spp() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((1, 4, *SHAPE), generator=generator)

    positive, negative = asymmetric_birth_fields(logits, (_target(),))
    deployed = spp_logits_to_outputs(logits)["birth_field"].float()

    torch.testing.assert_close(positive, deployed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(negative, deployed, rtol=0.0, atol=0.0)


def test_negative_field_never_backpropagates_to_support_logit() -> None:
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)

    _, negative = asymmetric_birth_fields(logits, (_target(),))
    negative.sum().backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[:, 1]).item() == np.prod(SHAPE)


def test_positive_field_backpropagates_to_support_only_inside_birth_core() -> None:
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)

    positive, _ = asymmetric_birth_fields(logits, (_target(),))
    positive.sum().backward()

    assert logits.grad is not None
    support_gradient = logits.grad[0, 0]
    assert torch.count_nonzero(support_gradient).item() == 1
    assert support_gradient[CORE].item() > 0.0
    outside = support_gradient.clone()
    outside[CORE] = 0.0
    assert torch.count_nonzero(outside).item() == 0
    assert torch.count_nonzero(logits.grad[:, 1]).item() == np.prod(SHAPE)


def test_asymmetric_fields_reject_target_batch_mismatch() -> None:
    logits = torch.zeros((2, 4, *SHAPE))

    with pytest.raises(ValueError, match="target count"):
        asymmetric_birth_fields(logits, (_target(),))

