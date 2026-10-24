from __future__ import annotations

import torch

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_spp_obr import (
    OneBirthReservationLoss,
    asymmetric_birth_fields,
)


SHAPE = (3, 3, 5)


def _target(
    core_ids: torch.Tensor,
    moat_ids: torch.Tensor | None = None,
) -> PersistentFieldTarget:
    core_ids = core_ids.to(torch.int32)
    moat_ids = (
        torch.zeros_like(core_ids)
        if moat_ids is None
        else moat_ids.to(torch.int32)
    )
    instance_ids = torch.maximum(core_ids, moat_ids)
    valid_ids = tuple(int(value) for value in torch.unique(core_ids) if value > 0)
    centers = tuple(
        tuple(int(value) for value in torch.nonzero(core_ids == instance_id)[0])
        for instance_id in valid_ids
    )
    return PersistentFieldTarget(
        field=torch.full(core_ids.shape, -1.0),
        instance_ids=instance_ids,
        dense_valid_mask=torch.ones_like(core_ids, dtype=torch.bool),
        valid_instance_ids=valid_ids,
        centers_zyx=centers,
        birth_core_ids=core_ids,
        birth_moat_ids=moat_ids,
        precompute_topology=False,
    )


def test_missed_singleton_reservation_opens_support_only_at_gt_core() -> None:
    core_ids = torch.zeros(SHAPE, dtype=torch.int32)
    core_ids[1, 1, 2] = 1
    target = _target(core_ids)
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = -4.0

    positive, negative = asymmetric_birth_fields(logits, (target,))
    total, existence, prominence = OneBirthReservationLoss()(
        positive, negative, (target,)
    )
    total.backward()

    assert existence.item() > 0.0
    assert prominence.item() == 0.0
    assert logits.grad is not None
    support_gradient = logits.grad[0, 0]
    assert support_gradient[1, 1, 2].item() < 0.0
    outside = support_gradient.clone()
    outside[1, 1, 2] = 0.0
    assert torch.count_nonzero(outside).item() == 0


def test_two_objects_contribute_one_equal_existence_term_each() -> None:
    core_ids = torch.zeros(SHAPE, dtype=torch.int32)
    core_ids[1, 1, 1] = 4
    core_ids[1, 1, 3] = 9
    target = _target(core_ids)
    positive = torch.full((1, 1, *SHAPE), -1.0)
    negative = positive.clone()
    positive[0, 0, 1, 1, 1] = 0.50
    positive[0, 0, 1, 1, 3] = 0.00
    negative.copy_(positive)

    total, existence, prominence = OneBirthReservationLoss()(
        positive, negative, (target,)
    )

    torch.testing.assert_close(existence, torch.tensor(0.03125))
    torch.testing.assert_close(total, existence)
    assert prominence.item() == 0.0


def test_prominence_penalizes_a_core_that_is_not_above_its_moat() -> None:
    core_ids = torch.zeros(SHAPE, dtype=torch.int32)
    moat_ids = torch.zeros(SHAPE, dtype=torch.int32)
    core_ids[1, 1, 2] = 5
    moat_ids[1, 1, 1] = 5
    moat_ids[1, 1, 3] = 5
    target = _target(core_ids, moat_ids)
    positive = torch.full((1, 1, *SHAPE), -1.0)
    negative = positive.clone()
    positive[0, 0, 1, 1, 2] = 0.40
    negative[0, 0, 1, 1, 2] = 0.40
    negative[0, 0, 1, 1, 1] = 0.30
    negative[0, 0, 1, 1, 3] = 0.30

    total, existence, prominence = OneBirthReservationLoss()(
        positive, negative, (target,)
    )

    assert existence.item() == 0.0
    torch.testing.assert_close(prominence, torch.tensor(0.0225))
    torch.testing.assert_close(total, prominence)


def test_normalized_smooth_max_is_invariant_to_equal_core_voxel_count() -> None:
    one_core = torch.zeros(SHAPE, dtype=torch.int32)
    one_core[1, 1, 2] = 1
    many_core = one_core.clone()
    many_core[1, 1, 1] = 1
    many_core[1, 1, 3] = 1
    positive = torch.full((1, 1, *SHAPE), -1.0)
    negative = positive.clone()
    positive[0, 0, 1, 1, 1:4] = 0.0
    negative.copy_(positive)
    loss = OneBirthReservationLoss()

    one = loss(positive, negative, (_target(one_core),))[0]
    many = loss(positive, negative, (_target(many_core),))[0]

    torch.testing.assert_close(one, torch.tensor(0.0625))
    torch.testing.assert_close(many, one)


def test_empty_target_has_exactly_zero_birth_reservation() -> None:
    target = _target(torch.zeros(SHAPE, dtype=torch.int32))
    positive = torch.randn((1, 1, *SHAPE), requires_grad=True)
    negative = positive.detach().clone().requires_grad_()

    total, existence, prominence = OneBirthReservationLoss()(
        positive, negative, (target,)
    )

    assert total.item() == 0.0
    assert existence.item() == 0.0
    assert prominence.item() == 0.0

