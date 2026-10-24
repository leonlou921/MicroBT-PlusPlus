from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nnunetv2.nets.microbt_pf.model import field_logits_to_region_probabilities
from nnunetv2.topology.persistent_field_0d import (
    InducedMatching0D,
    PersistencePairs0D,
    critical_values,
    induced_matching_0d,
    induced_matching_0d_precomputed_target,
)
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    PersistentTargetTopology,
    build_persistent_field_target,
)
from nnunetv2.training.loss import microbt_pf_losses as losses


SHAPE = (3, 3, 5)


def _flat(z: int, y: int, x: int) -> int:
    return int(np.ravel_multi_index((z, y, x), SHAPE))


def _pairs(
    births: list[int], deaths: list[int], essential: list[bool] | None = None
) -> PersistencePairs0D:
    if essential is None:
        essential = [death < 0 for death in deaths]
    return PersistencePairs0D(
        birth_flat=np.asarray(births, dtype=np.uint32),
        death_flat=np.asarray(deaths, dtype=np.int64),
        essential=np.asarray(essential, dtype=bool),
    )


def _matching(
    prediction: PersistencePairs0D,
    target: PersistencePairs0D,
    matched_prediction: list[int] | None = None,
    matched_target: list[int] | None = None,
    unmatched_prediction: list[int] | None = None,
    unmatched_target: list[int] | None = None,
) -> InducedMatching0D:
    return InducedMatching0D(
        prediction=prediction,
        target=target,
        matched_prediction=np.asarray(matched_prediction or [], dtype=np.int64),
        matched_target=np.asarray(matched_target or [], dtype=np.int64),
        unmatched_prediction=np.asarray(unmatched_prediction or [], dtype=np.int64),
        unmatched_target=np.asarray(unmatched_target or [], dtype=np.int64),
    )


def _target(
    field: torch.Tensor,
    instance_ids: torch.Tensor | None = None,
    valid_instance_ids: tuple[int, ...] = (),
    dense_valid_mask: torch.Tensor | None = None,
) -> PersistentFieldTarget:
    if instance_ids is None:
        instance_ids = torch.zeros_like(field, dtype=torch.int32)
    if dense_valid_mask is None:
        dense_valid_mask = torch.ones_like(field, dtype=torch.bool)
    return PersistentFieldTarget(
        field=field.to(torch.float32),
        instance_ids=instance_ids.to(torch.int32),
        dense_valid_mask=dense_valid_mask.to(torch.bool),
        valid_instance_ids=valid_instance_ids,
        centers_zyx=(),
    )


def _target_with_pairs(
    target: PersistentFieldTarget, pairs: PersistencePairs0D
) -> PersistentFieldTarget:
    flat = target.field.numpy().ravel()
    birth_values = flat[pairs.birth_flat.astype(np.int64)].copy()
    death_values = np.full(pairs.death_flat.size, -1.0, dtype=np.float32)
    finite = pairs.death_flat >= 0
    death_values[finite] = flat[pairs.death_flat[finite]]
    instance_ids = target.instance_ids.numpy().ravel()
    valid_ids = frozenset(target.valid_instance_ids)
    branch_valid = np.asarray(
        [int(instance_ids[int(index)]) in valid_ids for index in pairs.birth_flat],
        dtype=bool,
    )
    return PersistentFieldTarget(
        field=target.field,
        instance_ids=target.instance_ids,
        dense_valid_mask=target.dense_valid_mask,
        valid_instance_ids=target.valid_instance_ids,
        centers_zyx=target.centers_zyx,
        topology=PersistentTargetTopology(
            pairs=pairs,
            birth_values=birth_values,
            death_values=death_values,
            branch_valid=branch_valid,
        ),
    )


def test_region_fidelity_is_mean_region_soft_dice_plus_bce() -> None:
    probability = torch.tensor([[[[[0.8, 0.2]]], [[[0.25, 0.75]]]]])
    target = torch.tensor([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]])
    epsilon = 1e-5

    observed = losses.RegionFidelityLoss(epsilon=epsilon)(probability, target)

    intersection = (probability * target).flatten(2).sum(-1)
    denominator = probability.flatten(2).sum(-1) + target.flatten(2).sum(-1)
    expected_dice = (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
    expected_bce = F.binary_cross_entropy(probability, target, reduction="none").flatten(2).mean(-1).mean()
    torch.testing.assert_close(observed, expected_dice + expected_bce)


def test_region_fidelity_accepts_hierarchy_compatible_probabilities() -> None:
    logits = torch.tensor([[[[[0.2]]], [[[0.7]]], [[[-0.3]]]]], requires_grad=True)
    probability = field_logits_to_region_probabilities(logits)
    assert bool(torch.all(probability[:, 0] >= probability[:, 1]))
    assert bool(torch.all(probability[:, 1] >= probability[:, 2]))
    value = losses.RegionFidelityLoss()(probability, torch.ones_like(probability))
    value.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_region_fidelity_is_safe_inside_cuda_bf16_autocast() -> None:
    logits = torch.randn((1, 3, 4, 4, 4), device="cuda", requires_grad=True)
    target = torch.randint(0, 2, logits.shape, device="cuda", dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        probability = field_logits_to_region_probabilities(logits)
        value = losses.RegionFidelityLoss()(probability, target)

    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_equal_object_field_weights_small_and_large_instances_equally() -> None:
    prediction = torch.tensor([[[[[1.0, 0.0, 0.0, 0.0, 0.0, -1.0]]]]])
    target_field = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]])
    instance_ids = torch.tensor([[[1, 2, 2, 2, 2, 0]]], dtype=torch.int32)
    target = _target(target_field, instance_ids, valid_instance_ids=(1, 2))

    observed = losses.EqualObjectFieldLoss()(prediction, (target,))

    torch.testing.assert_close(observed, torch.tensor(0.25))


def test_equal_object_field_is_half_foreground_half_background() -> None:
    prediction = torch.tensor([[[[[0.4, -0.8]]]]])
    target = _target(
        torch.tensor([[[0.0, -1.0]]]),
        torch.tensor([[[1, 0]]]),
        valid_instance_ids=(1,),
    )
    observed = losses.EqualObjectFieldLoss()(prediction, (target,))
    torch.testing.assert_close(observed, torch.tensor(0.3))


def test_equal_object_field_empty_case_uses_background_only() -> None:
    prediction = torch.tensor([[[[[-0.7, -0.7]]]]])
    target = _target(torch.full((1, 1, 2), -1.0))
    observed = losses.EqualObjectFieldLoss()(prediction, (target,))
    torch.testing.assert_close(observed, torch.tensor(0.3))


def test_dense_invalid_fragment_contributes_neither_object_nor_background() -> None:
    prediction = torch.tensor([[[[[0.0, 100.0, -1.0]]]]])
    target = _target(
        torch.tensor([[[0.0, -1.0, -1.0]]]),
        torch.tensor([[[1, 2, 0]]]),
        valid_instance_ids=(1,),
        dense_valid_mask=torch.tensor([[[True, False, True]]]),
    )
    observed = losses.EqualObjectFieldLoss()(prediction, (target,))
    torch.testing.assert_close(observed, torch.tensor(0.0))


def test_betti_matched_barcode_uses_squared_euclidean_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    prediction[0, 0, 1, 1, 1] = 0.8
    prediction[0, 0, 1, 1, 2] = -0.2
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    target_field[1, 1, 3] = -0.4
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(1, 1, 1)], [_flat(1, 1, 2)]),
        _pairs([_flat(1, 1, 2)], [_flat(1, 1, 3)]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    observed = losses.BettiMatching0Loss()(prediction, (target,))

    torch.testing.assert_close(observed, torch.tensor(0.08), atol=1e-6, rtol=1e-6)


def test_betti_both_unmatched_directions_and_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    prediction[0, 0, 1, 1, 1] = 0.8
    prediction[0, 0, 1, 1, 2] = 0.2
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    target_field[1, 1, 3] = 0.0
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(1, 1, 1)], [_flat(1, 1, 2)]),
        _pairs([_flat(1, 1, 2)], [_flat(1, 1, 3)]),
        unmatched_prediction=[0],
        unmatched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    observed = losses.BettiMatching0Loss()(prediction, (target,))

    expected = (0.5 * (0.8 - 0.2) ** 2 + 0.5 * (1.0 - 0.0) ** 2) / 2.0
    torch.testing.assert_close(observed, torch.tensor(expected), atol=1e-6, rtol=1e-6)


def test_invalid_target_side_turns_valid_prediction_into_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    prediction[0, 0, 1, 1, 1] = 0.8
    prediction[0, 0, 1, 1, 2] = 0.2
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    target_field[1, 1, 3] = 0.0
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 2
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(1, 1, 1)], [_flat(1, 1, 2)]),
        _pairs([_flat(1, 1, 2)], [_flat(1, 1, 3)]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    observed = losses.BettiMatching0Loss()(prediction, (target,))

    torch.testing.assert_close(observed, torch.tensor(0.18), atol=1e-6, rtol=1e-6)


def test_boundary_prediction_side_turns_valid_target_into_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    prediction[0, 0, 0, 1, 1] = 0.8
    prediction[0, 0, 1, 1, 2] = 0.2
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    target_field[1, 1, 3] = 0.0
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(0, 1, 1)], [_flat(1, 1, 2)]),
        _pairs([_flat(1, 1, 2)], [_flat(1, 1, 3)]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    observed = losses.BettiMatching0Loss()(prediction, (target,))

    torch.testing.assert_close(observed, torch.tensor(0.5), atol=1e-6, rtol=1e-6)


def test_prediction_birth_on_invalid_fragment_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    target_field = torch.full(SHAPE, -1.0)
    prediction[0, 0, 1, 1, 1] = 0.8
    prediction[0, 0, 1, 1, 3] = 0.2
    target_field[1, 1, 2] = 1.0
    target_field[1, 1, 3] = 0.0
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 1] = 2
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(1, 1, 1)], [_flat(1, 1, 3)]),
        _pairs([_flat(1, 1, 2)], [_flat(1, 1, 3)]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    observed = losses.BettiMatching0Loss()(prediction, (target,))

    torch.testing.assert_close(observed, torch.tensor(0.5), atol=1e-6, rtol=1e-6)


def test_both_invalid_matched_sides_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    target_field = torch.full(SHAPE, -1.0)
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 2
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(0, 1, 1)], [-1], [True]),
        _pairs([_flat(1, 1, 2)], [-1], [True]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )
    observed = losses.BettiMatching0Loss()(prediction, (target,))
    torch.testing.assert_close(observed, torch.tensor(0.0))


def test_discrete_assignment_is_detached_but_critical_values_have_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0)
    prediction[0, 0, 1, 1, 1] = 0.8
    prediction[0, 0, 1, 1, 2] = -0.2
    prediction.requires_grad_()
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    target_field[1, 1, 3] = -0.4
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([_flat(1, 1, 1)], [_flat(1, 1, 2)]),
        _pairs([_flat(1, 1, 2)], [_flat(1, 1, 3)]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)

    def fake_matching(
        discrete_prediction: torch.Tensor,
        discrete_target: torch.Tensor,
        target_pairs: PersistencePairs0D,
    ):
        assert not discrete_prediction.requires_grad
        assert not discrete_target.requires_grad
        assert target_pairs is target.topology.pairs
        return result

    monkeypatch.setattr(losses, "induced_matching_0d_precomputed_target", fake_matching)
    value = losses.BettiMatching0Loss()(prediction, (target,))
    value.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 1, 1, 1] != 0
    assert prediction.grad[0, 0, 1, 1, 2] != 0


def test_microbt_pf_loss_has_exact_keys_lambda_one_and_fp32_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros((1, 1, *SHAPE), dtype=torch.bfloat16, requires_grad=True)
    output = {
        "field_logits": logits,
        "region_probabilities": field_logits_to_region_probabilities(logits),
    }
    region_target = torch.zeros((1, 1, *SHAPE), dtype=torch.float32)
    target = _target(torch.full(SHAPE, -1.0))
    center = _flat(1, 1, 2)
    result = _matching(
        _pairs([center], [-1], [True]),
        _pairs([center], [-1], [True]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)

    def fake_matching(
        discrete_prediction: torch.Tensor,
        discrete_target: torch.Tensor,
        target_pairs: PersistencePairs0D,
    ):
        assert discrete_prediction.dtype == torch.float32
        assert discrete_target.dtype == torch.float32
        assert not discrete_prediction.requires_grad
        assert target_pairs is target.topology.pairs
        return result

    monkeypatch.setattr(losses, "induced_matching_0d_precomputed_target", fake_matching)
    observed = losses.MicroBTPFLoss()(output, region_target, (target,))

    assert set(observed) == {"loss", "region", "persistent_field", "field", "betti_0"}
    torch.testing.assert_close(
        observed["persistent_field"], observed["field"] + observed["betti_0"]
    )
    torch.testing.assert_close(
        observed["loss"], observed["region"] + observed["persistent_field"]
    )
    assert all(torch.isfinite(value) for value in observed.values())
    observed["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def _reference_betti_matching_0_loss(
    predicted_field: torch.Tensor,
    persistent_targets: tuple[PersistentFieldTarget, ...],
) -> torch.Tensor:
    sample_losses: list[torch.Tensor] = []
    for batch_index, target in enumerate(persistent_targets):
        prediction = predicted_field[batch_index, 0]
        target_field = target.field.to(device=prediction.device, dtype=prediction.dtype)
        with torch.no_grad():
            matching = induced_matching_0d(prediction.detach(), target_field.detach())
        prediction_birth, prediction_death = critical_values(
            prediction, matching.prediction, floor=-1.0
        )
        target_birth, target_death = critical_values(
            target_field, matching.target, floor=-1.0
        )
        matched, unmatched_prediction, unmatched_target = losses._resolved_branch_terms(
            matching, target
        )
        value = prediction.sum() * 0.0
        for prediction_index, target_index in matched:
            value = value + (
                prediction_birth[prediction_index] - target_birth[target_index]
            ).square()
            value = value + (
                prediction_death[prediction_index] - target_death[target_index]
            ).square()
        for prediction_index in unmatched_prediction:
            value = value + 0.5 * (
                prediction_birth[prediction_index] - prediction_death[prediction_index]
            ).square()
        for target_index in unmatched_target:
            value = value + 0.5 * (
                target_birth[target_index] - target_death[target_index]
            ).square()
        count = len(matched) + len(unmatched_prediction) + len(unmatched_target)
        sample_losses.append(value / max(1, count))
    return torch.stack(sample_losses).mean()


def test_fast_betti_loss_matches_reference_value_and_every_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = np.zeros((5, 5, 5), dtype=np.int32)
    instances[1:3, 1:3, 1:3] = 1
    instances[3, 3, 3] = 2
    target = build_persistent_field_target(
        instances,
        {1: (1.0, 1.0, 1.0), 2: (3.0, 3.0, 3.0)},
        (1.0, 1.0, 1.0),
    )
    base = torch.linspace(-0.95, 0.95, 125).reshape(1, 1, 5, 5, 5)
    reference_prediction = base.clone().requires_grad_()
    fast_prediction = base.clone().requires_grad_()
    calls: list[int] = []

    def tracked_matching(prediction, target_field, target_pairs):
        calls.append(int(target_pairs.birth_flat.size))
        return induced_matching_0d_precomputed_target(
            prediction, target_field, target_pairs
        )

    monkeypatch.setattr(losses, "induced_matching_0d_precomputed_target", tracked_matching)
    reference_value = _reference_betti_matching_0_loss(
        reference_prediction, (target,)
    )
    fast_value = losses.BettiMatching0Loss()(fast_prediction, (target,))
    reference_value.backward()
    fast_value.backward()

    assert calls == [target.topology.pairs.birth_flat.size]
    torch.testing.assert_close(fast_value, reference_value, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        fast_prediction.grad, reference_prediction.grad, atol=1e-7, rtol=1e-7
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_betti_path_is_finite_under_real_cuda_bf16_autocast() -> None:
    instances = np.zeros((5, 5, 5), dtype=np.int32)
    instances[1:3, 1:3, 1:3] = 1
    target = build_persistent_field_target(
        instances, {1: (1.0, 1.0, 1.0)}, (1.0, 1.0, 1.0)
    )
    logits = torch.linspace(
        -1.0, 1.0, 125, device="cuda", dtype=torch.float32
    ).reshape(1, 1, 5, 5, 5).requires_grad_()
    region_target = torch.zeros((1, 1, 5, 5, 5), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = {
            "field_logits": logits,
            "region_probabilities": field_logits_to_region_probabilities(logits),
        }
        result = losses.MicroBTPFLoss()(output, region_target, (target,))

    result["loss"].backward()
    assert all(torch.isfinite(value) for value in result.values())
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_sbpf_unmatched_target_birth_has_live_prediction_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = torch.full((1, 1, *SHAPE), -1.0, requires_grad=True)
    target_field = torch.full(SHAPE, -1.0)
    birth = _flat(1, 1, 2)
    target_field[1, 1, 2] = 1.0
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([], []),
        _pairs([birth], [-1], [True]),
        unmatched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    _, _, birth_completeness = losses.ObjectMorsePersistentFieldLoss().components(
        prediction, (target,)
    )
    birth_completeness.backward()

    torch.testing.assert_close(birth_completeness, torch.tensor(4.0))
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 1, 1, 2] < 0
    assert torch.count_nonzero(prediction.grad) == 1


def test_sbpf_matched_target_birth_is_not_double_supervised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    birth = _flat(1, 1, 2)
    prediction = torch.full((1, 1, *SHAPE), -1.0, requires_grad=True)
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    ids = torch.zeros(SHAPE, dtype=torch.int32)
    ids[1, 1, 2] = 1
    target = _target(target_field, ids, valid_instance_ids=(1,))
    result = _matching(
        _pairs([birth], [-1], [True]),
        _pairs([birth], [-1], [True]),
        matched_prediction=[0],
        matched_target=[0],
    )
    target = _target_with_pairs(target, result.target)
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: result,
    )

    _, _, birth_completeness = losses.ObjectMorsePersistentFieldLoss().components(
        prediction, (target,)
    )

    torch.testing.assert_close(birth_completeness, prediction.sum() * 0.0)


def test_microbt_sbpf_loss_has_two_families_and_explicit_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)
    whole_tumor = torch.sigmoid(raw_logits[:, 0:1])
    regions = torch.cat(
        (
            whole_tumor,
            whole_tumor * torch.sigmoid(raw_logits[:, 2:3]),
            whole_tumor
            * torch.sigmoid(raw_logits[:, 2:3])
            * torch.sigmoid(raw_logits[:, 3:4]),
        ),
        dim=1,
    )
    output = {
        "field_logits": raw_logits,
        "birth_field": torch.tanh(raw_logits[:, 1:2]),
        "region_probabilities": regions,
    }
    target = _target_with_pairs(
        _target(torch.full(SHAPE, -1.0)), _pairs([], [])
    )
    empty = _matching(_pairs([], []), _pairs([], []))
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target, target_pairs: empty,
    )

    observed = losses.MicroBTSBPFLoss()(
        output, torch.zeros((1, 3, *SHAPE)), (target,)
    )

    assert set(observed) == {
        "loss",
        "region",
        "object_morse",
        "field",
        "betti_0",
        "birth_completeness",
    }
    torch.testing.assert_close(
        observed["object_morse"],
        observed["field"] + observed["betti_0"] + observed["birth_completeness"],
    )
    torch.testing.assert_close(observed["loss"], observed["region"] + observed["object_morse"])
    observed["loss"].backward()
    assert raw_logits.grad is not None and torch.isfinite(raw_logits.grad).all()
