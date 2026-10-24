from __future__ import annotations

import numpy as np
import pytest
import torch

import nnunetv2.training.loss.microbt_spp_losses as losses
from nnunetv2.topology.persistent_field_0d import InducedMatching0D, PersistencePairs0D
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    PersistentTargetTopology,
)


SHAPE = (3, 3, 5)


def _flat(x: int) -> int:
    return int(np.ravel_multi_index((1, 1, x), SHAPE))


def _pairs(birth: list[int], death: list[int]) -> PersistencePairs0D:
    return PersistencePairs0D(
        birth_flat=np.asarray(birth, dtype=np.uint32),
        death_flat=np.asarray(death, dtype=np.int64),
        essential=np.asarray([value < 0 for value in death], dtype=bool),
    )


def _target() -> PersistentFieldTarget:
    pairs = _pairs([], [])
    topology = PersistentTargetTopology(
        pairs=pairs,
        birth_values=np.asarray([], dtype=np.float32),
        death_values=np.asarray([], dtype=np.float32),
        branch_valid=np.asarray([], dtype=bool),
    )
    return PersistentFieldTarget(
        field=torch.full(SHAPE, -1.0),
        instance_ids=torch.zeros(SHAPE, dtype=torch.int32),
        dense_valid_mask=torch.ones(SHAPE, dtype=torch.bool),
        valid_instance_ids=(),
        centers_zyx=(),
        topology=topology,
        precompute_topology=False,
    )


def _matching(prediction_pairs: PersistencePairs0D) -> InducedMatching0D:
    target_pairs = _pairs([], [])
    return InducedMatching0D(
        prediction=prediction_pairs,
        target=target_pairs,
        matched_prediction=np.asarray([], dtype=np.int64),
        matched_target=np.asarray([], dtype=np.int64),
        unmatched_prediction=np.arange(prediction_pairs.birth_flat.size, dtype=np.int64),
        unmatched_target=np.asarray([], dtype=np.int64),
    )


@pytest.mark.filterwarnings("error:The given NumPy array is not writable")
def test_false_persistence_cost_grows_with_every_extra_false_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    prediction_one = torch.full((1, 1, *SHAPE), -1.0)
    prediction_one[0, 0, 1, 1, 1] = 0.5
    prediction_two = prediction_one.clone()
    prediction_two[0, 0, 1, 1, 3] = 0.5
    results = iter((_matching(_pairs([_flat(1)], [-1])), _matching(_pairs([_flat(1), _flat(3)], [-1, -1]))))
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target_field, target_pairs: next(results),
    )

    false_one = losses.SupportPreservingOBRPersistentFieldLoss().components(
        prediction_one, prediction_one, (target,)
    )[2]
    false_two = losses.SupportPreservingOBRPersistentFieldLoss().components(
        prediction_two, prediction_two, (target,)
    )[2]

    assert false_one > 0
    torch.testing.assert_close(false_two, 2.0 * false_one)


def test_spp_loss_has_only_region_and_one_object_family_with_explicit_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    empty = _matching(_pairs([], []))
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target_field, target_pairs: empty,
    )
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)
    support = torch.sigmoid(logits[:, 0:1])
    output = {
        "field_logits": logits,
        "birth_field": -1.0 + support.detach() * (1.0 + torch.tanh(logits[:, 1:2])),
        "region_probabilities": torch.cat(
            (
                support,
                support * torch.sigmoid(logits[:, 2:3]),
                support * torch.sigmoid(logits[:, 2:3]) * torch.sigmoid(logits[:, 3:4]),
            ),
            dim=1,
        ),
    }

    observed = losses.MicroBTSPPOBRLoss()(output, torch.zeros((1, 3, *SHAPE)), (target,))

    assert set(observed) == {
        "loss",
        "region",
        "object_morse",
        "field",
        "betti_0",
        "false_persistence",
        "birth_reservation",
        "birth_existence",
        "birth_prominence",
    }
    torch.testing.assert_close(
        observed["object_morse"],
        observed["field"] + observed["betti_0"] + observed["birth_reservation"],
    )
    torch.testing.assert_close(observed["loss"], observed["region"] + observed["object_morse"])


def test_control_spp_loss_retains_original_completeness_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target()
    empty = _matching(_pairs([], []))
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target_field, target_pairs: empty,
    )
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)
    support = torch.sigmoid(logits[:, 0:1])
    output = {
        "field_logits": logits,
        "birth_field": -1.0
        + support.detach() * (1.0 + torch.tanh(logits[:, 1:2])),
        "region_probabilities": torch.cat(
            (
                support,
                support * torch.sigmoid(logits[:, 2:3]),
                support
                * torch.sigmoid(logits[:, 2:3])
                * torch.sigmoid(logits[:, 3:4]),
            ),
            dim=1,
        ),
    }

    observed = losses.MicroBTSPPLoss()(
        output, torch.zeros((1, 3, *SHAPE)), (target,)
    )

    assert "birth_completeness" in observed
    assert "birth_reservation" not in observed


def test_matched_true_branch_can_open_support_but_false_branch_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_birth = _flat(2)
    target_pairs = _pairs([target_birth], [-1])
    topology = PersistentTargetTopology(
        pairs=target_pairs,
        birth_values=np.asarray([1.0], dtype=np.float32),
        death_values=np.asarray([-1.0], dtype=np.float32),
        branch_valid=np.asarray([True], dtype=bool),
    )
    target_field = torch.full(SHAPE, -1.0)
    target_field[1, 1, 2] = 1.0
    instance_ids = torch.zeros(SHAPE, dtype=torch.int32)
    instance_ids[1, 1, 2] = 1
    core_ids = instance_ids.clone()
    target = PersistentFieldTarget(
        field=target_field,
        instance_ids=instance_ids,
        dense_valid_mask=torch.ones(SHAPE, dtype=torch.bool),
        valid_instance_ids=(1,),
        centers_zyx=((1, 1, 2),),
        topology=topology,
        birth_core_ids=core_ids,
        birth_moat_ids=torch.zeros_like(core_ids),
        precompute_topology=False,
    )
    prediction_pairs = _pairs([_flat(2), _flat(3)], [-1, -1])
    matching = InducedMatching0D(
        prediction=prediction_pairs,
        target=target_pairs,
        matched_prediction=np.asarray([0], dtype=np.int64),
        matched_target=np.asarray([0], dtype=np.int64),
        unmatched_prediction=np.asarray([1], dtype=np.int64),
        unmatched_target=np.asarray([], dtype=np.int64),
    )
    monkeypatch.setattr(
        losses,
        "induced_matching_0d_precomputed_target",
        lambda prediction, target_field, target_pairs: matching,
    )
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)
    output = {
        "field_logits": logits,
        "birth_field": -1.0
        + torch.sigmoid(logits[:, 0:1]).detach()
        * (1.0 + torch.tanh(logits[:, 1:2])),
        "region_probabilities": torch.cat(
            (
                torch.sigmoid(logits[:, 0:1]),
                torch.sigmoid(logits[:, 0:1]) * torch.sigmoid(logits[:, 2:3]),
                torch.sigmoid(logits[:, 0:1])
                * torch.sigmoid(logits[:, 2:3])
                * torch.sigmoid(logits[:, 3:4]),
            ),
            dim=1,
        ),
    }

    observed = losses.MicroBTSPPOBRLoss()(
        output, torch.zeros((1, 3, *SHAPE)), (target,)
    )
    observed["object_morse"].backward()

    assert logits.grad is not None
    support_gradient = logits.grad[0, 0]
    assert support_gradient[1, 1, 2].item() != 0.0
    assert support_gradient[1, 1, 3].item() == 0.0
    outside = support_gradient.clone()
    outside[1, 1, 2] = 0.0
    assert torch.count_nonzero(outside).item() == 0
