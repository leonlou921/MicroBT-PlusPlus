from __future__ import annotations

import numpy as np
import pytest
import torch

from nnunetv2.nets.microbt_umctf.model import umctf_logits_to_outputs
from nnunetv2.inference.microbt_umctf_decode import decode_umctf_instances
from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_umctf_losses import UnitMassConservationLoss


def _target(ids: torch.Tensor, centers: tuple[tuple[int, int, int], ...]):
    valid = tuple(int(v) for v in torch.unique(ids).tolist() if int(v) > 0)
    return PersistentFieldTarget(
        field=torch.zeros_like(ids, dtype=torch.float32),
        instance_ids=ids.to(torch.int32),
        dense_valid_mask=torch.ones_like(ids, dtype=torch.bool),
        valid_instance_ids=valid,
        centers_zyx=centers,
        precompute_topology=False,
    )


def test_seven_logits_preserve_regions_and_gate_density_by_wt() -> None:
    logits = torch.zeros((1, 7, 3, 3, 3), requires_grad=True)
    output = umctf_logits_to_outputs(logits)
    assert output["transport"].shape == (1, 3, 3, 3, 3)
    assert output["region_probabilities"].shape == (1, 3, 3, 3, 3)
    assert output["object_density"].shape == (1, 1, 3, 3, 3)
    torch.testing.assert_close(output["object_density"], torch.full_like(output["object_density"], 0.25))
    wt, tc, et = output["region_probabilities"].unbind(1)
    assert torch.all(et <= tc)
    assert torch.all(tc <= wt)


def test_unit_mass_is_size_invariant_at_one_integrated_mass() -> None:
    ids = torch.zeros((5, 5, 5), dtype=torch.int32)
    ids[0, 0, 0] = 1
    ids[2:4, 2:4, 2:4] = 2
    density = torch.zeros((1, 1, 5, 5, 5), requires_grad=True)
    density.data[0, 0, 0, 0, 0] = 1.0
    density.data[0, 0, 2:4, 2:4, 2:4] = 1.0 / 8.0
    loss, object_mass, background_mass = UnitMassConservationLoss()(
        density, [_target(ids, ((0, 0, 0), (2, 2, 2)))]
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(object_mass, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(background_mass, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_missing_micro_mass_has_restoring_gradient_and_background_is_penalized() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    ids[1, 1, 1] = 1
    density = torch.zeros((1, 1, 3, 3, 3), requires_grad=True)
    density.data[0, 0, 0, 0, 0] = 0.2
    total, object_mass, background_mass = UnitMassConservationLoss(background_weight=1.0)(
        density, [_target(ids, ((1, 1, 1),))]
    )
    assert object_mass.item() > 0
    assert background_mass.item() > 0
    total.backward()
    assert density.grad[0, 0, 1, 1, 1].item() < 0
    assert density.grad[0, 0, 0, 0, 0].item() > 0


def test_hard_survival_gradient_is_not_diluted_by_easy_objects() -> None:
    ids = torch.arange(1, 7, dtype=torch.int32).reshape(1, 1, 6)
    density = torch.tensor(
        [[[[[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]]], requires_grad=True
    )
    target = _target(
        ids,
        ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5)),
    )

    total, _, _ = UnitMassConservationLoss(
        hard_survival_mass=0.25
    )(density, [target])
    total.backward()

    # The ordinary six-object mean contributes -1/6. The hard-object term
    # contributes another -1 without being divided by the five easy objects.
    torch.testing.assert_close(
        density.grad[0, 0, 0, 0, 0], torch.tensor(-7.0 / 6.0), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        density.grad[0, 0, 0, 0, 1:], torch.zeros(5), atol=1e-6, rtol=0
    )


def test_empty_patch_penalizes_density_and_remains_differentiable() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    density = torch.full((1, 1, 3, 3, 3), 0.1, requires_grad=True)
    total, object_mass, background_mass = UnitMassConservationLoss()(
        density, [_target(ids, ())]
    )
    torch.testing.assert_close(object_mass, torch.tensor(0.0))
    assert background_mass.item() > 0
    total.backward()
    assert torch.all(density.grad > 0)


def test_background_term_constrains_total_mass_not_only_mean_density() -> None:
    small_ids = torch.zeros((1, 1, 1), dtype=torch.int32)
    large_ids = torch.zeros((10, 10, 10), dtype=torch.int32)
    _, _, small_background = UnitMassConservationLoss()(
        torch.full((1, 1, 1, 1, 1), 0.01), [_target(small_ids, ())]
    )
    _, _, large_background = UnitMassConservationLoss()(
        torch.full((1, 1, 10, 10, 10), 0.01), [_target(large_ids, ())]
    )
    assert large_background > small_background * 100


def test_trilinear_transport_conserves_mass_and_merges_one_lesion_atom() -> None:
    density = np.zeros((5, 5, 5), dtype=np.float32)
    density[1, 1, 1] = 0.5
    density[1, 1, 2] = 0.5
    support = density > 0
    transport = np.zeros((3, 5, 5, 5), dtype=np.float32)
    transport[:, 1, 1, 1] = np.array([1.5, 1.5, 1.5]) - np.array([1, 1, 1])
    transport[:, 1, 1, 2] = np.array([1.5, 1.5, 1.5]) - np.array([1, 1, 2])
    decoded = decode_umctf_instances(
        support.astype(np.float32), density, transport,
        offset_scale_zyx=(1, 1, 1), density_threshold=0.0,
        min_atom_mass=0.2, nms_radius=1, max_centers=8,
    )
    assert decoded.count == 1
    assert abs(decoded.total_input_mass - 1.0) < 1e-6
    assert abs(decoded.total_atom_mass - 1.0) < 1e-6
    assert np.count_nonzero(decoded.instance_labels) == 2


def test_two_separated_unit_masses_decode_as_two_candidates() -> None:
    density = np.zeros((7, 7, 7), dtype=np.float32)
    density[1, 1, 1] = 1.0
    density[5, 5, 5] = 1.0
    decoded = decode_umctf_instances(
        (density > 0).astype(np.float32), density,
        np.zeros((3, 7, 7, 7), dtype=np.float32),
        offset_scale_zyx=(1, 1, 1), density_threshold=0,
        min_atom_mass=0.5, nms_radius=1, max_centers=8,
    )
    assert decoded.count == 2
    assert set(decoded.centers_zyx) == {(1, 1, 1), (5, 5, 5)}


def test_decoder_rejects_negative_density() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        decode_umctf_instances(
            np.ones((2, 2, 2), dtype=np.float32),
            -np.ones((2, 2, 2), dtype=np.float32),
            np.zeros((3, 2, 2, 2), dtype=np.float32),
        )
