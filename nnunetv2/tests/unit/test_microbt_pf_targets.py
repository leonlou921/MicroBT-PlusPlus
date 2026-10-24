from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch
from scipy import ndimage

from nnunetv2.training.dataloading import microbt_pf_targets
from nnunetv2.training.dataloading.microbt_pf_targets import (
    build_persistent_field_target,
    deterministic_interior_center,
    label_instances_26,
    spacing_aware_geodesic,
    transform_centers_affine,
)
from nnunetv2.topology.persistent_field_0d import persistence_pairs_0d


def test_instance_labeling_uses_26_connectivity() -> None:
    diagonal = np.zeros((4, 4, 4), dtype=bool)
    diagonal[1, 1, 1] = True
    diagonal[2, 2, 2] = True
    instances = label_instances_26(diagonal)
    assert instances.dtype == np.int32
    assert instances.max() == 1

    separated = np.zeros((5, 3, 3), dtype=bool)
    separated[1, 1, 1] = True
    separated[3, 1, 1] = True
    assert label_instances_26(separated).max() == 2


def test_center_uses_anisotropic_physical_distance_then_lexicographic_tie() -> None:
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[tuple(np.asarray([[2, 2, 1], [3, 1, 1], [4, 2, 1], [4, 3, 1]]).T)] = True
    assert deterministic_interior_center(mask, (3.0, 1.0, 1.0)) == (3, 1, 1)

    tied = np.zeros((5, 5, 5), dtype=bool)
    tied[2, 2, 2:4] = True
    assert deterministic_interior_center(tied, (1.0, 1.0, 1.0)) == (2, 2, 2)


def test_geodesic_is_spacing_aware_for_a_diagonal_edge() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True

    distance = spacing_aware_geodesic(mask, (1, 1, 1), (2.0, 1.0, 1.0))

    assert distance[1, 1, 1] == 0.0
    assert distance[2, 2, 2] == pytest.approx(math.sqrt(6.0))
    assert np.isinf(distance[~mask]).all()


def test_geodesic_solver_receives_tight_component_bbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_solver = microbt_pf_targets.MCP_Geometric
    observed_shapes: list[tuple[int, ...]] = []

    class RecordingSolver:
        def __init__(self, costs, *args, **kwargs) -> None:
            observed_shapes.append(tuple(costs.shape))
            self.delegate = real_solver(costs, *args, **kwargs)

        def find_costs(self, starts):
            return self.delegate.find_costs(starts)

    monkeypatch.setattr(microbt_pf_targets, "MCP_Geometric", RecordingSolver)
    mask = np.zeros((20, 21, 22), dtype=bool)
    mask[8:10, 9:12, 10:14] = True

    distance = spacing_aware_geodesic(mask, (8, 9, 10), (1.0, 1.0, 1.0))

    assert observed_shapes == [(2, 3, 4)]
    assert distance[8, 9, 10] == 0.0
    assert np.isfinite(distance[mask]).all()
    assert np.isinf(distance[~mask]).all()


def test_canonical_field_has_one_peak_per_lesion_and_connected_superlevels() -> None:
    whole_tumor = np.zeros((7, 7, 7), dtype=bool)
    whole_tumor[1:3, 1:3, 1:3] = True
    whole_tumor[4, 4, 4] = True
    instance_ids = label_instances_26(whole_tumor)
    centers = {
        instance_id: deterministic_interior_center(
            instance_ids == instance_id, (2.0, 1.0, 1.0)
        )
        for instance_id in range(1, int(instance_ids.max()) + 1)
    }

    target = build_persistent_field_target(
        instance_ids, centers, (2.0, 1.0, 1.0), topology_margin=1
    )

    field = target.field.numpy()
    assert torch.equal(target.instance_ids, torch.from_numpy(instance_ids))
    assert np.all(field[instance_ids == 0] == -1.0)
    assert np.all(field[instance_ids > 0] >= 0.25)
    assert target.valid_instance_ids == (1, 2)
    assert target.centers_zyx == (centers[1], centers[2])
    for instance_id in (1, 2):
        lesion_values = field[instance_ids == instance_id]
        assert np.count_nonzero(lesion_values == 1.0) == 1
        for level in np.unique(lesion_values[lesion_values > 0.25]):
            superlevel = (field >= level) & (instance_ids == instance_id)
            assert ndimage.label(superlevel, np.ones((3, 3, 3), dtype=np.uint8))[1] == 1


def test_one_voxel_lesion_is_one() -> None:
    instance_ids = np.zeros((3, 3, 3), dtype=np.int32)
    instance_ids[1, 1, 1] = 1
    target = build_persistent_field_target(
        instance_ids, {1: (1, 1, 1)}, (1.0, 1.0, 1.0)
    )
    assert target.field[1, 1, 1].item() == 1.0


def test_no_topology_target_preserves_literal_dense_supervision() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[2, 2, 2:4] = 9

    target = build_persistent_field_target(
        instance_ids,
        {9: (2.0, 2.0, 2.0)},
        (1.0, 1.0, 1.0),
        precompute_topology=False,
    )

    assert target.topology is None
    assert target.valid_instance_ids == (9,)
    assert target.centers_zyx == ((2, 2, 2),)
    assert target.dense_valid_mask.all()
    assert target.field[2, 2, 2].item() == 1.0
    assert target.field[2, 2, 3].item() == 0.25
    assert torch.all(target.field[torch.from_numpy(instance_ids == 0)] == -1.0)


def test_precompute_topology_returns_enriched_view_without_dense_copies() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[2, 2, 2:4] = 9
    target = build_persistent_field_target(
        instance_ids,
        {9: (2.0, 2.0, 2.0)},
        (1.0, 1.0, 1.0),
        precompute_topology=False,
    )

    enriched = microbt_pf_targets.precompute_persistent_target_topology(target)

    assert target.topology is None
    assert enriched.topology is not None
    assert enriched.field.data_ptr() == target.field.data_ptr()
    assert enriched.instance_ids.data_ptr() == target.instance_ids.data_ptr()
    assert enriched.dense_valid_mask.data_ptr() == target.dense_valid_mask.data_ptr()


def test_disconnected_same_id_fragment_is_excluded_from_pf_supervision() -> None:
    instance_ids = np.zeros((7, 7, 7), dtype=np.int32)
    instance_ids[2, 2, 1:3] = 7
    instance_ids[4, 4, 5] = 7

    target = build_persistent_field_target(
        instance_ids, {7: (2.0, 2.0, 1.0)}, (1.0, 1.0, 1.0)
    )

    assert torch.isfinite(target.field).all()
    assert target.valid_instance_ids == (7,)
    assert target.dense_valid_mask[2, 2, 1:3].all()
    assert not target.dense_valid_mask[4, 4, 5]
    assert target.field[4, 4, 5].item() == -1.0


@pytest.mark.parametrize(
    ("affine", "mirror_axes", "expected"),
    [
        (np.eye(3), (), (1.0, 2.0, 2.0)),
        (
            np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            (),
            (2.0, 1.0, 2.0),
        ),
        (2.0 * np.eye(3), (), (1.5, 2.0, 2.0)),
        (np.eye(3), (0,), (3.0, 2.0, 2.0)),
    ],
)
def test_center_affine_transform_contract(
    affine: np.ndarray, mirror_axes: tuple[int, ...], expected: tuple[float, ...]
) -> None:
    transformed = transform_centers_affine(
        np.asarray([[1.0, 2.0, 2.0]]),
        input_shape=(5, 5, 5),
        output_shape=(5, 5, 5),
        affine=affine,
        center_location_in_pixels=(2.5, 2.5, 2.5),
        mirror_axes=mirror_axes,
    )
    np.testing.assert_allclose(transformed[0], expected)


def test_center_is_snapped_only_to_same_id_and_margin_controls_validity() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[2, 2, 2:4] = 1
    target = build_persistent_field_target(
        instance_ids, {1: (2.0, 1.8, 2.9)}, (1.0, 1.0, 1.0), topology_margin=1
    )
    assert target.centers_zyx == ((2, 2, 3),)
    assert target.valid_instance_ids == (1,)

    edge_ids = np.zeros((5, 5, 5), dtype=np.int32)
    edge_ids[0, 2, 2] = 1
    invalid = build_persistent_field_target(
        edge_ids, {1: (0.0, 2.0, 2.0)}, (1.0, 1.0, 1.0), topology_margin=1
    )
    assert invalid.valid_instance_ids == ()
    assert not invalid.dense_valid_mask[0, 2, 2]
    assert invalid.dense_valid_mask[torch.from_numpy(edge_ids == 0)].all()


def test_out_of_patch_center_excludes_visible_fragment() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[2, 2, 2:4] = 7
    target = build_persistent_field_target(
        instance_ids, {7: (-1.0, 2.0, 2.0)}, (1.0, 1.0, 1.0), topology_margin=1
    )
    assert target.valid_instance_ids == ()
    assert not target.dense_valid_mask[torch.from_numpy(instance_ids == 7)].any()


def test_target_carries_exact_immutable_topology() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[1:3, 1:3, 1:3] = 1
    instance_ids[3, 3, 3] = 2
    target = build_persistent_field_target(
        instance_ids,
        {1: (1, 1, 1), 2: (3, 3, 3)},
        (1.0, 1.0, 1.0),
    )

    expected = persistence_pairs_0d(target.field)
    np.testing.assert_array_equal(
        target.topology.pairs.birth_flat, expected.birth_flat
    )
    np.testing.assert_array_equal(
        target.topology.pairs.death_flat, expected.death_flat
    )
    np.testing.assert_array_equal(target.topology.pairs.essential, expected.essential)
    flat = target.field.numpy().ravel()
    expected_birth = flat[expected.birth_flat.astype(np.int64)]
    expected_death = np.full(expected.death_flat.size, -1.0, dtype=np.float32)
    finite = expected.death_flat >= 0
    expected_death[finite] = flat[expected.death_flat[finite]]
    np.testing.assert_array_equal(target.topology.birth_values, expected_birth)
    np.testing.assert_array_equal(target.topology.death_values, expected_death)
    np.testing.assert_array_equal(target.topology.branch_valid, [True, True])
    for array in (
        target.topology.pairs.birth_flat,
        target.topology.pairs.death_flat,
        target.topology.pairs.essential,
        target.topology.birth_values,
        target.topology.death_values,
        target.topology.branch_valid,
    ):
        assert not array.flags.writeable


def test_target_topology_survives_spawn_pickle() -> None:
    instance_ids = np.zeros((3, 3, 3), dtype=np.int32)
    instance_ids[1, 1, 1] = 1
    target = build_persistent_field_target(
        instance_ids, {1: (1, 1, 1)}, (1.0, 1.0, 1.0)
    )

    restored = pickle.loads(pickle.dumps(target))

    np.testing.assert_array_equal(
        restored.topology.pairs.death_flat, target.topology.pairs.death_flat
    )
    np.testing.assert_array_equal(
        restored.topology.branch_valid, target.topology.branch_valid
    )
    assert not restored.topology.pairs.death_flat.flags.writeable
    assert not restored.topology.branch_valid.flags.writeable
