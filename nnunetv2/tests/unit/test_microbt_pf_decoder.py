from __future__ import annotations

import numpy as np
import pytest

from nnunetv2.topology import persistent_field_0d as topology


pytestmark = pytest.mark.skipif(
    topology._backend is None,
    reason="The compiled H0 backend is tested in the isolated Linux clone",
)


def test_branch_threshold_equality_is_kept() -> None:
    decoded = topology.decode_persistent_field(
        np.asarray([[[0.25]]], dtype=np.float32)
    )
    assert decoded["count"] == 1
    assert decoded["instance_ids"].tolist() == [[[1]]]


def test_death_at_zero_is_not_kept() -> None:
    field = np.asarray([[[1.0, 0.0, 0.8]]], dtype=np.float32)
    decoded = topology.decode_persistent_field(field)
    assert decoded["count"] == 1
    assert decoded["instance_ids"].tolist() == [[[1, 1, 1]]]


def test_two_lesions_merge_below_zero_and_are_both_kept() -> None:
    field = np.asarray([[[1.0, -0.2, 0.8]]], dtype=np.float32)
    decoded = topology.decode_persistent_field(field)

    assert decoded["count"] == 2
    assert decoded["instance_ids"].tolist() == [[[1, 0, 2]]]
    assert [item["centroid_zyx"] for item in decoded["objects"]] == [
        [0, 0, 0],
        [0, 0, 2],
    ]
    assert [item["box_zyx"] for item in decoded["objects"]] == [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 2, 1, 1, 3],
    ]
    assert [mask.sum() for mask in decoded["masks"]] == [1, 1]


def test_diagonal_foreground_uses_26_connectivity() -> None:
    field = np.full((2, 2, 2), -1.0, dtype=np.float32)
    field[0, 0, 0] = 1.0
    field[1, 1, 1] = 0.8
    decoded = topology.decode_persistent_field(field)
    assert decoded["count"] == 1
    assert decoded["masks"][0].sum() == 2


def test_decoder_has_no_topk_or_nms_cap() -> None:
    field = np.full((1, 1, 9), -0.5, dtype=np.float32)
    field[0, 0, ::2] = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6])
    decoded = topology.decode_persistent_field(field)
    assert decoded["count"] == 5
    assert sorted(np.unique(decoded["instance_ids"]).tolist()) == [0, 1, 2, 3, 4, 5]


def test_centroid_is_lexicographically_first_argmax_and_box_is_exclusive() -> None:
    field = np.full((2, 2, 3), -1.0, dtype=np.float32)
    field[0, 0, 0] = 1.0
    field[0, 0, 1] = 1.0
    field[1, 1, 1] = 0.2
    decoded = topology.decode_persistent_field(field)
    assert decoded["objects"][0]["centroid_zyx"] == [0, 0, 0]
    assert decoded["objects"][0]["box_zyx"] == [0, 0, 0, 2, 2, 2]


def test_two_retained_births_in_one_zero_component_raise_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = topology.PersistencePairs0D(
        birth_flat=np.asarray([0, 2], dtype=np.uint32),
        death_flat=np.asarray([-1, 3], dtype=np.int64),
        essential=np.asarray([True, False], dtype=bool),
    )
    monkeypatch.setattr(topology, "persistence_pairs_0d", lambda field, floor: pairs)
    field = np.asarray([[[1.0, 0.2, 0.8, -0.5]]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="retained branches share one zero-level component"):
        topology.decode_persistent_field(field)


def test_sbpf_connected_support_is_partitioned_by_two_birth_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = topology.PersistencePairs0D(
        birth_flat=np.asarray([0, 4], dtype=np.uint32),
        death_flat=np.asarray([-1, 2], dtype=np.int64),
        essential=np.asarray([True, False], dtype=bool),
    )
    monkeypatch.setattr(topology, "persistence_pairs_0d", lambda field, floor: pairs)
    birth = np.asarray([[[1.0, 0.6, 0.2, 0.6, 0.8]]], dtype=np.float32)
    support = np.full_like(birth, 0.9)

    decoded = topology.decode_support_birth_field(birth, support)

    assert decoded["count"] == 2
    assert np.all(decoded["instance_ids"] > 0)
    assert decoded["instance_ids"].ravel()[0] == 1
    assert decoded["instance_ids"].ravel()[4] == 2
    assert sum(mask.sum() for mask in decoded["masks"]) == support.size
    assert [item["centroid_zyx"] for item in decoded["objects"]] == [
        [0, 0, 0],
        [0, 0, 4],
    ]


def test_sbpf_marker_outside_semantic_support_is_retained_as_one_voxel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = topology.PersistencePairs0D(
        birth_flat=np.asarray([1], dtype=np.uint32),
        death_flat=np.asarray([-1], dtype=np.int64),
        essential=np.asarray([True], dtype=bool),
    )
    monkeypatch.setattr(topology, "persistence_pairs_0d", lambda field, floor: pairs)
    birth = np.asarray([[[-1.0, 0.8, -1.0]]], dtype=np.float32)
    support = np.zeros_like(birth)

    decoded = topology.decode_support_birth_field(birth, support)

    assert decoded["count"] == 1
    assert decoded["instance_ids"].tolist() == [[[0, 1, 0]]]
    assert decoded["marker_outside_support"] == 1


def test_sbpf_decoder_is_empty_without_retained_markers_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = topology.PersistencePairs0D(
        birth_flat=np.asarray([], dtype=np.uint32),
        death_flat=np.asarray([], dtype=np.int64),
        essential=np.asarray([], dtype=bool),
    )
    monkeypatch.setattr(topology, "persistence_pairs_0d", lambda field, floor: pairs)
    birth = np.zeros((2, 2, 2), dtype=np.float32)
    support = np.ones_like(birth)

    first = topology.decode_support_birth_field(birth, support)
    second = topology.decode_support_birth_field(birth, support)

    assert first["count"] == 0
    assert np.count_nonzero(first["instance_ids"]) == 0
    np.testing.assert_array_equal(first["instance_ids"], second["instance_ids"])
