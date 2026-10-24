from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest
import torch  # noqa: F401  # initializes shared libraries required by CppExtension

from nnunetv2 import _persistent_field_0d as backend


_OFFSETS_26 = tuple(
    offset
    for offset in itertools.product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0)
)


def _find(parent: np.ndarray, vertex: int) -> int:
    root = vertex
    while parent[root] != root:
        root = int(parent[root])
    while parent[vertex] != vertex:
        next_vertex = int(parent[vertex])
        parent[vertex] = root
        vertex = next_vertex
    return root


def _link_elder(
    parent: np.ndarray, birth_values: np.ndarray, root_a: int, root_b: int
) -> int:
    if root_a == root_b:
        return root_a
    a_is_elder = (birth_values[root_a] > birth_values[root_b]) or (
        birth_values[root_a] == birth_values[root_b] and root_a < root_b
    )
    survivor, killed = (root_a, root_b) if a_is_elder else (root_b, root_a)
    parent[killed] = survivor
    return killed


def _neighbors(flat_index: int, shape: tuple[int, int, int]):
    coordinate = np.unravel_index(flat_index, shape)
    for offset in _OFFSETS_26:
        neighbor = tuple(coordinate[axis] + offset[axis] for axis in range(3))
        if all(0 <= neighbor[axis] < shape[axis] for axis in range(3)):
            yield int(np.ravel_multi_index(neighbor, shape))


def _oracle_pairs(field: np.ndarray):
    shape = field.shape
    values = field.ravel()
    order = np.lexsort((np.arange(values.size, dtype=np.int64), -values))
    parent = np.arange(values.size, dtype=np.int64)
    active = np.zeros(values.size, dtype=bool)
    birth_flat: list[int] = []
    death_flat: list[int] = []
    pair_by_birth: dict[int, int] = {}
    for vertex in order:
        vertex = int(vertex)
        active[vertex] = True
        for neighbor in _neighbors(vertex, shape):
            if not active[neighbor]:
                continue
            root_vertex = _find(parent, vertex)
            root_neighbor = _find(parent, neighbor)
            if root_vertex == root_neighbor:
                continue
            killed = _link_elder(parent, values, root_vertex, root_neighbor)
            if values[killed] > values[vertex]:
                pair_by_birth[killed] = len(birth_flat)
                birth_flat.append(killed)
                death_flat.append(vertex)
    essential_birth = _find(parent, 0)
    pair_by_birth[essential_birth] = len(birth_flat)
    birth_flat.append(essential_birth)
    death_flat.append(-1)
    return (
        {
            "birth_flat": np.asarray(birth_flat, dtype=np.uint32),
            "death_flat": np.asarray(death_flat, dtype=np.int64),
            "essential": np.asarray(
                [False] * (len(birth_flat) - 1) + [True], dtype=bool
            ),
        },
        pair_by_birth,
    )


def _oracle_induced_matching(prediction: np.ndarray, target: np.ndarray):
    prediction_pairs, prediction_map = _oracle_pairs(prediction)
    target_pairs, target_map = _oracle_pairs(target)
    comparison = np.maximum(prediction, target)
    values_comparison = comparison.ravel()
    values_prediction = prediction.ravel()
    values_target = target.ravel()
    order = np.lexsort(
        (np.arange(values_comparison.size, dtype=np.int64), -values_comparison)
    )
    parent_comparison = np.arange(values_comparison.size, dtype=np.int64)
    parent_prediction = np.arange(values_comparison.size, dtype=np.int64)
    parent_target = np.arange(values_comparison.size, dtype=np.int64)
    active = np.zeros(values_comparison.size, dtype=bool)
    matched_prediction: list[int] = []
    matched_target: list[int] = []
    for vertex in order:
        vertex = int(vertex)
        active[vertex] = True
        for neighbor in _neighbors(vertex, comparison.shape):
            if not active[neighbor]:
                continue
            comparison_a = _find(parent_comparison, vertex)
            comparison_b = _find(parent_comparison, neighbor)
            if comparison_a == comparison_b:
                continue
            killed_comparison = _link_elder(
                parent_comparison, values_comparison, comparison_a, comparison_b
            )
            prediction_owner = _link_elder(
                parent_prediction,
                values_prediction,
                _find(parent_prediction, vertex),
                _find(parent_prediction, neighbor),
            )
            target_owner = _link_elder(
                parent_target,
                values_target,
                _find(parent_target, vertex),
                _find(parent_target, neighbor),
            )
            if values_comparison[killed_comparison] > values_comparison[vertex]:
                if prediction_owner in prediction_map and target_owner in target_map:
                    matched_prediction.append(prediction_map[prediction_owner])
                    matched_target.append(target_map[target_owner])

    prediction_essential = int(np.flatnonzero(prediction_pairs["essential"])[0])
    target_essential = int(np.flatnonzero(target_pairs["essential"])[0])
    matched_prediction.append(prediction_essential)
    matched_target.append(target_essential)
    matched_prediction_array = np.asarray(matched_prediction, dtype=np.int64)
    matched_target_array = np.asarray(matched_target, dtype=np.int64)
    unmatched_prediction = np.asarray(
        sorted(set(range(len(prediction_pairs["birth_flat"]))) - set(matched_prediction)),
        dtype=np.int64,
    )
    unmatched_target = np.asarray(
        sorted(set(range(len(target_pairs["birth_flat"]))) - set(matched_target)),
        dtype=np.int64,
    )
    return {
        "prediction_birth_flat": prediction_pairs["birth_flat"],
        "prediction_death_flat": prediction_pairs["death_flat"],
        "prediction_essential": prediction_pairs["essential"],
        "target_birth_flat": target_pairs["birth_flat"],
        "target_death_flat": target_pairs["death_flat"],
        "target_essential": target_pairs["essential"],
        "matched_prediction": matched_prediction_array,
        "matched_target": matched_target_array,
        "unmatched_prediction": unmatched_prediction,
        "unmatched_target": unmatched_target,
    }


def _assert_array_dict_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_one_peak_has_one_essential_pair_with_fixed_floor_marker() -> None:
    field = np.asarray([[[0.7]]], dtype=np.float32)
    result = backend.persistence_pairs(field, -1.0)
    _assert_array_dict_equal(result, _oracle_pairs(field)[0])
    assert result["death_flat"].tolist() == [-1]
    assert result["essential"].tolist() == [True]


def test_two_finite_branches_and_lexicographic_elder_tie() -> None:
    field = np.asarray([[[1.0, 0.0, 0.8, 0.0, 0.7]]], dtype=np.float32)
    result = backend.persistence_pairs(field, -1.0)
    assert np.count_nonzero(~result["essential"]) == 2
    _assert_array_dict_equal(result, _oracle_pairs(field)[0])

    tied = np.asarray([[[1.0, 0.0, 1.0]]], dtype=np.float32)
    tied_result = backend.persistence_pairs(tied, -1.0)
    assert tied_result["birth_flat"].tolist() == [2, 0]
    assert tied_result["death_flat"].tolist() == [1, -1]


def test_diagonal_contact_is_connected_under_26_neighborhood() -> None:
    field = np.zeros((2, 2, 2), dtype=np.float32)
    field[0, 0, 0] = 1.0
    field[1, 1, 1] = 0.8
    result = backend.persistence_pairs(field, -1.0)
    assert result["birth_flat"].tolist() == [0]
    assert result["essential"].tolist() == [True]


@pytest.mark.parametrize(
    "field",
    [
        np.empty((0, 1, 1), dtype=np.float32),
        np.asarray([[[np.nan]]], dtype=np.float32),
        np.asarray([[[np.inf]]], dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.float64),
        np.ones((2, 2, 4), dtype=np.float32)[:, :, ::2],
        np.ones((2, 2), dtype=np.float32),
    ],
)
def test_invalid_inputs_are_rejected(field: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        backend.persistence_pairs(field, -1.0)


def test_backend_matches_bruteforce_oracle_on_100_seeded_volumes() -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 5, size=3))
        field = np.ascontiguousarray(rng.normal(size=shape).astype(np.float32))
        _assert_array_dict_equal(
            backend.persistence_pairs(field, -1.0), _oracle_pairs(field)[0]
        )


def test_induced_matching_tracks_displaced_spatial_correspondence() -> None:
    prediction = np.full((1, 3, 5), -1.0, dtype=np.float32)
    target = prediction.copy()
    prediction[0, 1, 1] = target[0, 1, 1] = 1.0
    prediction[0, 1, 3] = 0.8
    target[0, 0, 3] = 0.8
    prediction[0, 1, 2] = target[0, 0, 2] = -0.2

    actual = backend.induced_matching(prediction, target, -1.0)
    expected = _oracle_induced_matching(prediction, target)

    _assert_array_dict_equal(actual, expected)
    assert len(actual["matched_prediction"]) == 2


def test_induced_matching_reports_each_unmatched_direction() -> None:
    flat = np.asarray([[[-1.0, -1.0, 1.0, -1.0, -1.0]]], dtype=np.float32)
    extra = np.asarray([[[0.8, -0.2, 1.0, -1.0, -1.0]]], dtype=np.float32)

    prediction_extra = backend.induced_matching(extra, flat, -1.0)
    target_extra = backend.induced_matching(flat, extra, -1.0)

    assert prediction_extra["unmatched_prediction"].size == 1
    assert prediction_extra["unmatched_target"].size == 0
    assert target_extra["unmatched_prediction"].size == 0
    assert target_extra["unmatched_target"].size == 1


def test_induced_matching_validates_equal_shapes() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        backend.induced_matching(
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 4), dtype=np.float32),
            -1.0,
        )


def test_all_backend_arrays_are_deterministic_across_20_calls() -> None:
    rng = np.random.default_rng(19)
    prediction = np.ascontiguousarray(rng.normal(size=(4, 4, 4)).astype(np.float32))
    target = np.ascontiguousarray(rng.normal(size=(4, 4, 4)).astype(np.float32))
    hashes = []
    for _ in range(20):
        result = backend.induced_matching(prediction, target, -1.0)
        digest = hashlib.sha256()
        for key in sorted(result):
            digest.update(key.encode("utf-8"))
            digest.update(result[key].dtype.str.encode("ascii"))
            digest.update(result[key].tobytes())
        hashes.append(digest.hexdigest())
    assert len(set(hashes)) == 1


def test_profiled_induced_matching_preserves_arrays_and_reports_stages() -> None:
    prediction = np.asarray(
        [[[1.0, -0.25, 0.75], [-0.5, 0.25, -1.0]]], dtype=np.float32
    )
    target = np.asarray(
        [[[0.75, -0.5, 1.0], [-0.25, 0.5, -1.0]]], dtype=np.float32
    )
    reference = backend.induced_matching(prediction, target, -1.0)

    profiled = backend.induced_matching_profiled(prediction, target, -1.0)

    assert set(profiled) == {*reference, "profile_seconds"}
    for key, expected in reference.items():
        np.testing.assert_array_equal(profiled[key], expected, err_msg=key)
    profile = profiled["profile_seconds"]
    assert set(profile) == {
        "prediction_pairs",
        "target_pairs",
        "comparison_matching",
        "total",
    }
    assert all(isinstance(profile[key], float) and profile[key] >= 0.0 for key in profile)
    assert profile["total"] >= max(
        profile["prediction_pairs"],
        profile["target_pairs"],
        profile["comparison_matching"],
    )


def _precomputed_backend_matching(
    prediction: np.ndarray,
    target: np.ndarray,
    pairs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return backend.induced_matching_precomputed_target(
        prediction,
        target,
        pairs["birth_flat"],
        pairs["death_flat"],
        pairs["essential"],
        -1.0,
    )


def test_precomputed_target_matches_normal_backend_on_100_tie_heavy_volumes() -> None:
    rng = np.random.default_rng(42)
    levels = np.asarray([-1.0, -0.0, 0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    for case_index in range(100):
        shape = tuple(int(value) for value in rng.integers(1, 6, size=3))
        prediction = np.ascontiguousarray(rng.choice(levels, size=shape))
        target = np.ascontiguousarray(rng.choice(levels, size=shape))
        if target.size >= 3 and case_index % 4 == 0:
            target.ravel()[0] = np.float32(1.0)
            target.ravel()[-1] = np.float32(0.75)
            prediction.ravel()[0] = np.float32(0.75)
            prediction.ravel()[-1] = np.float32(1.0)
        target_pairs = backend.persistence_pairs(target, -1.0)

        reference = backend.induced_matching(prediction, target, -1.0)
        precomputed = _precomputed_backend_matching(prediction, target, target_pairs)

        _assert_array_dict_equal(precomputed, reference)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pairs: pairs.__setitem__(
            "birth_flat", np.asarray([99], dtype=np.uint32)
        ),
        lambda pairs: pairs.__setitem__(
            "essential", np.zeros_like(pairs["essential"])
        ),
        lambda pairs: pairs.__setitem__(
            "birth_flat",
            np.asarray([pairs["birth_flat"][0], pairs["birth_flat"][0]], dtype=np.uint32),
        ),
        lambda pairs: pairs.__setitem__(
            "birth_flat", pairs["birth_flat"].astype(np.int64)
        ),
    ],
)
def test_precomputed_target_rejects_malformed_pair_arrays(mutate) -> None:
    prediction = np.asarray([[[1.0, -0.5, 0.75]]], dtype=np.float32)
    target = np.asarray([[[0.75, -0.5, 1.0]]], dtype=np.float32)
    pairs = dict(backend.persistence_pairs(target, -1.0))
    mutate(pairs)

    with pytest.raises((TypeError, ValueError)):
        _precomputed_backend_matching(prediction, target, pairs)


def test_precomputed_target_rejects_pairs_indexed_for_another_shape() -> None:
    larger = np.asarray([[[1.0, -0.5, 0.75, -0.25, 0.9]]], dtype=np.float32)
    pairs = backend.persistence_pairs(larger, -1.0)
    smaller = np.asarray([[[1.0, -0.5, 0.75]]], dtype=np.float32)

    with pytest.raises(ValueError, match="target pair index"):
        _precomputed_backend_matching(smaller, smaller, pairs)


def test_radix_order_matches_reference_for_signed_zero_subnormals_and_ties() -> None:
    smallest = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    values = np.asarray(
        [
            -0.0,
            0.0,
            1.0,
            -1.0,
            smallest,
            -smallest,
            1.0,
            -0.0,
            0.25,
            -0.25,
        ],
        dtype=np.float32,
    )
    field = values.reshape(1, 1, -1)
    expected = np.lexsort((np.arange(values.size, dtype=np.int64), -values)).astype(
        np.uint32
    )

    result = backend.descending_order_profiled(field)

    np.testing.assert_array_equal(result["radix_order"], expected)
    np.testing.assert_array_equal(result["reference_order"], expected)
    assert type(result["radix_seconds"]) is float and result["radix_seconds"] >= 0
    assert type(result["reference_seconds"]) is float and result["reference_seconds"] >= 0


def test_radix_order_matches_reference_on_100_random_finite_arrays() -> None:
    rng = np.random.default_rng(54321)
    for _ in range(100):
        length = int(rng.integers(1, 500))
        values = rng.normal(size=length).astype(np.float32)
        tied = rng.random(length) < 0.25
        values[tied] = rng.choice(
            np.asarray([-1.0, -0.0, 0.0, 0.5, 1.0], dtype=np.float32),
            size=np.count_nonzero(tied),
        )
        if length >= 5:
            values[:5] = np.asarray([-0.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)
        field = np.ascontiguousarray(values.reshape(1, 1, -1))
        result = backend.descending_order_profiled(field)
        np.testing.assert_array_equal(result["radix_order"], result["reference_order"])


@pytest.mark.parametrize(
    ("shape", "vertex"),
    [
        ((3, 3, 3), 13),
        ((4, 5, 6), int(np.ravel_multi_index((2, 3, 4), (4, 5, 6)))),
        ((4, 5, 6), 0),
        ((4, 5, 6), int(np.ravel_multi_index((3, 4, 5), (4, 5, 6)))),
    ],
)
def test_neighbor_fast_path_preserves_nested_offset_order(shape, vertex) -> None:
    field = np.zeros(shape, dtype=np.float32)
    expected = np.asarray(list(_neighbors(vertex, shape)), dtype=np.uint32)

    observed = backend.neighbor_order(field, vertex)

    np.testing.assert_array_equal(observed, expected)
