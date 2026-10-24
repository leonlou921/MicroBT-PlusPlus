from __future__ import annotations

import numpy as np
import pytest
import torch

from nnunetv2.topology import persistent_field_0d as topology


class _FakeBackend:
    @staticmethod
    def persistence_pairs(field: np.ndarray, floor: float) -> dict[str, np.ndarray]:
        assert field.dtype == np.float32
        assert field.flags.c_contiguous
        assert floor == -1.0
        return {
            "birth_flat": np.asarray([2, 0], dtype=np.uint32),
            "death_flat": np.asarray([1, -1], dtype=np.int64),
            "essential": np.asarray([False, True], dtype=bool),
        }

    @staticmethod
    def induced_matching(
        prediction: np.ndarray, target: np.ndarray, floor: float
    ) -> dict[str, np.ndarray]:
        pairs = _FakeBackend.persistence_pairs(prediction, floor)
        return {
            "prediction_birth_flat": pairs["birth_flat"],
            "prediction_death_flat": pairs["death_flat"],
            "prediction_essential": pairs["essential"],
            "target_birth_flat": pairs["birth_flat"],
            "target_death_flat": pairs["death_flat"],
            "target_essential": pairs["essential"],
            "matched_prediction": np.asarray([0, 1], dtype=np.int64),
            "matched_target": np.asarray([0, 1], dtype=np.int64),
            "unmatched_prediction": np.empty(0, dtype=np.int64),
            "unmatched_target": np.empty(0, dtype=np.int64),
        }


def test_wrapper_returns_typed_dataclasses_and_normalizes_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(topology, "_backend", _FakeBackend())
    noncontiguous = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).transpose(0, 1)

    pairs = topology.persistence_pairs_0d(noncontiguous, floor=-1.0)
    matching = topology.induced_matching_0d(noncontiguous, noncontiguous, floor=-1.0)

    assert isinstance(pairs, topology.PersistencePairs0D)
    assert pairs.birth_flat.dtype == np.uint32
    assert pairs.death_flat.dtype == np.int64
    assert pairs.essential.dtype == bool
    assert isinstance(matching, topology.InducedMatching0D)
    np.testing.assert_array_equal(matching.prediction.birth_flat, pairs.birth_flat)
    np.testing.assert_array_equal(matching.target.death_flat, pairs.death_flat)
    np.testing.assert_array_equal(matching.matched_prediction, [0, 1])


def test_missing_extension_has_exact_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(topology, "_backend", None)
    with pytest.raises(RuntimeError, match="^MicroBT-PF H0 backend is unavailable$"):
        topology.persistence_pairs_0d(np.zeros((1, 1, 1), dtype=np.float32))


def test_backend_indices_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidBackend(_FakeBackend):
        @staticmethod
        def persistence_pairs(field, floor):
            result = _FakeBackend.persistence_pairs(field, floor)
            result["birth_flat"] = np.asarray([99, 0], dtype=np.uint32)
            return result

    monkeypatch.setattr(topology, "_backend", InvalidBackend())
    with pytest.raises(RuntimeError, match="birth index"):
        topology.persistence_pairs_0d(np.zeros((1, 1, 4), dtype=np.float32))


def test_exactly_one_essential_branch_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidBackend(_FakeBackend):
        @staticmethod
        def persistence_pairs(field, floor):
            result = _FakeBackend.persistence_pairs(field, floor)
            result["essential"] = np.asarray([False, False], dtype=bool)
            return result

    monkeypatch.setattr(topology, "_backend", InvalidBackend())
    with pytest.raises(RuntimeError, match="exactly one essential"):
        topology.persistence_pairs_0d(np.zeros((1, 1, 4), dtype=np.float32))


def test_critical_values_keep_live_gradient_and_use_constant_floor() -> None:
    field = torch.tensor([[[0.7, -0.2, 0.9, 0.1]]], requires_grad=True)
    pairs = topology.PersistencePairs0D(
        birth_flat=np.asarray([2, 0], dtype=np.uint32),
        death_flat=np.asarray([1, -1], dtype=np.int64),
        essential=np.asarray([False, True], dtype=bool),
    )

    birth, death = topology.critical_values(field, pairs, floor=-1.0)

    torch.testing.assert_close(birth, torch.tensor([0.9, 0.7]))
    torch.testing.assert_close(death, torch.tensor([-0.2, -1.0]))
    (birth.sum() + death.sum()).backward()
    torch.testing.assert_close(field.grad, torch.tensor([[[1.0, 1.0, 1.0, 0.0]]]))


def test_nonfinite_discrete_input_is_rejected_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(topology, "_backend", _FakeBackend())
    with pytest.raises(ValueError, match="finite"):
        topology.persistence_pairs_0d(torch.tensor([[[float("nan")]]]))


def test_precomputed_target_wrapper_is_bitwise_equal_to_normal_matching() -> None:
    prediction = np.asarray(
        [[[1.0, -0.25, 0.75], [-0.5, 0.25, -1.0]]], dtype=np.float32
    )
    target = np.asarray(
        [[[0.75, -0.5, 1.0], [-0.25, 0.5, -1.0]]], dtype=np.float32
    )
    target_pairs = topology.persistence_pairs_0d(target)

    reference = topology.induced_matching_0d(prediction, target)
    precomputed = topology.induced_matching_0d_precomputed_target(
        prediction, target, target_pairs
    )

    for pair_name in ("prediction", "target"):
        actual_pairs = getattr(precomputed, pair_name)
        expected_pairs = getattr(reference, pair_name)
        np.testing.assert_array_equal(actual_pairs.birth_flat, expected_pairs.birth_flat)
        np.testing.assert_array_equal(actual_pairs.death_flat, expected_pairs.death_flat)
        np.testing.assert_array_equal(actual_pairs.essential, expected_pairs.essential)
    for name in (
        "matched_prediction",
        "matched_target",
        "unmatched_prediction",
        "unmatched_target",
    ):
        np.testing.assert_array_equal(getattr(precomputed, name), getattr(reference, name))
def test_cuda_backend_requirement_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_cuda_backend", None)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        topology.require_cuda_exact_backend()


def test_backend_info_has_explicit_cuda_identity(monkeypatch) -> None:
    class FakeCudaBackend:
        @staticmethod
        def build_id():
            return "a" * 64

    monkeypatch.setattr(topology, "_cuda_backend", FakeCudaBackend())
    assert topology.exact_backend_info() == {
        "cpu_available": topology._backend is not None,
        "cuda_available": True,
        "cuda_build_id": "a" * 64,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("seed", [0, 11, 37])
def test_cuda_canonical_order_matches_cpu_lexicographic_oracle(seed: int) -> None:
    generator = np.random.default_rng(seed)
    field = generator.integers(-4, 5, size=(7, 9, 11)).astype(np.float32)
    expected = np.lexsort(
        (np.arange(field.size, dtype=np.int64), -field.reshape(-1))
    )
    actual = topology.canonical_order_cuda(torch.from_numpy(field).cuda())
    assert actual.is_cuda
    assert actual.dtype == torch.int64
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_canonical_order_is_bitwise_deterministic() -> None:
    field = torch.randint(-3, 4, (13, 17, 19), device="cuda", dtype=torch.int32)
    field = field.to(torch.float32)
    reference = topology.canonical_order_cuda(field)
    for _ in range(10):
        torch.testing.assert_close(
            topology.canonical_order_cuda(field), reference, rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_canonical_order_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="nonempty three-dimensional"):
        topology.canonical_order_cuda(torch.empty((0, 2, 3), device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        topology.canonical_order_cuda(torch.ones((2, 3, 4), device="cuda")[:, :, ::2])
    invalid = torch.ones((2, 3, 4), device="cuda")
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        topology.canonical_order_cuda(invalid)


def _assert_pairs_equal(actual, expected) -> None:
    np.testing.assert_array_equal(actual.birth_flat, expected.birth_flat)
    np.testing.assert_array_equal(actual.death_flat, expected.death_flat)
    np.testing.assert_array_equal(actual.essential, expected.essential)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "shape,seed",
    [((1, 1, 1), 0), ((1, 1, 7), 1), ((3, 4, 5), 2), ((7, 8, 9), 3)],
)
def test_cuda_persistence_pairs_are_bitwise_equal_to_cpu(shape, seed: int) -> None:
    generator = np.random.default_rng(seed)
    field = generator.integers(-5, 6, size=shape).astype(np.float32)
    expected = topology.persistence_pairs_0d(field)
    actual = topology.persistence_pairs_0d_cuda(torch.from_numpy(field).cuda())
    _assert_pairs_equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_persistence_pairs_preserve_neighbor_and_elder_ties() -> None:
    fields = (
        np.asarray([[[1.0, 0.0, 1.0]]], dtype=np.float32),
        np.asarray(
            [
                [[2.0, 0.0, 2.0], [0.0, -1.0, 0.0]],
                [[2.0, 0.0, 2.0], [0.0, -1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    for field in fields:
        _assert_pairs_equal(
            topology.persistence_pairs_0d_cuda(torch.from_numpy(field).cuda()),
            topology.persistence_pairs_0d(field),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_persistence_pairs_never_use_cpu_field_conversion(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("full field CPU conversion is forbidden")

    monkeypatch.setattr(topology, "_discrete_field", forbidden)
    pairs = topology.persistence_pairs_0d_cuda(
        torch.tensor([[[1.0, 0.0, 0.5]]], device="cuda")
    )
    assert pairs.birth_flat.tolist() == [2, 0]
    assert pairs.death_flat.tolist() == [1, -1]
