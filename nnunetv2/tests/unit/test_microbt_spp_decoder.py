from __future__ import annotations

import numpy as np
import pytest

from nnunetv2.topology import support_preserving_persistence as spp


def _pairs(birth: list[int], death: list[int], essential: list[bool]):
    return spp.PersistencePairs0D(
        birth_flat=np.asarray(birth, dtype=np.uint32),
        death_flat=np.asarray(death, dtype=np.int64),
        essential=np.asarray(essential, dtype=bool),
    )


def test_spp_decoder_rejects_an_off_support_retained_birth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spp, "persistence_pairs_0d", lambda field, floor: _pairs([1], [-1], [True])
    )
    field = np.asarray([[[-1.0, 0.8, -1.0]]], dtype=np.float32)
    support = np.zeros_like(field)

    with pytest.raises(RuntimeError, match="outside semantic support"):
        spp.decode_support_preserving_birth_field(field, support)


def test_spp_decoder_completes_every_support_component_without_changing_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spp, "persistence_pairs_0d", lambda field, floor: _pairs([], [], [])
    )
    field = np.asarray([[[-0.9, -0.2, -1.0, -0.4, -0.8]]], dtype=np.float32)
    support_probability = np.asarray([[[0.9, 0.8, 0.1, 0.7, 0.9]]], dtype=np.float32)
    support = support_probability >= 0.5

    decoded = spp.decode_support_preserving_birth_field(field, support_probability)

    assert decoded["count"] == 2
    assert decoded["fallback_marker_count"] == 2
    assert decoded["marker_outside_support"] == 0
    np.testing.assert_array_equal(decoded["instance_ids"] > 0, support)
    assert all(bool(item["fallback_marker"]) for item in decoded["objects"])


def test_spp_decoder_partitions_one_support_component_with_two_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spp,
        "persistence_pairs_0d",
        lambda field, floor: _pairs([0, 4], [-1, 2], [True, False]),
    )
    field = np.asarray([[[1.0, 0.6, 0.2, 0.6, 0.8]]], dtype=np.float32)
    support = np.full_like(field, 0.9)

    decoded = spp.decode_support_preserving_birth_field(field, support)

    assert decoded["count"] == 2
    assert decoded["fallback_marker_count"] == 0
    assert np.all(decoded["instance_ids"] > 0)
    assert decoded["instance_ids"].ravel()[0] == 1
    assert decoded["instance_ids"].ravel()[4] == 2
