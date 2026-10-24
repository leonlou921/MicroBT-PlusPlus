from __future__ import annotations

import numpy as np
import pytest
import torch

import nnunetv2.training.dataloading.microbt_pf_targets as target_module
from nnunetv2.topology.persistent_field_0d import PersistencePairs0D
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentTargetTopology,
    build_persistent_field_target,
    precompute_persistent_target_topology,
)


def test_singleton_valid_object_owns_one_birth_core_and_no_moat() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[2, 2, 2] = 7

    target = build_persistent_field_target(
        instance_ids,
        centers_by_id={7: (2.0, 2.0, 2.0)},
        spacing_zyx=(1.0, 1.0, 1.0),
        precompute_topology=False,
    )

    assert torch.count_nonzero(target.birth_core_ids == 7).item() == 1
    assert target.birth_core_ids[2, 2, 2].item() == 7
    assert torch.count_nonzero(target.birth_moat_ids).item() == 0


def test_invalid_border_object_is_excluded_from_birth_reservation_maps() -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[0, 2, 2] = 3

    target = build_persistent_field_target(
        instance_ids,
        centers_by_id={3: (0.0, 2.0, 2.0)},
        spacing_zyx=(1.0, 1.0, 1.0),
        topology_margin=1,
        precompute_topology=False,
    )

    assert target.valid_instance_ids == ()
    assert torch.count_nonzero(target.birth_core_ids).item() == 0
    assert torch.count_nonzero(target.birth_moat_ids).item() == 0


def test_core_and_moat_ids_never_cross_object_identity() -> None:
    instance_ids = np.zeros((9, 9, 9), dtype=np.int32)
    instance_ids[1:4, 1:4, 1:4] = 4
    instance_ids[5:8, 5:8, 5:8] = 9

    target = build_persistent_field_target(
        instance_ids,
        centers_by_id={4: (2.0, 2.0, 2.0), 9: (6.0, 6.0, 6.0)},
        spacing_zyx=(1.0, 1.0, 1.0),
        precompute_topology=False,
    )

    for reservation_ids in (target.birth_core_ids, target.birth_moat_ids):
        nonzero = reservation_ids > 0
        assert torch.equal(
            reservation_ids[nonzero], target.instance_ids[nonzero]
        )
    assert set(torch.unique(target.birth_core_ids).tolist()) == {0, 4, 9}
    assert set(torch.unique(target.birth_moat_ids).tolist()) == {0, 4, 9}


def test_topology_precomputation_preserves_birth_reservation_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
    instance_ids[1:4, 1:4, 1:4] = 2
    target = build_persistent_field_target(
        instance_ids,
        centers_by_id={2: (2.0, 2.0, 2.0)},
        spacing_zyx=(1.0, 1.0, 1.0),
        precompute_topology=False,
    )
    empty = PersistentTargetTopology(
        pairs=PersistencePairs0D(
            birth_flat=np.asarray([], dtype=np.uint32),
            death_flat=np.asarray([], dtype=np.int64),
            essential=np.asarray([], dtype=bool),
        ),
        birth_values=np.asarray([], dtype=np.float32),
        death_values=np.asarray([], dtype=np.float32),
        branch_valid=np.asarray([], dtype=bool),
    )
    monkeypatch.setattr(target_module, "_build_target_topology", lambda value: empty)

    with_topology = precompute_persistent_target_topology(target)

    assert with_topology.topology is not None
    assert torch.equal(with_topology.birth_core_ids, target.birth_core_ids)
    assert torch.equal(with_topology.birth_moat_ids, target.birth_moat_ids)
