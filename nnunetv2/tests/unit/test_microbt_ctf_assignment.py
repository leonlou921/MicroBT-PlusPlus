from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_ctf_losses import EqualInstanceAssignmentConsistencyLoss
from nnunetv2.training.nnUNetTrainer.MicroBTCTFTrainer import MicroBTCTFAssignmentTrainerScratch60


def _target(ids: torch.Tensor, centers: tuple[tuple[int, int, int], ...]) -> PersistentFieldTarget:
    return PersistentFieldTarget(
        field=torch.zeros_like(ids, dtype=torch.float32),
        instance_ids=ids.to(torch.int32),
        dense_valid_mask=torch.ones_like(ids, dtype=torch.bool),
        valid_instance_ids=tuple(range(1, len(centers) + 1)),
        centers_zyx=centers,
        precompute_topology=False,
    )


def test_assignment_penalizes_one_mm_endpoint_error_in_physical_space() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    ids[1, 1, 1] = 1
    transport = torch.zeros((1, 3, 3, 3, 3), requires_grad=True)
    transport.data[0, 0, 1, 1, 1] = 0.1
    support = torch.ones((1, 1, 3, 3, 3), requires_grad=True)
    criterion = EqualInstanceAssignmentConsistencyLoss(
        offset_scale_zyx=(10, 10, 10), spacing_zyx=(1, 1, 1), endpoint_scale_mm=5
    )

    total, endpoint, support_assignment, endpoint_error = criterion(
        transport, support, [_target(ids, ((1, 1, 1),))]
    )

    assert total.item() > 0
    assert endpoint.item() > 0
    assert support_assignment.item() == 0
    torch.testing.assert_close(endpoint_error, torch.tensor(1.0))
    total.backward()
    assert transport.grad[0, 0, 1, 1, 1].item() > 0


def test_assignment_support_gradient_covers_every_lesion_voxel() -> None:
    ids = torch.zeros((4, 4, 4), dtype=torch.int32)
    ids[1:3, 1:3, 1:3] = 1
    support = torch.zeros((1, 1, 4, 4, 4), requires_grad=True)
    criterion = EqualInstanceAssignmentConsistencyLoss(
        offset_scale_zyx=(4, 4, 4), spacing_zyx=(1, 1, 1)
    )

    total, _, assignment, _ = criterion(
        torch.zeros((1, 3, 4, 4, 4)), support, [_target(ids, ((1, 1, 1),))]
    )
    assert assignment.item() > 0
    total.backward()
    assert torch.all(support.grad[0, 0][ids == 1] < 0)
    assert torch.count_nonzero(support.grad[0, 0][ids == 0]) == 0


def test_assignment_vectorizes_instance_coordinate_extraction(monkeypatch) -> None:
    ids = torch.zeros((6, 6, 6), dtype=torch.int32)
    centers = []
    for instance_id, flat_index in enumerate(range(20), start=1):
        coordinate = (flat_index // 36, (flat_index % 36) // 6, flat_index % 6)
        ids[coordinate] = instance_id
        centers.append(coordinate)
    real_nonzero = torch.nonzero
    calls = 0

    def counted_nonzero(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_nonzero(*args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", counted_nonzero)
    EqualInstanceAssignmentConsistencyLoss(
        offset_scale_zyx=(6, 6, 6), spacing_zyx=(1, 1, 1)
    )(
        torch.zeros((1, 3, 6, 6, 6)),
        torch.ones((1, 1, 6, 6, 6)),
        [_target(ids, tuple(centers))],
    )
    assert calls == 1


def test_assignment_trainer_and_launcher_are_scratch_isolated() -> None:
    assert MicroBTCTFAssignmentTrainerScratch60.scratch_seed == 42
    assert MicroBTCTFAssignmentTrainerScratch60.causal_epochs == 60
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_microbt_ctf_assignment60.sh"), "--print-contract"],
        check=True, capture_output=True, text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["isolated_root"] == "/path/to/example"
    assert contract["trainer"] == "MicroBTCTFAssignmentTrainerScratch60"
    assert contract["from_scratch"] is True
    assert contract["pretrained"] is False
    assert contract["exact_h0"] is False
    assert contract["full203_default"] is False
    assert contract["count_mae_gate_lt"] == 15
