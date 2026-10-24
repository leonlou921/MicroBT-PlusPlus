from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
import json
import subprocess
from pathlib import Path

from nnunetv2.nets.microbt_ctf.model import ctf_logits_to_outputs
from nnunetv2.inference.microbt_ctf_decode import decode_ctf_instances, instance_boxes
from nnunetv2.training.dataloading.microbt_ctf_targets import (
    build_centroid_transport_target,
)
from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.loss.microbt_ctf_losses import (
    EqualInstanceCentroidTransportLoss,
    MicroBTCTFLoss,
)
from nnunetv2.training.nnUNetTrainer.MicroBTCTFTrainer import (
    MicroBTCTFTrainerScratch60,
    ctf_object_weight,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _target(instance_ids: torch.Tensor, centers: tuple[tuple[int, int, int], ...]):
    valid = tuple(int(v) for v in torch.unique(instance_ids).tolist() if int(v) > 0)
    return PersistentFieldTarget(
        field=torch.zeros_like(instance_ids, dtype=torch.float32),
        instance_ids=instance_ids.to(torch.int32),
        dense_valid_mask=torch.ones_like(instance_ids, dtype=torch.bool),
        valid_instance_ids=valid,
        centers_zyx=centers,
        precompute_topology=False,
    )


def _exact_transport(
    instance_ids: torch.Tensor,
    centers: tuple[tuple[int, int, int], ...],
    scale: tuple[float, float, float],
) -> torch.Tensor:
    result = torch.zeros((1, 3, *instance_ids.shape), dtype=torch.float32)
    for instance_id, center in enumerate(centers, start=1):
        coords = torch.nonzero(instance_ids == instance_id, as_tuple=False)
        target = (torch.tensor(center, dtype=torch.float32) - coords) / torch.tensor(
            scale, dtype=torch.float32
        )
        result[0, :, coords[:, 0], coords[:, 1], coords[:, 2]] = target.T
    return result


def test_ctf_six_logits_decode_to_nested_regions_and_bounded_transport() -> None:
    logits = torch.zeros((1, 6, 3, 3, 3), dtype=torch.float32)
    logits[:, 1:4] = 100.0
    output = ctf_logits_to_outputs(logits)

    assert output["transport"].shape == (1, 3, 3, 3, 3)
    assert output["region_probabilities"].shape == (1, 3, 3, 3, 3)
    assert torch.all(output["transport"] <= 1.0)
    wt, tc, et = output["region_probabilities"].unbind(dim=1)
    assert torch.all(et <= tc)
    assert torch.all(tc <= wt)


def test_transport_loss_is_zero_for_exact_centroid_offsets_and_backpropagates() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    ids[1, 1, 1] = 1
    target = _target(ids, ((1, 1, 1),))
    transport = torch.zeros((1, 3, 3, 3, 3), requires_grad=True)
    support = torch.ones((1, 1, 3, 3, 3), requires_grad=True)
    criterion = EqualInstanceCentroidTransportLoss(offset_scale_zyx=(1, 1, 1))

    total, offset, anchor, endpoint_error = criterion(transport, support, [target])

    torch.testing.assert_close(total, torch.tensor(0.0))
    torch.testing.assert_close(offset, torch.tensor(0.0))
    torch.testing.assert_close(anchor, torch.tensor(0.0))
    torch.testing.assert_close(endpoint_error, torch.tensor(0.0))
    total.backward()
    assert transport.grad is not None
    assert support.grad is not None


def test_transport_loss_weights_small_and_large_instances_equally() -> None:
    ids = torch.zeros((4, 4, 4), dtype=torch.int32)
    ids[0, 0, 0] = 1
    ids[2:4, 2:4, 2:4] = 2
    target = _target(ids, ((0, 0, 0), (2, 2, 2)))
    support = torch.ones((1, 1, 4, 4, 4))
    criterion = EqualInstanceCentroidTransportLoss(
        offset_scale_zyx=(4, 4, 4), anchor_weight=0.0
    )

    pred_small_wrong = _exact_transport(ids, ((0, 0, 0), (2, 2, 2)), (4, 4, 4))
    pred_small_wrong[:, :, 0, 0, 0] = 0.5
    _, small_offset, _, _ = criterion(pred_small_wrong, support, [target])

    pred_large_wrong = _exact_transport(ids, ((0, 0, 0), (2, 2, 2)), (4, 4, 4))
    pred_large_wrong[:, :, 2:4, 2:4, 2:4] += 0.5
    _, large_offset, _, _ = criterion(pred_large_wrong, support, [target])

    # Each instance owns one half of the batch loss, independent of voxel count.
    torch.testing.assert_close(small_offset, large_offset)


def test_centroid_support_anchor_penalizes_a_missing_micro_object() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    ids[1, 1, 1] = 1
    target = _target(ids, ((1, 1, 1),))
    transport = torch.zeros((1, 3, 3, 3, 3))
    support = torch.zeros((1, 1, 3, 3, 3), requires_grad=True)
    criterion = EqualInstanceCentroidTransportLoss(offset_scale_zyx=(1, 1, 1))

    total, _, anchor, _ = criterion(transport, support, [target])

    assert anchor.item() > 0.0
    total.backward()
    assert support.grad is not None
    assert support.grad[0, 0, 1, 1, 1].item() < 0.0


def test_empty_patch_has_differentiable_zero_object_loss() -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    target = _target(ids, ())
    transport = torch.zeros((1, 3, 3, 3, 3), requires_grad=True)
    support = torch.zeros((1, 1, 3, 3, 3), requires_grad=True)
    criterion = EqualInstanceCentroidTransportLoss(offset_scale_zyx=(1, 1, 1))

    total, offset, anchor, endpoint_error = criterion(transport, support, [target])

    for value in (total, offset, anchor, endpoint_error):
        torch.testing.assert_close(value, torch.tensor(0.0))
    total.backward()
    assert transport.grad is not None
    assert support.grad is not None


def test_transport_loss_vectorizes_instance_coordinate_extraction(monkeypatch) -> None:
    ids = torch.zeros((6, 6, 6), dtype=torch.int32)
    centers = []
    for instance_id, flat_index in enumerate(range(20), start=1):
        coordinate = (
            flat_index // 36,
            (flat_index % 36) // 6,
            flat_index % 6,
        )
        ids[coordinate] = instance_id
        centers.append(coordinate)
    target = _target(ids, tuple(centers))
    transport = torch.zeros((1, 3, 6, 6, 6))
    support = torch.ones((1, 1, 6, 6, 6))
    real_nonzero = torch.nonzero
    calls = 0

    def counted_nonzero(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_nonzero(*args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", counted_nonzero)
    EqualInstanceCentroidTransportLoss(offset_scale_zyx=(6, 6, 6))(
        transport, support, [target]
    )

    assert calls == 1


def test_vectorized_transport_is_numerically_equal_to_scalar_definition() -> None:
    ids = torch.zeros((5, 5, 5), dtype=torch.int32)
    ids[0, 0, 0] = 1
    ids[1:3, 1:4, 2:4] = 2
    ids[4, 3:5, 0:2] = 3
    centers = ((0, 0, 0), (1, 2, 2), (4, 3, 0))
    target = _target(ids, centers)
    generator = torch.Generator().manual_seed(42)
    transport = torch.rand((1, 3, 5, 5, 5), generator=generator) * 2.0 - 1.0
    support = torch.rand((1, 1, 5, 5, 5), generator=generator)
    criterion = EqualInstanceCentroidTransportLoss(
        offset_scale_zyx=(5, 5, 5), beta=0.1,
        center_support_threshold=0.5, anchor_weight=0.25,
    )

    actual = criterion(transport, support, [target])
    scalar_offsets = []
    scalar_anchors = []
    scalar_endpoints = []
    scale = torch.tensor((5.0, 5.0, 5.0))
    for instance_id, center in enumerate(centers, start=1):
        coords = torch.nonzero(ids == instance_id, as_tuple=False)
        wanted = ((torch.tensor(center) - coords) / scale).clamp(-1.0, 1.0)
        predicted = transport[0, :, coords[:, 0], coords[:, 1], coords[:, 2]].T
        scalar_offsets.append(F.smooth_l1_loss(predicted, wanted, beta=0.1))
        scalar_endpoints.append(torch.linalg.vector_norm((predicted - wanted) * scale, dim=1).mean())
        scalar_anchors.append(torch.relu(0.5 - support[(0, 0, *center)]).square())
    expected_offset = torch.stack(scalar_offsets).mean()
    expected_anchor = torch.stack(scalar_anchors).mean()
    expected_endpoint = torch.stack(scalar_endpoints).mean()
    expected_total = expected_offset + 0.25 * expected_anchor

    for observed, expected in zip(
        actual,
        (expected_total, expected_offset, expected_anchor, expected_endpoint),
        strict=True,
    ):
        torch.testing.assert_close(observed, expected)


def test_zero_object_weight_skips_all_transport_work(monkeypatch) -> None:
    ids = torch.zeros((3, 3, 3), dtype=torch.int32)
    ids[1, 1, 1] = 1
    criterion = MicroBTCTFLoss(object_weight=0.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("zero-weight warmup must not compute transport loss")

    monkeypatch.setattr(criterion.object_transport, "forward", forbidden)
    probabilities = torch.full((1, 3, 3, 3, 3), 0.5, requires_grad=True)
    values = criterion(
        {
            "transport": torch.zeros((1, 3, 3, 3, 3), requires_grad=True),
            "region_probabilities": probabilities,
        },
        torch.zeros((1, 3, 3, 3, 3)),
        [_target(ids, ((1, 1, 1),))],
    )

    torch.testing.assert_close(values["object_transport_raw"], torch.tensor(0.0))
    values["loss"].backward()
    assert probabilities.grad is not None


def test_centroid_only_target_matches_legacy_instance_centres_without_geodesics(
    monkeypatch,
) -> None:
    from nnunetv2.training.dataloading import microbt_pf_targets

    ids = torch.zeros((7, 7, 7), dtype=torch.int32)
    ids[1:3, 1:4, 1:3] = 2
    ids[4:6, 3:6, 4:6] = 9
    centers = {2: (1.2, 2.1, 1.4), 9: (4.8, 4.2, 4.7)}
    legacy = microbt_pf_targets.build_persistent_field_target(
        ids.numpy(), centers, (1.0, 0.9375, 0.9375), precompute_topology=False
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CTF centroid targets must not compute geodesic fields")

    monkeypatch.setattr(microbt_pf_targets, "spacing_aware_geodesic", forbidden)
    fast = build_centroid_transport_target(
        ids.numpy(), centers, (1.0, 0.9375, 0.9375), precompute_topology=False
    )

    assert fast.valid_instance_ids == legacy.valid_instance_ids
    assert fast.centers_zyx == legacy.centers_zyx
    assert torch.equal(fast.instance_ids, legacy.instance_ids)
    assert fast.topology is None
    assert torch.count_nonzero(fast.field) == 0


def test_ctf_trainer_requests_centroid_only_loader(monkeypatch) -> None:
    import nnunetv2.training.nnUNetTrainer.MicroBTCTFTrainer as trainer_module

    captured = {}

    def fake_builder(trainer, **kwargs):
        captured.update(kwargs)
        return "train", "validation"

    monkeypatch.setattr(trainer_module, "build_microbt_pf_dataloaders", fake_builder)
    trainer = object.__new__(MicroBTCTFTrainerScratch60)

    assert trainer.get_dataloaders() == ("train", "validation")
    assert captured["component_uniform"] is True
    assert captured["precompute_training_topology"] is False
    assert captured["persistent_target_builder"] is build_centroid_transport_target


def test_ctf_trainer_contract_is_scratch_h0_free_and_sixty_epochs() -> None:
    assert MicroBTCTFTrainerScratch60.scratch_seed == 42
    assert MicroBTCTFTrainerScratch60.causal_epochs == 60
    assert MicroBTCTFTrainerScratch60.snapshot_epochs == (15, 30, 40, 50, 60)
    assert MicroBTCTFTrainerScratch60.exact_h0_schedule(59, 249) == (
        "ctf_no_h0",
        False,
        0.0,
    )
    assert ctf_object_weight(0) == 0.0
    assert ctf_object_weight(4) == 0.0
    assert 0.0 < ctf_object_weight(5) < 0.25
    assert ctf_object_weight(29) == 0.25
    assert ctf_object_weight(59) == 0.25


def test_ctf_trainer_builds_a_real_six_channel_wrapper(monkeypatch) -> None:
    class DummyUNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 6, *x.shape[2:]), device=x.device)

    def build(*args, **kwargs):
        assert args[4] == 6
        assert kwargs["enable_deep_supervision"] is False
        return DummyUNet()

    monkeypatch.setattr(nnUNetTrainer, "build_network_architecture", build)
    network = MicroBTCTFTrainerScratch60.build_network_architecture(
        "dummy", {}, (), 4, 3, enable_deep_supervision=True
    )
    output = network(torch.zeros((1, 4, 3, 3, 3)))
    assert output["transport"].shape == (1, 3, 3, 3, 3)


def test_ctf_trainer_uses_only_the_two_family_ctf_loss() -> None:
    trainer = object.__new__(MicroBTCTFTrainerScratch60)
    criterion = trainer._build_loss()
    assert isinstance(criterion, MicroBTCTFLoss)


def test_ctf_voting_recovers_two_centres_counts_and_derived_boxes() -> None:
    shape = (8, 8, 8)
    support = torch.zeros(shape)
    support[1:3, 1:3, 1:3] = 0.9
    support[6, 6, 6] = 0.9
    transport = torch.zeros((3, *shape))
    for slices, center in (
        ((slice(1, 3), slice(1, 3), slice(1, 3)), (1, 1, 1)),
        ((slice(6, 7), slice(6, 7), slice(6, 7)), (6, 6, 6)),
    ):
        coords = torch.nonzero(support[slices] > 0, as_tuple=False)
        origin = torch.tensor(
            [s.start for s in slices], dtype=torch.float32
        )
        coords = coords.to(torch.float32) + origin
        offsets = torch.tensor(center, dtype=torch.float32) - coords
        transport[:, slices[0], slices[1], slices[2]] = offsets.T.reshape(
            3, *(s.stop - s.start for s in slices)
        )

    decoded = decode_ctf_instances(
        support.numpy(),
        transport.numpy(),
        offset_scale_zyx=(1.0, 1.0, 1.0),
        support_threshold=0.5,
        min_vote_mass=0.5,
        nms_radius=1,
    )

    assert decoded.count == 2
    assert set(decoded.centers_zyx) == {(1, 1, 1), (6, 6, 6)}
    assert set(instance_boxes(decoded.instance_labels)) == {
        (1, 1, 1, 3, 3, 3),
        (6, 6, 6, 7, 7, 7),
    }


def test_ctf_launcher_contract_is_fresh_isolated_and_gate_first() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "run_microbt_ctf_scratch60.sh"), "--print-contract"],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(completed.stdout)
    assert contract["isolated_root"] == "/path/to/example"
    assert contract["from_scratch"] is True
    assert contract["pretrained"] is False
    assert contract["exact_h0"] is False
    assert contract["formal_epochs"] == 60
    assert contract["stage_gate_before_full203"] is True
    assert contract["full203_default"] is False
    assert contract["transport_implementation"] == "vectorized_equal_instance"
    assert contract["target_builder"] == "centroid_only_no_geodesic"
