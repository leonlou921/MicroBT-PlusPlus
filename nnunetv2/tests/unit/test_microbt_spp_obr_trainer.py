from __future__ import annotations

import torch

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.nnUNetTrainer.MicroBTSPPTrainer import (
    MicroBTSPPOBRTrainerScheduledExact,
    _MicroBTSPPOBRVariantCriterion,
    set_microbt_spp_scratch_seed,
)


SHAPE = (3, 3, 5)


def _scalar_values() -> dict[str, torch.Tensor]:
    return {
        "loss": torch.tensor(10.0),
        "region": torch.tensor(1.0),
        "object_morse": torch.tensor(9.0),
        "persistent_field": torch.tensor(9.0),
        "field": torch.tensor(2.0),
        "betti_0": torch.tensor(3.0),
        "false_persistence": torch.tensor(1.0),
        "birth_reservation": torch.tensor(4.0),
        "birth_existence": torch.tensor(3.0),
        "birth_prominence": torch.tensor(1.0),
    }


def _singleton_target() -> PersistentFieldTarget:
    instance_ids = torch.zeros(SHAPE, dtype=torch.int32)
    instance_ids[1, 1, 2] = 1
    core_ids = instance_ids.clone()
    return PersistentFieldTarget(
        field=torch.full(SHAPE, -1.0),
        instance_ids=instance_ids,
        dense_valid_mask=torch.ones(SHAPE, dtype=torch.bool),
        valid_instance_ids=(1,),
        centers_zyx=((1, 1, 2),),
        birth_core_ids=core_ids,
        birth_moat_ids=torch.zeros_like(core_ids),
        precompute_topology=False,
    )


def test_object_family_ramp_keeps_raw_obr_telemetry() -> None:
    weighted = _MicroBTSPPOBRVariantCriterion._apply_object_morse_weight(
        _scalar_values(), 0.5
    )

    torch.testing.assert_close(weighted["field"], torch.tensor(1.0))
    torch.testing.assert_close(weighted["betti_0"], torch.tensor(1.5))
    torch.testing.assert_close(weighted["false_persistence"], torch.tensor(0.5))
    torch.testing.assert_close(weighted["birth_reservation"], torch.tensor(2.0))
    torch.testing.assert_close(weighted["birth_existence"], torch.tensor(1.5))
    torch.testing.assert_close(weighted["birth_prominence"], torch.tensor(0.5))
    torch.testing.assert_close(weighted["object_morse"], torch.tensor(4.5))
    torch.testing.assert_close(weighted["loss"], torch.tensor(5.5))
    torch.testing.assert_close(weighted["birth_reservation_raw"], torch.tensor(4.0))
    torch.testing.assert_close(weighted["birth_existence_raw"], torch.tensor(3.0))
    torch.testing.assert_close(weighted["birth_prominence_raw"], torch.tensor(1.0))


def test_sparse_exact_h0_weight_does_not_multiply_every_step_obr() -> None:
    values = _MicroBTSPPOBRVariantCriterion._apply_object_morse_weight(
        _scalar_values(), 0.5
    )

    weighted = MicroBTSPPOBRTrainerScheduledExact._apply_betti_weight(values, 10.0)

    torch.testing.assert_close(weighted["field"], torch.tensor(1.0))
    torch.testing.assert_close(weighted["betti_0"], torch.tensor(15.0))
    torch.testing.assert_close(weighted["false_persistence"], torch.tensor(5.0))
    torch.testing.assert_close(weighted["birth_reservation"], torch.tensor(2.0))
    torch.testing.assert_close(weighted["object_morse"], torch.tensor(18.0))
    torch.testing.assert_close(weighted["loss"], torch.tensor(19.0))
    torch.testing.assert_close(weighted["birth_reservation_raw"], torch.tensor(4.0))


def test_dense_steps_optimize_real_field_plus_birth_reservation() -> None:
    target = _singleton_target()
    logits = torch.zeros((1, 4, *SHAPE), requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = -4.0
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

    observed = _MicroBTSPPOBRVariantCriterion(h0_matching=False)(
        output, torch.zeros((1, 3, *SHAPE)), (target,)
    )

    assert observed["birth_reservation"].item() > 0.0
    assert observed["betti_0"].item() == 0.0
    torch.testing.assert_close(
        observed["object_morse"],
        observed["field"] + observed["birth_reservation"],
    )
    observed["object_morse"].backward()
    assert logits.grad is not None
    support_gradient = logits.grad[0, 0]
    assert support_gradient[1, 1, 2].item() < 0.0
    outside = support_gradient.clone()
    outside[1, 1, 2] = 0.0
    assert torch.count_nonzero(outside).item() == 0


def test_scratch_seed_recreates_the_same_untrained_parameter_initialization() -> None:
    set_microbt_spp_scratch_seed(42)
    first = torch.nn.Conv3d(4, 4, 1).state_dict()
    set_microbt_spp_scratch_seed(42)
    second = torch.nn.Conv3d(4, 4, 1).state_dict()

    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key])
