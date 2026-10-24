from __future__ import annotations

import json

import torch

from nnunetv2.training.dataloading.microbt_pf_targets import PersistentFieldTarget
from nnunetv2.training.nnUNetTrainer.MicroBTSPPOBRCurriculumTrainer import (
    MicroBTSPPOBRCurriculumTrainerScratch200,
    _MicroBTSPPOBRCurriculumCriterion,
    curriculum_weights,
)


def _target(shape: tuple[int, int, int]) -> PersistentFieldTarget:
    field = torch.full(shape, -1.0)
    instance_ids = torch.zeros(shape, dtype=torch.int32)
    instance_ids[1:4, 1:4, 1:4] = 1
    field[1:4, 1:4, 1:4] = 0.25
    field[2, 2, 2] = 1.0
    return PersistentFieldTarget(
        field=field,
        instance_ids=instance_ids,
        dense_valid_mask=torch.ones(shape, dtype=torch.bool),
        valid_instance_ids=(1,),
        centers_zyx=((2, 2, 2),),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    shape = (5, 5, 5)
    target = _target(shape)
    region_target = torch.zeros((1, 3, *shape), device="cuda")
    region_target[:, :, 1:4, 1:4, 1:4] = 1.0
    results = []
    for epoch in (0, 15, 30):
        weights = curriculum_weights(epoch)
        _, exact, betti_weight = (
            MicroBTSPPOBRCurriculumTrainerScratch200.exact_h0_schedule(epoch, 0)
        )
        criterion = _MicroBTSPPOBRCurriculumCriterion(h0_matching=exact).cuda()
        criterion.set_curriculum_weights(weights)
        logits = torch.zeros((1, 4, *shape), device="cuda", requires_grad=True)
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
        values = criterion(output, region_target, (target,))
        if exact:
            values = MicroBTSPPOBRCurriculumTrainerScratch200._apply_betti_weight(
                values, betti_weight
            )
        values["loss"].backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise RuntimeError(f"nonfinite CUDA gradient at epoch {epoch}")
        if not torch.isfinite(values["loss"]):
            raise RuntimeError(f"nonfinite CUDA loss at epoch {epoch}")
        results.append(
            {
                "epoch": epoch,
                "exact_h0": exact,
                "loss": float(values["loss"].detach().cpu()),
                "gradient_norm": float(logits.grad.norm().detach().cpu()),
                "weights": weights._asdict(),
            }
        )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
