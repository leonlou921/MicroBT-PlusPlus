from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MicroBTResEncM200Trainer(nnUNetTrainer):
    """Official nnU-Net ResEnc-M segmentation control, matched to 200 epochs."""

    training_epochs = 200

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.training_epochs
        self.save_every = 50
