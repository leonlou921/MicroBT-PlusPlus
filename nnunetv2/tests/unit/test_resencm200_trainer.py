from __future__ import annotations

import unittest
from unittest.mock import patch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.MicroBTResEncM200Trainer import (
    MicroBTResEncM200Trainer,
)


class ResEncM200TrainerTest(unittest.TestCase):
    def test_is_pure_resencm_a0_with_matched_200_epoch_horizon(self) -> None:
        self.assertEqual(MicroBTResEncM200Trainer.__bases__, (nnUNetTrainer,))
        self.assertEqual(MicroBTResEncM200Trainer.training_epochs, 200)
        self.assertIs(MicroBTResEncM200Trainer._build_loss, nnUNetTrainer._build_loss)
        self.assertIs(
            MicroBTResEncM200Trainer.build_network_architecture,
            nnUNetTrainer.build_network_architecture,
        )

    def test_constructor_sets_num_epochs_before_optimizer_setup(self) -> None:
        def fake_parent_init(instance, *args, **kwargs) -> None:
            instance.num_epochs = 1000
            instance.save_every = 50

        with patch.object(nnUNetTrainer, "__init__", fake_parent_init):
            trainer = MicroBTResEncM200Trainer({}, "3d_fullres", "all", {})
        self.assertEqual(trainer.num_epochs, 200)
        self.assertEqual(trainer.save_every, 50)


if __name__ == "__main__":
    unittest.main()
