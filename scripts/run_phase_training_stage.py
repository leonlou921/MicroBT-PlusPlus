from __future__ import annotations

import argparse
import json
from pathlib import Path


CONTRACT = {
    "from_scratch_without_continue": True,
    "performs_actual_validation": False,
    "requires_checkpoint_to_continue": True,
}


def _normalize_fold(value: str) -> int | str:
    return value if value == "all" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--dataset", default="1")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--trainer", required=False)
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    args = parser.parse_args()
    if args.print_contract:
        print(json.dumps(CONTRACT, sort_keys=True))
        return
    if not args.trainer:
        parser.error("--trainer is required")

    import torch
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    device = torch.device("cuda")
    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        _normalize_fold(args.fold),
        args.trainer,
        args.plans,
        device=device,
    )
    if args.continue_training:
        candidates = (
            Path(trainer.output_folder) / "checkpoint_final.pth",
            Path(trainer.output_folder) / "checkpoint_latest.pth",
            Path(trainer.output_folder) / "checkpoint_best.pth",
        )
        if not any(path.is_file() for path in candidates):
            raise RuntimeError("continuation requested without an existing checkpoint")
    maybe_load_checkpoint(trainer, args.continue_training, False, None)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    trainer.run_training()


if __name__ == "__main__":
    main()
