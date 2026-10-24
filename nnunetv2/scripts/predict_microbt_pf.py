from __future__ import annotations

import argparse
import torch

from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-volume raw-logit MicroBT-PF inference"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--model-folder", required=True)
    parser.add_argument("--folds", nargs="+", default=None)
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-field", action="store_true")
    parser.add_argument("--a0-decoder", action="store_true")
    parser.add_argument("--no-mirroring", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folds = None if args.folds is None else tuple(
        int(value) if value != "all" else value for value in args.folds
    )
    predictor = MicroBTPFPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.no_mirroring,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=folds,
        checkpoint_name=args.checkpoint,
    )
    predictor.predict_microbt_pf_from_files(
        args.input,
        args.output,
        save_field=args.save_field,
        a0=args.a0_decoder,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
