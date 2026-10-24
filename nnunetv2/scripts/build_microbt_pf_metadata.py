from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nnunetv2.training.dataloading.microbt_pf_metadata import build_metadata_cache


def resolve_metadata_locations(
    dataset_directory: str | Path,
    cache_root: str | Path,
    configuration: str,
    plans_identifier: str,
) -> tuple[Path, Path]:
    dataset_directory = Path(dataset_directory).resolve()
    plans_path = dataset_directory / f"{plans_identifier}.json"
    if not plans_path.is_file():
        raise FileNotFoundError(f"Plans file does not exist: {plans_path}")
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    try:
        data_identifier = plans["configurations"][configuration]["data_identifier"]
    except KeyError as error:
        raise KeyError(
            f"Configuration {configuration!r} has no data_identifier in {plans_path}"
        ) from error
    source = (dataset_directory / data_identifier).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Preprocessed configuration directory does not exist: {source}")
    cache = (
        Path(cache_root)
        / dataset_directory.name
        / plans_identifier
        / configuration
    ).resolve()
    return source, cache


def build_cache_from_nnunet(
    dataset: str | int,
    configuration: str,
    plans_identifier: str,
    limit: int | None = None,
    workers: int = 1,
) -> list[Path]:
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import (
        maybe_convert_to_dataset_name,
    )

    if nnUNet_preprocessed is None:
        raise RuntimeError("nnUNet_preprocessed is not configured")
    cache_root = os.environ.get("MICROBT_PF_CACHE")
    if not cache_root:
        raise RuntimeError("MICROBT_PF_CACHE must name the isolated cache root")
    dataset_name = maybe_convert_to_dataset_name(dataset)
    source, cache = resolve_metadata_locations(
        dataset_directory=Path(nnUNet_preprocessed) / dataset_name,
        cache_root=cache_root,
        configuration=configuration,
        plans_identifier=plans_identifier,
    )
    return build_metadata_cache(source, cache, limit=limit, workers=workers)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable MicroBT-PF instance metadata outside nnUNet_preprocessed"
    )
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-c", "--configuration", required=True)
    parser.add_argument("-p", "--plans", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    outputs = build_cache_from_nnunet(
        dataset=arguments.dataset,
        configuration=arguments.configuration,
        plans_identifier=arguments.plans,
        limit=arguments.limit,
        workers=arguments.workers,
    )
    print(
        json.dumps(
            {"case_count": len(outputs), "case_directories": [str(path) for path in outputs]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
