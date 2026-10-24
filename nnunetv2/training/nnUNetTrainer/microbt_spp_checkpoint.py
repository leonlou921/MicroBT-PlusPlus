from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from typing import Mapping

import torch


birth_initial_logit = 2.0
_SEGMENTATION_HEAD = re.compile(
    r"(?:^|\.)unet\.decoder\.seg_layers\.\d+\.(?:weight|bias)$"
)


def map_three_channel_head_tensor(
    tensor: torch.Tensor, *, is_bias: bool
) -> torch.Tensor:
    """Map PF `(q,rc,re)` to SPP `(2q,const_birth,rc,re)`."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != 3:
        raise ValueError("segmentation-head tensor must have three output channels")
    birth = torch.zeros_like(tensor[0:1])
    if is_bias:
        if tensor.ndim != 1:
            raise ValueError("a segmentation-head bias must be one-dimensional")
        birth.fill_(birth_initial_logit)
    return torch.cat((2.0 * tensor[0:1], birth, tensor[1:2], tensor[2:3]), dim=0)


def migrate_a3_network_weights(
    network_weights: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    head_keys = sorted(key for key in network_weights if _SEGMENTATION_HEAD.search(key))
    if any(network_weights[key].shape[0] == 4 for key in head_keys):
        raise ValueError("checkpoint already has four channels in a segmentation head")
    mapped_keys = tuple(key for key in head_keys if network_weights[key].shape[0] == 3)
    if not mapped_keys or len(mapped_keys) != len(head_keys):
        raise ValueError("checkpoint has no consistent three-channel segmentation heads")
    migrated = dict(network_weights)
    for key in mapped_keys:
        migrated[key] = map_three_channel_head_tensor(
            network_weights[key], is_bias=key.endswith(".bias")
        )
    return migrated, mapped_keys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate_checkpoint_file(source: Path, destination: Path) -> tuple[str, ...]:
    source = source.resolve(strict=True)
    destination = destination.resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "network_weights" not in checkpoint:
        raise ValueError("source is not an nnU-Net checkpoint with network_weights")
    network_weights, mapped_keys = migrate_a3_network_weights(checkpoint["network_weights"])
    migrated = dict(checkpoint)
    migrated["network_weights"] = network_weights
    migrated["trainer_name"] = "MicroBTSPPTrainerScheduledExact"
    migrated["microbt_spp_migration"] = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "mapping": "(q,rc,re)->(2q,const_birth_2,rc,re)",
        "mapped_keys": list(mapped_keys),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(migrated, temporary)
    temporary.replace(destination)
    return mapped_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a PF checkpoint to MicroBT-SPP")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit(f"refusing to overwrite {args.destination}")
    mapped = migrate_checkpoint_file(args.source, args.destination)
    print(f"migrated {len(mapped)} head tensors to {args.destination}")


if __name__ == "__main__":
    main()
