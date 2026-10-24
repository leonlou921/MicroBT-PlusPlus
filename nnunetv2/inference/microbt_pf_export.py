from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nnunetv2.topology.persistent_field_0d import (
    decode_persistent_field,
    decode_support_birth_field,
)
from nnunetv2.topology.support_preserving_persistence import (
    decode_support_preserving_birth_field,
)


def decode_microbt_sbpf_cropped(
    birth_field: np.ndarray, region_probabilities: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Decode cropped SB-PF outputs and enforce one WT support contract."""
    birth = np.asarray(birth_field, dtype=np.float32)
    probabilities = np.asarray(region_probabilities, dtype=np.float32)
    if probabilities.shape != (3, *birth.shape):
        raise ValueError("SB-PF region probabilities must have shape [3,D,H,W]")
    decoded = decode_support_birth_field(birth, probabilities[0])
    decoded_wt = decoded["instance_ids"] > 0
    semantic = np.zeros(birth.shape, dtype=np.uint8)
    semantic[decoded_wt] = 1
    semantic[decoded_wt & (probabilities[1] > 0.5)] = 2
    semantic[decoded_wt & (probabilities[2] > 0.5)] = 3
    return semantic, decoded


def decode_microbt_spp_cropped(
    birth_field: np.ndarray, region_probabilities: np.ndarray
) -> tuple[np.ndarray, dict]:
    birth = np.asarray(birth_field, dtype=np.float32)
    probabilities = np.asarray(region_probabilities, dtype=np.float32)
    if probabilities.shape != (3, *birth.shape):
        raise ValueError("SPP region probabilities must have shape [3,D,H,W]")
    decoded = decode_support_preserving_birth_field(birth, probabilities[0])
    support = probabilities[0] >= 0.5
    if not np.array_equal(decoded["instance_ids"] > 0, support):
        raise RuntimeError("SPP instances do not conserve semantic support")
    semantic = np.zeros(birth.shape, dtype=np.uint8)
    semantic[support] = 1
    semantic[support & (probabilities[1] > 0.5)] = 2
    semantic[support & (probabilities[2] > 0.5)] = 3
    return semantic, decoded


def _resample_raw_logits(
    raw_logits: torch.Tensor | np.ndarray,
    properties: dict,
    configuration_manager,
    plans_manager,
) -> torch.Tensor:
    logits = torch.as_tensor(raw_logits).float()
    spacing_transposed = [
        properties["spacing"][index] for index in plans_manager.transpose_forward
    ]
    target_shape = properties["shape_after_cropping_and_before_resampling"]
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(target_shape)
        else [spacing_transposed[0], *configuration_manager.spacing]
    )
    resampled = configuration_manager.resampling_fn_probabilities(
        logits,
        target_shape,
        current_spacing,
        spacing_transposed,
    )
    return torch.as_tensor(resampled).float()


def _insert_crop(array: np.ndarray, properties: dict, fill_value) -> np.ndarray:
    shape = tuple(int(value) for value in properties["shape_before_cropping"])
    output = np.full(shape, fill_value, dtype=array.dtype)
    bbox = properties["bbox_used_for_cropping"]
    slicer = tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)
    if output[slicer].shape != array.shape:
        raise ValueError("cropped prediction shape does not match bbox_used_for_cropping")
    output[slicer] = array
    return output


def _restore_spatial(array: np.ndarray, properties: dict, plans_manager, fill_value):
    transposed = _insert_crop(array, properties, fill_value)
    return transposed.transpose(tuple(int(value) for value in plans_manager.transpose_backward))


def _restore_probabilities(
    probabilities: np.ndarray, properties: dict, plans_manager
) -> np.ndarray:
    channels = [
        _restore_spatial(channel, properties, plans_manager, 0.0)
        for channel in probabilities
    ]
    return np.stack(channels).astype(np.float32, copy=False)


def _objects_in_original_geometry(
    decoded: dict, field: np.ndarray, instance_ids: np.ndarray
) -> tuple[dict, ...]:
    by_identifier = {int(item["instance_id"]): item for item in decoded["objects"]}
    objects: list[dict] = []
    for instance_id in sorted(int(value) for value in np.unique(instance_ids) if value > 0):
        mask = instance_ids == instance_id
        coordinates = np.argwhere(mask)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        maximum = float(field[mask].max())
        centroid = np.argwhere(mask & (field == maximum))[0]
        source = by_identifier[instance_id]
        objects.append(
            {
                "instance_id": int(instance_id),
                "centroid_zyx": [int(value) for value in centroid],
                "box_zyx": [
                    int(lower[0]),
                    int(lower[1]),
                    int(lower[2]),
                    int(upper[0]),
                    int(upper[1]),
                    int(upper[2]),
                ],
                "birth": float(source["birth"]),
                "death": float(source["death"]),
                "volume": int(mask.sum()),
            }
        )
    return tuple(objects)


def restore_microbt_pf_prediction(
    raw_logits: torch.Tensor | np.ndarray,
    properties: dict,
    configuration_manager,
    plans_manager,
    *,
    a0: bool = False,
    spp: bool = False,
) -> dict:
    from nnunetv2.inference.microbt_pf_predictor import (
        raw_logits_to_microbt_pf,
        raw_logits_to_microbt_sbpf,
        raw_logits_to_microbt_spp,
    )

    resampled_logits = _resample_raw_logits(
        raw_logits, properties, configuration_manager, plans_manager
    )
    is_sbpf = int(resampled_logits.shape[0]) == 4
    if is_sbpf:
        if a0:
            raise ValueError("the A0 decoder is incompatible with four-channel SB-PF")
        converted = (
            raw_logits_to_microbt_spp(resampled_logits)
            if spp
            else raw_logits_to_microbt_sbpf(resampled_logits)
        )
        field_cropped = (
            converted["birth_field"].cpu().numpy().astype(np.float32, copy=False)
        )
    else:
        converted = raw_logits_to_microbt_pf(resampled_logits, a0=a0)
        field_cropped = converted["field"].cpu().numpy().astype(np.float32, copy=False)
    probabilities_cropped = (
        converted["region_probabilities"].cpu().numpy().astype(np.float32, copy=False)
    )
    if is_sbpf:
        decoder = decode_microbt_spp_cropped if spp else decode_microbt_sbpf_cropped
        semantic_cropped, decoded = decoder(field_cropped, probabilities_cropped)
    else:
        semantic_cropped = np.zeros(field_cropped.shape, dtype=np.uint8)
        semantic_cropped[probabilities_cropped[0] > 0.5] = 1
        semantic_cropped[probabilities_cropped[1] > 0.5] = 2
        semantic_cropped[probabilities_cropped[2] > 0.5] = 3
        decoded = decode_persistent_field(field_cropped)
    segmentation = _restore_spatial(
        semantic_cropped, properties, plans_manager, 0
    ).astype(np.uint8, copy=False)
    field = _restore_spatial(field_cropped, properties, plans_manager, -1.0).astype(
        np.float32, copy=False
    )
    instance_ids = _restore_spatial(
        decoded["instance_ids"].astype(np.int32, copy=False),
        properties,
        plans_manager,
        0,
    ).astype(np.int32, copy=False)
    probabilities = _restore_probabilities(
        probabilities_cropped, properties, plans_manager
    )
    objects = _objects_in_original_geometry(decoded, field, instance_ids)
    return {
        "raw_logits": resampled_logits.cpu().numpy().astype(np.float32, copy=False),
        "region_probabilities": probabilities,
        "segmentation": segmentation,
        "field": field,
        "instance_ids": instance_ids,
        "objects": objects,
        "count": len(objects),
        "diagnostics": {
            key: decoded[key]
            for key in (
                "marker_outside_support",
                "retained_marker_count",
                "fallback_marker_count",
                "support_voxel_count",
                "support_conserved",
            )
            if key in decoded
        },
    }


def export_microbt_pf_prediction(
    raw_logits: torch.Tensor | np.ndarray,
    properties: dict,
    configuration_manager,
    plans_manager,
    dataset_json: dict,
    output_file_truncated: str,
    *,
    save_field: bool = False,
    a0: bool = False,
    spp: bool = False,
) -> dict:
    restored = restore_microbt_pf_prediction(
        raw_logits,
        properties,
        configuration_manager,
        plans_manager,
        a0=a0,
        spp=spp,
    )
    output = Path(output_file_truncated)
    output.parent.mkdir(parents=True, exist_ok=True)
    ending = dataset_json["file_ending"]
    writer = plans_manager.image_reader_writer_class()
    writer.write_seg(restored["segmentation"], str(output) + ending, properties)
    writer.write_seg(
        restored["instance_ids"].astype(np.int32, copy=False),
        str(output) + "_instances" + ending,
        properties,
    )
    object_payload = {
        "count": int(restored["count"]),
        "objects": list(restored["objects"]),
    }
    (output.parent / f"{output.name}_objects.json").write_text(
        json.dumps(object_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if save_field:
        np.savez_compressed(
            output.parent / f"{output.name}_field.npz",
            raw_logits=restored["raw_logits"],
            field=restored["field"],
        )
    return restored
