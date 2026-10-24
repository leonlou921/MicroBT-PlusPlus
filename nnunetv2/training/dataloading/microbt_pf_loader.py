from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from nnunetv2.training.dataloading.data_loader import (
    nnUNetDataLoader,
    crop_and_pad_nd,
)
from nnunetv2.training.dataloading.microbt_pf_targets import (
    PersistentFieldTarget,
    build_persistent_field_target,
    transform_centers_affine,
)


def _random_index(rng, length: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(length))
    return int(rng.randint(length))


def select_batch_sources(
    selected_keys: Sequence[str],
    oversampled_slots: Sequence[bool],
    global_instances: Sequence[tuple[str, int]],
    component_uniform: bool,
    rng=np.random,
) -> tuple[list[str], list[int | None]]:
    if len(selected_keys) != len(oversampled_slots):
        raise ValueError("selected_keys and oversampled_slots must have equal length")
    keys = list(selected_keys)
    selected_instances: list[int | None] = [None] * len(keys)
    if component_uniform and not global_instances and any(oversampled_slots):
        return keys, selected_instances
    if component_uniform:
        for slot, oversampled in enumerate(oversampled_slots):
            if oversampled:
                case_identifier, instance_id = global_instances[
                    _random_index(rng, len(global_instances))
                ]
                keys[slot] = case_identifier
                selected_instances[slot] = int(instance_id)
    return keys, selected_instances


def apply_synchronized_spatial(
    image: torch.Tensor,
    class_segmentation: torch.Tensor,
    instance_ids: torch.Tensor,
    centers_zyx: np.ndarray,
    spatial_transform=None,
    mirror_transform=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    if image.ndim != 4 or class_segmentation.ndim != 4 or instance_ids.ndim != 4:
        raise ValueError("image, class_segmentation, and instance_ids must be [C,D,H,W]")
    if instance_ids.shape[0] != 1:
        raise ValueError("instance_ids must have exactly one channel")
    centers = np.asarray(centers_zyx, dtype=np.float64)
    if centers.size == 0:
        centers = np.empty((0, 3), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_zyx must have shape [N,3]")

    input_shape = tuple(int(value) for value in image.shape[1:])
    stacked_segmentation = torch.cat(
        (class_segmentation.to(torch.int32), instance_ids.to(torch.int32)), dim=0
    )
    data_dict = {"image": image, "segmentation": stacked_segmentation}
    if spatial_transform is None:
        spatial_parameters = {
            "affine": None,
            "elastic_offsets": None,
            "center_location_in_pixels": [value / 2.0 for value in input_shape],
        }
    else:
        spatial_parameters = spatial_transform.get_parameters(**data_dict)
        if spatial_parameters.get("elastic_offsets") is not None:
            raise RuntimeError("MicroBT-PF requires p_elastic_deform=0")
        data_dict = spatial_transform.apply(data_dict, **spatial_parameters)

    mirror_axes: tuple[int, ...] = ()
    if mirror_transform is not None:
        mirror_parameters = mirror_transform.get_parameters(**data_dict)
        mirror_axes = tuple(int(axis) for axis in mirror_parameters["axes"])
        data_dict = mirror_transform.apply(data_dict, **mirror_parameters)

    output_shape = tuple(int(value) for value in data_dict["image"].shape[1:])
    transformed_centers = transform_centers_affine(
        centers,
        input_shape=input_shape,
        output_shape=output_shape,
        affine=spatial_parameters.get("affine"),
        center_location_in_pixels=tuple(
            float(value) for value in spatial_parameters["center_location_in_pixels"]
        ),
        mirror_axes=mirror_axes,
    )
    transformed_segmentation = data_dict["segmentation"]
    class_channels = class_segmentation.shape[0]
    transformed_class = transformed_segmentation[:class_channels].to(
        class_segmentation.dtype
    )
    transformed_instances = torch.round(
        transformed_segmentation[class_channels : class_channels + 1].float()
    ).to(instance_ids.dtype)
    return (
        data_dict["image"],
        transformed_class,
        transformed_instances,
        transformed_centers,
    )


def _region_membership(labelmap: torch.Tensor, region) -> torch.Tensor:
    labels = region if isinstance(region, (tuple, list)) else (region,)
    membership = torch.zeros_like(labelmap, dtype=torch.bool)
    for label_value in labels:
        membership |= labelmap == int(label_value)
    return membership


def build_region_and_persistent_targets(
    class_segmentation: torch.Tensor,
    instance_ids: torch.Tensor,
    centers_by_id: dict[int, tuple[float, float, float]],
    spacing_zyx: tuple[float, float, float],
    label_manager,
    precompute_topology: bool = True,
    persistent_target_builder=build_persistent_field_target,
) -> tuple[torch.Tensor, PersistentFieldTarget]:
    if class_segmentation.ndim != 4 or class_segmentation.shape[0] < 1:
        raise ValueError("class_segmentation must have shape [C,D,H,W]")
    if instance_ids.ndim != 3:
        raise ValueError("instance_ids must have shape [D,H,W]")
    labelmap = class_segmentation[0]
    regions = (
        label_manager.foreground_regions
        if label_manager.has_regions
        else label_manager.foreground_labels
    )
    region_target = torch.stack(
        [_region_membership(labelmap, region) for region in regions]
    ).to(torch.float32)
    persistent_target = persistent_target_builder(
        instance_ids.detach().cpu().numpy().astype(np.int32, copy=False),
        centers_by_id=centers_by_id,
        spacing_zyx=spacing_zyx,
        topology_margin=1,
        precompute_topology=precompute_topology,
    )
    return region_target, persistent_target


def _load_cached_case(
    cache_directory: Path, identifier: str
) -> tuple[np.ndarray, dict[int, tuple[float, float, float]], tuple[float, float, float]]:
    case_directory = cache_directory / identifier
    with np.load(case_directory / "instances.npz") as archive:
        instance_ids = archive["instance_ids"].astype(np.int32, copy=False)
    metadata = json.loads(
        (case_directory / "metadata.json").read_text(encoding="utf-8")
    )
    centers = {
        int(record["id"]): tuple(float(value) for value in record["center_zyx"])
        for record in metadata["instances"]
    }
    spacing = tuple(float(value) for value in metadata["spacing_zyx"])
    return instance_ids, centers, spacing


def _global_instance_index(cache_directory: Path, identifiers: Iterable[str]):
    index: list[tuple[str, int]] = []
    for identifier in sorted(identifiers):
        metadata = json.loads(
            (cache_directory / identifier / "metadata.json").read_text(encoding="utf-8")
        )
        index.extend((identifier, int(record["id"])) for record in metadata["instances"])
    return tuple(index)


class MicroBTPFDataLoader(nnUNetDataLoader):
    def __init__(
        self,
        data,
        batch_size,
        patch_size,
        final_patch_size,
        label_manager,
        metadata_cache_directory: str | Path,
        component_uniform: bool,
        precompute_topology: bool = True,
        persistent_target_builder=build_persistent_field_target,
        oversample_foreground_percent: float = 0.33,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        spatial_transform=None,
        mirror_transform=None,
        intensity_transform=None,
    ) -> None:
        super().__init__(
            data,
            batch_size,
            patch_size,
            final_patch_size,
            label_manager,
            oversample_foreground_percent=oversample_foreground_percent,
            sampling_probabilities=sampling_probabilities,
            pad_sides=pad_sides,
            probabilistic_oversampling=probabilistic_oversampling,
            transforms=None,
        )
        if self.patch_size_was_2d:
            raise RuntimeError("MicroBT-PF mainline requires a three-dimensional configuration")
        self.label_manager = label_manager
        self.metadata_cache_directory = Path(metadata_cache_directory).resolve()
        self.component_uniform = bool(component_uniform)
        self.precompute_topology = bool(precompute_topology)
        self.persistent_target_builder = persistent_target_builder
        self.spatial_transform = spatial_transform
        self.mirror_transform = mirror_transform
        self.intensity_transform = intensity_transform
        self.global_instances = _global_instance_index(
            self.metadata_cache_directory, self._data.identifiers
        )

    def _bbox_around_instance_center(
        self, shape: Sequence[int], center_zyx: Sequence[float]
    ) -> tuple[list[int], list[int]]:
        need_to_pad = self.need_to_pad.copy()
        for axis in range(3):
            if need_to_pad[axis] + shape[axis] < self.patch_size[axis]:
                need_to_pad[axis] = self.patch_size[axis] - shape[axis]
        allowed_lower = [-int(need_to_pad[axis]) // 2 for axis in range(3)]
        lower = [
            max(
                allowed_lower[axis],
                int(round(center_zyx[axis])) - int(self.patch_size[axis]) // 2,
            )
            for axis in range(3)
        ]
        upper = [lower[axis] + int(self.patch_size[axis]) for axis in range(3)]
        return lower, upper

    def generate_train_batch(self) -> dict:
        standard_keys = list(self.get_indices())
        oversampled_slots = [self.get_do_oversample(slot) for slot in range(self.batch_size)]
        selected_keys, selected_instances = select_batch_sources(
            standard_keys,
            oversampled_slots,
            self.global_instances,
            self.component_uniform,
            np.random,
        )
        images: list[torch.Tensor] = []
        region_targets: list[torch.Tensor] = []
        persistent_targets: list[PersistentFieldTarget] = []

        for slot, identifier in enumerate(selected_keys):
            data, segmentation, previous_segmentation, properties = self._data.load_case(
                identifier
            )
            if previous_segmentation is not None:
                raise RuntimeError("MicroBT-PF mainline does not support cascaded input")
            full_instance_ids, full_centers, spacing = _load_cached_case(
                self.metadata_cache_directory, identifier
            )
            shape = data.shape[1:]
            selected_instance = selected_instances[slot]
            if selected_instance is None:
                bbox_lower, bbox_upper = self.get_bbox(
                    shape,
                    oversampled_slots[slot],
                    properties["class_locations"],
                )
            else:
                bbox_lower, bbox_upper = self._bbox_around_instance_center(
                    shape, full_centers[selected_instance]
                )
            bbox = [[int(lower), int(upper)] for lower, upper in zip(bbox_lower, bbox_upper)]
            cropped_image = crop_and_pad_nd(np.asarray(data), bbox, 0)
            cropped_class = crop_and_pad_nd(np.asarray(segmentation), bbox, -1)
            cropped_instances = crop_and_pad_nd(full_instance_ids[None], bbox, 0)
            cropped_centers = {
                instance_id: tuple(
                    center[axis] - bbox_lower[axis] for axis in range(3)
                )
                for instance_id, center in full_centers.items()
            }
            ordered_ids = tuple(sorted(cropped_centers))
            centers_array = np.asarray(
                [cropped_centers[instance_id] for instance_id in ordered_ids],
                dtype=np.float64,
            ).reshape(-1, 3)
            image, class_segmentation, instance_ids, transformed_centers = (
                apply_synchronized_spatial(
                    image=torch.from_numpy(cropped_image).float(),
                    class_segmentation=torch.from_numpy(cropped_class).to(torch.int32),
                    instance_ids=torch.from_numpy(cropped_instances).to(torch.int32),
                    centers_zyx=centers_array,
                    spatial_transform=self.spatial_transform,
                    mirror_transform=self.mirror_transform,
                )
            )
            transformed_centers_by_id = {
                instance_id: tuple(float(value) for value in transformed_centers[index])
                for index, instance_id in enumerate(ordered_ids)
            }
            if self.intensity_transform is not None:
                transformed = self.intensity_transform(
                    **{"image": image, "segmentation": class_segmentation}
                )
                image = transformed["image"]
                class_segmentation = transformed["segmentation"]
            region_target, persistent_target = build_region_and_persistent_targets(
                class_segmentation=class_segmentation,
                instance_ids=instance_ids[0],
                centers_by_id=transformed_centers_by_id,
                spacing_zyx=spacing,
                label_manager=self.label_manager,
                precompute_topology=self.precompute_topology,
                persistent_target_builder=self.persistent_target_builder,
            )
            images.append(image)
            region_targets.append(region_target)
            persistent_targets.append(persistent_target)

        return {
            "data": torch.stack(images),
            "target": torch.stack(region_targets),
            "persistent_targets": tuple(persistent_targets),
            "keys": selected_keys,
        }


class _TransformSequence:
    def __init__(self, transforms: Sequence) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, **data_dict):
        for transform in self.transforms:
            data_dict = transform(**data_dict)
        return data_dict


def _split_training_transforms(compose):
    spatial = None
    mirror = None
    intensity = []
    skipped_targets = {
        "RemoveLabelTansform",
        "ConvertSegmentationToRegionsTransform",
        "DownsampleSegForDSTransform",
    }
    unsupported = {
        "Convert3DTo2DTransform",
        "Convert2DTo3DTransform",
        "MoveSegAsOneHotToDataTransform",
        "ApplyRandomBinaryOperatorTransform",
        "RemoveRandomConnectedComponentFromOneHotEncodingTransform",
    }
    for transform in compose.transforms:
        name = type(transform).__name__
        if name == "SpatialTransform":
            if getattr(transform, "p_elastic_deform", 0) != 0:
                raise RuntimeError("MicroBT-PF requires p_elastic_deform=0")
            spatial = transform
        elif name == "MirrorTransform":
            mirror = transform
        elif name in skipped_targets:
            continue
        elif name in unsupported:
            raise RuntimeError(f"Unsupported MicroBT-PF transform: {name}")
        else:
            intensity.append(transform)
    if spatial is None:
        raise RuntimeError("The nnU-Net training transform has no SpatialTransform")
    return spatial, mirror, _TransformSequence(intensity)


def build_microbt_pf_dataloaders(
    trainer,
    component_uniform: bool,
    precompute_training_topology: bool = True,
    semantic_validation: bool = False,
    persistent_target_builder=build_persistent_field_target,
):
    if getattr(trainer, "enable_deep_supervision", False):
        raise RuntimeError("MicroBT-PF requires deep supervision to be disabled")
    if trainer.dataset_class is None:
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        trainer.dataset_class = infer_dataset_class(trainer.preprocessed_dataset_folder)
    patch_size = trainer.configuration_manager.patch_size
    rotation, dummy_2d, initial_patch_size, mirror_axes = (
        trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    )
    transforms = trainer.get_training_transforms(
        patch_size,
        rotation,
        None,
        mirror_axes,
        dummy_2d,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,
        is_cascaded=trainer.is_cascaded,
        foreground_labels=trainer.label_manager.foreground_labels,
        regions=(
            trainer.label_manager.foreground_regions
            if trainer.label_manager.has_regions
            else None
        ),
        ignore_label=trainer.label_manager.ignore_label,
    )
    spatial, mirror, intensity = _split_training_transforms(transforms)
    dataset_train, dataset_validation = trainer.get_tr_and_val_datasets()
    cache_root = os.environ.get("MICROBT_PF_CACHE")
    if not cache_root:
        raise RuntimeError("MICROBT_PF_CACHE must name the isolated cache root")
    dataset_name = Path(trainer.preprocessed_dataset_folder).parent.name
    plans_name = trainer.plans_manager.plans_name
    configuration_name = trainer.configuration_name
    cache_directory = (
        Path(cache_root) / dataset_name / plans_name / configuration_name
    )
    loader_train = MicroBTPFDataLoader(
        dataset_train,
        trainer.batch_size,
        initial_patch_size,
        patch_size,
        trainer.label_manager,
        cache_directory,
        component_uniform=component_uniform,
        precompute_topology=precompute_training_topology,
        persistent_target_builder=persistent_target_builder,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        probabilistic_oversampling=trainer.probabilistic_oversampling,
        spatial_transform=spatial,
        mirror_transform=mirror,
        intensity_transform=intensity,
    )
    if semantic_validation:
        validation_transforms = trainer.get_validation_transforms(
            trainer._get_deep_supervision_scales(),
            is_cascaded=trainer.is_cascaded,
            foreground_labels=trainer.label_manager.foreground_labels,
            regions=(
                trainer.label_manager.foreground_regions
                if trainer.label_manager.has_regions
                else None
            ),
            ignore_label=trainer.label_manager.ignore_label,
        )
        loader_validation = nnUNetDataLoader(
            dataset_validation,
            trainer.batch_size,
            patch_size,
            patch_size,
            trainer.label_manager,
            oversample_foreground_percent=trainer.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=validation_transforms,
            probabilistic_oversampling=trainer.probabilistic_oversampling,
        )
    else:
        loader_validation = MicroBTPFDataLoader(
            dataset_validation,
            trainer.batch_size,
            patch_size,
            patch_size,
            trainer.label_manager,
            cache_directory,
            component_uniform=False,
            precompute_topology=False,
            persistent_target_builder=persistent_target_builder,
            oversample_foreground_percent=trainer.oversample_foreground_percent,
            probabilistic_oversampling=trainer.probabilistic_oversampling,
        )

    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
    from batchgenerators.dataloading.single_threaded_augmenter import (
        SingleThreadedAugmenter,
    )
    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

    allowed_processes = get_allowed_n_proc_DA()
    if allowed_processes == 0:
        return (
            SingleThreadedAugmenter(loader_train, None),
            SingleThreadedAugmenter(loader_validation, None),
        )
    return (
        NonDetMultiThreadedAugmenter(
            data_loader=loader_train,
            transform=None,
            num_processes=allowed_processes,
            num_cached=max(6, allowed_processes // 2),
            seeds=None,
            pin_memory=trainer.device.type == "cuda",
            wait_time=0.002,
        ),
        NonDetMultiThreadedAugmenter(
            data_loader=loader_validation,
            transform=None,
            num_processes=max(1, allowed_processes // 2),
            num_cached=max(3, allowed_processes // 4),
            seeds=None,
            pin_memory=trainer.device.type == "cuda",
            wait_time=0.002,
        ),
    )
