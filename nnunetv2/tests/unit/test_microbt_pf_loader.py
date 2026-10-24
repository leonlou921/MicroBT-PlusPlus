from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)


def _crop_and_pad_nd(array: np.ndarray, bbox: list[list[int]], pad_value: int | float):
    output_shape = (array.shape[0], *(upper - lower for lower, upper in bbox))
    output = np.full(output_shape, pad_value, dtype=array.dtype)
    source_slices = []
    target_slices = []
    for axis, (lower, upper) in enumerate(bbox):
        source_lower = max(0, lower)
        source_upper = min(array.shape[axis + 1], upper)
        target_lower = source_lower - lower
        target_upper = target_lower + source_upper - source_lower
        source_slices.append(slice(source_lower, source_upper))
        target_slices.append(slice(target_lower, target_upper))
    output[(slice(None), *target_slices)] = array[(slice(None), *source_slices)]
    return output


class _StubNNUNetDataLoader:
    def __init__(
        self,
        data,
        batch_size,
        patch_size,
        final_patch_size,
        label_manager,
        oversample_foreground_percent=0.0,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling=False,
        transforms=None,
    ) -> None:
        self._data = data
        self.batch_size = batch_size
        self.patch_size = np.asarray(patch_size)
        self.final_patch_size = np.asarray(final_patch_size)
        self.need_to_pad = self.patch_size - self.final_patch_size
        self.label_manager = label_manager
        self.oversample_foreground_percent = oversample_foreground_percent
        self.patch_size_was_2d = False
        sample_data, sample_seg, _, _ = data.load_case(data.identifiers[0])
        self.data_shape = (batch_size, sample_data.shape[0], *self.patch_size)
        self.seg_shape = (batch_size, sample_seg.shape[0], *self.patch_size)

    def get_indices(self):
        return [self._data.identifiers[0]] * self.batch_size

    def get_do_oversample(self, sample_idx: int) -> bool:
        return not sample_idx < round(
            self.batch_size * (1 - self.oversample_foreground_percent)
        )

    def get_bbox(self, shape, force_fg, class_locations):
        return [0, 0, 0], self.patch_size.tolist()


_data_loader_module = types.ModuleType("nnunetv2.training.dataloading.data_loader")
_data_loader_module.nnUNetDataLoader = _StubNNUNetDataLoader
_data_loader_module.crop_and_pad_nd = _crop_and_pad_nd
sys.modules.setdefault("nnunetv2.training.dataloading.data_loader", _data_loader_module)

from nnunetv2.training.dataloading.microbt_pf_loader import (  # noqa: E402
    MicroBTPFDataLoader,
    apply_synchronized_spatial,
    build_region_and_persistent_targets,
    select_batch_sources,
)
from nnunetv2.training.dataloading.microbt_pf_targets import (  # noqa: E402
    build_persistent_field_target,
)
from nnunetv2.topology.persistent_field_0d import persistence_pairs_0d  # noqa: E402


class _FixedRotation:
    def get_parameters(self, **data_dict):
        return {
            "affine": np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "elastic_offsets": None,
            "center_location_in_pixels": [2.5, 2.5, 2.5],
        }

    def apply(self, data_dict, **params):
        data_dict["image"] = torch.rot90(data_dict["image"], 1, (1, 2))
        data_dict["segmentation"] = torch.rot90(
            data_dict["segmentation"], 1, (1, 2)
        )
        return data_dict


class _FixedMirror:
    def get_parameters(self, **data_dict):
        return {"axes": [0]}

    def apply(self, data_dict, **params):
        data_dict["image"] = torch.flip(data_dict["image"], (1,))
        data_dict["segmentation"] = torch.flip(data_dict["segmentation"], (1,))
        return data_dict


class _RegionLabelManager:
    has_regions = True
    has_ignore_label = False
    all_labels = [0, 1, 2, 3]
    foreground_regions = [(1, 2, 3), (1, 3), 3]


def test_one_rotation_and_mirror_are_shared_by_all_spatial_objects() -> None:
    image = torch.zeros(1, 5, 5, 5)
    class_segmentation = torch.zeros(1, 5, 5, 5, dtype=torch.int32)
    instance_ids = torch.zeros(1, 5, 5, 5, dtype=torch.int32)
    source_center = (1, 1, 2)
    image[(0, *source_center)] = 9.0
    class_segmentation[(0, *source_center)] = 3
    instance_ids[(0, *source_center)] = 17

    transformed = apply_synchronized_spatial(
        image=image,
        class_segmentation=class_segmentation,
        instance_ids=instance_ids,
        centers_zyx=np.asarray([source_center], dtype=np.float64),
        spatial_transform=_FixedRotation(),
        mirror_transform=_FixedMirror(),
    )

    transformed_image, transformed_class, transformed_ids, transformed_centers = transformed
    expected_center = (1, 1, 2)
    assert tuple(torch.nonzero(transformed_image[0] == 9.0)[0].tolist()) == expected_center
    assert tuple(torch.nonzero(transformed_class[0] == 3)[0].tolist()) == expected_center
    assert tuple(torch.nonzero(transformed_ids[0] == 17)[0].tolist()) == expected_center
    np.testing.assert_allclose(transformed_centers[0], expected_center)
    assert transformed_ids.dtype == torch.int32


def test_fragment_keeps_region_supervision_but_has_no_pf_supervision() -> None:
    class_segmentation = torch.zeros(1, 5, 5, 5, dtype=torch.int32)
    class_segmentation[0, 2, 2, 2:4] = 1
    instance_ids = torch.zeros(5, 5, 5, dtype=torch.int32)
    instance_ids[2, 2, 2:4] = 7

    region_target, persistent = build_region_and_persistent_targets(
        class_segmentation,
        instance_ids,
        centers_by_id={7: (-1.0, 2.0, 2.0)},
        spacing_zyx=(1.0, 1.0, 1.0),
        label_manager=_RegionLabelManager(),
    )

    assert region_target[0, 2, 2, 2]
    assert persistent.valid_instance_ids == ()
    assert not persistent.dense_valid_mask[instance_ids == 7].any()
    assert torch.all(persistent.field[instance_ids == 7] == -1.0)


def test_negative_patch_has_all_floor_field() -> None:
    region_target, persistent = build_region_and_persistent_targets(
        torch.zeros(1, 4, 4, 4, dtype=torch.int32),
        torch.zeros(4, 4, 4, dtype=torch.int32),
        centers_by_id={},
        spacing_zyx=(1.0, 1.0, 1.0),
        label_manager=_RegionLabelManager(),
    )
    assert not region_target.any()
    assert torch.all(persistent.field == -1.0)
    assert persistent.dense_valid_mask.all()


def test_a3_samples_global_instances_uniformly_and_a1_a2_keep_cases() -> None:
    global_instances = [("case_a", 1), ("case_a", 2), ("case_b", 9), ("case_c", 4)]
    rng = np.random.default_rng(42)
    counts = {item: 0 for item in global_instances}
    for _ in range(12000):
        keys, instance_ids = select_batch_sources(
            selected_keys=["standard"],
            oversampled_slots=[True],
            global_instances=global_instances,
            component_uniform=True,
            rng=rng,
        )
        counts[(keys[0], instance_ids[0])] += 1
    expected = 12000 / len(global_instances)
    assert all(abs(count - expected) / expected < 0.06 for count in counts.values())

    keys, instance_ids = select_batch_sources(
        selected_keys=["case_a", "case_b"],
        oversampled_slots=[False, True],
        global_instances=global_instances,
        component_uniform=False,
        rng=np.random.default_rng(0),
    )
    assert keys == ["case_a", "case_b"]
    assert instance_ids == [None, None]


class _MemoryDataset:
    identifiers = ["negative"]

    def load_case(self, identifier: str):
        data = np.zeros((2, 4, 4, 4), dtype=np.float32)
        segmentation = np.zeros((1, 4, 4, 4), dtype=np.int16)
        properties = {"class_locations": {1: np.empty((0, 4), dtype=int)}}
        return data, segmentation, None, properties


class _TopologyWorkerLoader(SlimDataLoaderBase):
    def __init__(self) -> None:
        super().__init__(data=None, batch_size=1)

    def generate_train_batch(self) -> dict:
        instance_ids = np.zeros((5, 5, 5), dtype=np.int32)
        instance_ids[1:3, 1:3, 1:3] = 1
        target = build_persistent_field_target(
            instance_ids,
            {1: (1, 1, 1)},
            (1.0, 1.0, 1.0),
        )
        return {"persistent_targets": (target,)}


def test_batch_contract_has_no_deep_supervision_and_never_reads_cached_field(
    tmp_path: Path, monkeypatch
) -> None:
    case_cache = tmp_path / "negative"
    case_cache.mkdir()
    np.savez_compressed(
        case_cache / "instances.npz",
        instance_ids=np.zeros((4, 4, 4), dtype=np.int32),
    )
    (case_cache / "metadata.json").write_text(
        json.dumps({"spacing_zyx": [1, 1, 1], "instances": []}), encoding="utf-8"
    )
    np.save(case_cache / "field.npy", np.full((4, 4, 4), 123, dtype=np.float32))
    loaded_paths: list[Path] = []
    original_load = np.load

    def recording_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    loader = MicroBTPFDataLoader(
        _MemoryDataset(),
        batch_size=1,
        patch_size=(4, 4, 4),
        final_patch_size=(4, 4, 4),
        label_manager=_RegionLabelManager(),
        metadata_cache_directory=tmp_path,
        component_uniform=False,
        oversample_foreground_percent=0.33,
    )

    batch = loader.generate_train_batch()

    assert set(batch) == {"data", "target", "persistent_targets", "keys"}
    assert batch["data"].shape == (1, 2, 4, 4, 4)
    assert batch["target"].shape == (1, 3, 4, 4, 4)
    assert isinstance(batch["target"], torch.Tensor)
    assert len(batch["persistent_targets"]) == 1
    assert batch["persistent_targets"][0].topology is not None
    assert not batch["persistent_targets"][0].topology.branch_valid.flags.writeable
    assert loader.oversample_foreground_percent == 0.33
    assert all(path.name != "field.npy" for path in loaded_paths)


def test_validation_loader_can_emit_dense_target_without_topology(tmp_path: Path) -> None:
    case_cache = tmp_path / "negative"
    case_cache.mkdir()
    np.savez_compressed(
        case_cache / "instances.npz",
        instance_ids=np.zeros((4, 4, 4), dtype=np.int32),
    )
    (case_cache / "metadata.json").write_text(
        json.dumps({"spacing_zyx": [1, 1, 1], "instances": []}),
        encoding="utf-8",
    )
    loader = MicroBTPFDataLoader(
        _MemoryDataset(),
        batch_size=1,
        patch_size=(4, 4, 4),
        final_patch_size=(4, 4, 4),
        label_manager=_RegionLabelManager(),
        metadata_cache_directory=tmp_path,
        component_uniform=False,
        precompute_topology=False,
    )

    batch = loader.generate_train_batch()

    assert batch["persistent_targets"][0].topology is None
    assert torch.all(batch["persistent_targets"][0].field == -1.0)


def test_target_topology_survives_real_two_worker_prefetch() -> None:
    previous_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")
    augmenter = NonDetMultiThreadedAugmenter(
        data_loader=_TopologyWorkerLoader(),
        transform=None,
        num_processes=2,
        num_cached=6,
        seeds=[42, 42],
        pin_memory=False,
        wait_time=0.01,
    )
    try:
        batches = [next(augmenter), next(augmenter)]
    finally:
        if augmenter.initialized:
            augmenter.abort_event.set()
            augmenter.results_loop_thread.join(timeout=5.0)
            assert not augmenter.results_loop_thread.is_alive()
        augmenter._finish()
        torch.multiprocessing.set_sharing_strategy(previous_sharing_strategy)

    for batch in batches:
        target = batch["persistent_targets"][0]
        reference = persistence_pairs_0d(target.field)
        np.testing.assert_array_equal(
            target.topology.pairs.birth_flat, reference.birth_flat
        )
        np.testing.assert_array_equal(
            target.topology.pairs.death_flat, reference.death_flat
        )
        assert not target.topology.pairs.birth_flat.flags.writeable
        assert not target.topology.branch_valid.flags.writeable
