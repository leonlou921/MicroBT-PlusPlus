from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from nnunetv2.inference.microbt_pf_export import restore_microbt_pf_prediction
from nnunetv2.topology.persistent_field_0d import (
    decode_persistent_field,
    persistence_pairs_0d,
)
from nnunetv2.training.dataloading.microbt_pf_loader import MicroBTPFDataLoader
from nnunetv2.training.dataloading.microbt_pf_targets import (
    build_persistent_field_target,
    label_instances_26,
)
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.MicroBTPFTrainer import (
    MicroBTPFTrainerA3ExactFast,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from scripts.benchmark_microbt_pf_topology import benchmark_one_shape


DATASET_NAME = "Dataset001_Example"


def _dataset_directory() -> Path:
    return Path(os.environ["nnUNet_preprocessed"]) / DATASET_NAME


def _cache_directory() -> Path:
    return (
        Path(os.environ["MICROBT_PF_CACHE"])
        / DATASET_NAME
        / "nnUNetResEncUNetMPlans"
        / "3d_fullres"
    )


def test_synthetic_five_case_target_and_decode_contract() -> None:
    shape = (7, 7, 7)
    negative = np.zeros(shape, dtype=bool)
    positive = np.zeros(shape, dtype=bool)
    positive[2:5, 2:5, 2:5] = True
    one_voxel = np.zeros(shape, dtype=bool)
    one_voxel[3, 3, 3] = True
    diagonal = np.zeros(shape, dtype=bool)
    diagonal[2, 2, 2] = True
    diagonal[3, 3, 3] = True
    two_lesion = np.zeros(shape, dtype=bool)
    two_lesion[2, 2, 2] = True
    two_lesion[4, 4, 4] = True
    cases = [negative, positive, one_voxel, diagonal, two_lesion]
    expected_counts = [0, 1, 1, 1, 2]

    observed_counts = []
    for mask in cases:
        instances = label_instances_26(mask)
        centers = {
            int(instance_id): tuple(int(value) for value in np.argwhere(instances == instance_id)[0])
            for instance_id in np.unique(instances)
            if instance_id > 0
        }
        target = build_persistent_field_target(
            instances,
            centers,
            spacing_zyx=(1.0, 1.0, 1.0),
            topology_margin=1,
        )
        observed_counts.append(
            decode_persistent_field(target.field.numpy())["count"]
        )
    assert observed_counts == expected_counts


def test_real_dataset001_loader_produces_two_independent_batches() -> None:
    dataset_directory = _dataset_directory()
    source = dataset_directory / "nnUNetPlans_3d_fullres"
    cache = _cache_directory()
    identifiers = sorted(path.name for path in cache.iterdir() if path.is_dir())[:5]
    dataset_class = infer_dataset_class(str(source))
    dataset = dataset_class(str(source), identifiers)
    plans = json.loads(
        (dataset_directory / "nnUNetResEncUNetMPlans.json").read_text(encoding="utf-8")
    )
    dataset_json = json.loads(
        (dataset_directory / "dataset.json").read_text(encoding="utf-8")
    )
    label_manager = PlansManager(plans).get_label_manager(dataset_json)
    loader = MicroBTPFDataLoader(
        dataset,
        batch_size=1,
        patch_size=(112, 160, 128),
        final_patch_size=(112, 160, 128),
        label_manager=label_manager,
        metadata_cache_directory=cache,
        component_uniform=True,
        oversample_foreground_percent=1.0,
    )

    first = loader.generate_train_batch()
    second = loader.generate_train_batch()

    for batch in (first, second):
        assert tuple(batch["data"].shape) == (1, 4, 112, 160, 128)
        assert tuple(batch["target"].shape) == (1, 3, 112, 160, 128)
        assert len(batch["persistent_targets"]) == 1
        persistent = batch["persistent_targets"][0]
        assert torch.isfinite(persistent.field).all()
        reference = persistence_pairs_0d(persistent.field)
        np.testing.assert_array_equal(
            persistent.topology.pairs.birth_flat, reference.birth_flat
        )
        np.testing.assert_array_equal(
            persistent.topology.pairs.death_flat, reference.death_flat
        )
        assert not persistent.topology.branch_valid.flags.writeable


def _tiny_plans(plans: dict) -> dict:
    plans = deepcopy(plans)
    configuration = plans["configurations"]["3d_fullres"]
    configuration["patch_size"] = [8, 8, 8]
    configuration["batch_size"] = 1
    architecture = configuration["architecture"]["arch_kwargs"]
    architecture["n_stages"] = 2
    architecture["features_per_stage"] = [4, 8]
    architecture["kernel_sizes"] = [[3, 3, 3], [3, 3, 3]]
    architecture["strides"] = [[1, 1, 1], [2, 2, 2]]
    architecture["n_blocks_per_stage"] = [1, 1]
    architecture["n_conv_per_stage_decoder"] = [1]
    return plans


def test_reduced_plan_exact_fast_two_steps_and_checkpoint_roundtrip(monkeypatch) -> None:
    dataset_directory = _dataset_directory()
    plans = _tiny_plans(
        json.loads(
            (dataset_directory / "nnUNetResEncUNetMPlans.json").read_text(encoding="utf-8")
        )
    )
    dataset_json = json.loads(
        (dataset_directory / "dataset.json").read_text(encoding="utf-8")
    )
    isolated = Path(os.environ["MICROBT_PF_TEST_ISOLATED_ROOT"])
    with tempfile.TemporaryDirectory(prefix="pipeline_", dir=isolated) as directory:
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module

        monkeypatch.setattr(trainer_module, "nnUNet_results", str(Path(directory) / "results"))
        os.environ["nnUNet_compile"] = "false"
        torch.manual_seed(11)
        trainer = MicroBTPFTrainerA3ExactFast(
            plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu")
        )
        trainer.initialize()
        instances = np.zeros((8, 8, 8), dtype=np.int32)
        instances[3:5, 3:5, 3:5] = 1
        persistent = build_persistent_field_target(
            instances,
            {1: (3.5, 3.5, 3.5)},
            (1.0, 1.0, 1.0),
            topology_margin=1,
        )
        region_target = torch.zeros((1, 3, 8, 8, 8))
        region_target[:, :, 3:5, 3:5, 3:5] = 1.0
        data = torch.linspace(-1.0, 1.0, 4 * 8 * 8 * 8).reshape(1, 4, 8, 8, 8)
        batch = {
            "data": data,
            "target": region_target,
            "persistent_targets": (persistent,),
        }
        first = trainer.train_step(batch)
        second = trainer.train_step(batch)
        assert np.isfinite(first["loss"])
        assert np.isfinite(second["loss"])
        assert trainer.exact_fast_telemetry["exact_h0_train_calls"] == 2
        trainer.network.eval()
        with torch.no_grad():
            expected = trainer.network(data)
        checkpoint = Path(directory) / "checkpoint.pth"
        trainer.save_checkpoint(str(checkpoint))
        restored = MicroBTPFTrainerA3ExactFast(
            plans, "3d_fullres", 0, dataset_json, device=torch.device("cpu")
        )
        restored.load_checkpoint(str(checkpoint))
        restored.network.eval()
        with torch.no_grad():
            observed = restored.network(data)
        assert all(torch.equal(observed[key], expected[key]) for key in expected)


class _IdentityConfiguration:
    spacing = (1.0, 1.0, 1.0)

    @staticmethod
    def resampling_fn_probabilities(array, shape, current_spacing, target_spacing):
        return array


class _Plans:
    transpose_forward = (0, 1, 2)
    transpose_backward = (0, 1, 2)


def test_full_volume_fusion_decode_and_export_schema() -> None:
    first_window = torch.full((3, 1, 1, 4), -4.0)
    second_window = torch.full((3, 1, 1, 4), -4.0)
    first_window[:, 0, 0, 3] = 4.0
    second_window[:, 0, 0, 0] = 4.0
    accumulated = torch.zeros((3, 1, 1, 7))
    counts = torch.zeros((1, 1, 7))
    accumulated[:, :, :, :4] += first_window
    accumulated[:, :, :, 3:] += second_window
    counts[:, :, :4] += 1
    counts[:, :, 3:] += 1
    raw = accumulated / counts
    properties = {
        "spacing": (1.0, 1.0, 1.0),
        "shape_after_cropping_and_before_resampling": (1, 1, 7),
        "shape_before_cropping": (3, 3, 9),
        "bbox_used_for_cropping": ((1, 2), (1, 2), (1, 8)),
    }
    restored = restore_microbt_pf_prediction(
        raw, properties, _IdentityConfiguration(), _Plans(), a0=False
    )
    assert restored["count"] == 1
    assert restored["instance_ids"].dtype == np.int32
    assert restored["field"].dtype == np.float32
    assert restored["objects"][0]["volume"] == int(
        np.count_nonzero(restored["instance_ids"] == 1)
    )
    assert set(restored) == {
        "raw_logits",
        "region_probabilities",
        "segmentation",
        "field",
        "instance_ids",
        "objects",
        "count",
    }


def test_topology_benchmark_reports_deterministic_hash_and_memory() -> None:
    result = benchmark_one_shape((8, 8, 8), repeats=2)
    assert result["shape"] == [8, 8, 8]
    assert result["deterministic"] is True
    assert len(result["hashes"]) == 2
    assert result["hashes"][0] == result["hashes"][1]
    assert result["peak_rss_bytes"] > 0
    assert result["pair_count"] >= 1
