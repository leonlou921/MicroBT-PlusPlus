from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from nnunetv2.inference.microbt_pf_export import (
    export_microbt_pf_prediction,
    restore_microbt_pf_prediction,
)
from nnunetv2.inference.microbt_pf_predictor import (
    MicroBTPFPredictor,
    raw_logits_to_microbt_pf,
)


class _IdentityConfiguration:
    spacing = (1.0, 1.0, 1.0)

    @staticmethod
    def resampling_fn_probabilities(array, shape, current_spacing, target_spacing):
        assert tuple(array.shape[1:]) == tuple(shape)
        return array


class _ArrayWriter:
    @staticmethod
    def write_seg(array, filename, properties):
        with Path(filename).open("wb") as handle:
            np.save(handle, np.asarray(array))


class _Plans:
    transpose_forward = (0, 1, 2)
    transpose_backward = (0, 1, 2)

    @staticmethod
    def image_reader_writer_class():
        return _ArrayWriter()


def _properties() -> dict:
    return {
        "spacing": (1.0, 1.0, 1.0),
        "shape_after_cropping_and_before_resampling": (2, 2, 3),
        "shape_before_cropping": (4, 4, 5),
        "bbox_used_for_cropping": ((1, 3), (1, 3), (1, 4)),
    }


def _raw_single_lesion() -> torch.Tensor:
    raw = torch.full((3, 2, 2, 3), -4.0)
    raw[:, 0, 0, 1] = 4.0
    return raw


def test_fold_averaging_is_raw_logit_averaging_before_nonlinearity() -> None:
    class BiasNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(3))

    predictor = object.__new__(MicroBTPFPredictor)
    predictor.network = BiasNetwork()
    predictor.list_of_parameters = [
        {"bias": torch.tensor([4.0, 2.0, -2.0])},
        {"bias": torch.tensor([-2.0, -4.0, 4.0])},
    ]
    predictor.verbose = False

    def fake_sliding_window(self, data):
        return self.network.bias[:, None, None, None].clone()

    predictor.predict_sliding_window_return_logits = MethodType(
        fake_sliding_window, predictor
    )
    raw = predictor.predict_logits_from_preprocessed_data(torch.zeros(1, 1, 1, 1))
    converted = raw_logits_to_microbt_pf(raw, a0=False)

    torch.testing.assert_close(raw[:, 0, 0, 0], torch.tensor([1.0, -1.0, 1.0]))
    torch.testing.assert_close(converted["field"], torch.tanh(torch.tensor([[[1.0]]])))
    assert not torch.allclose(
        converted["field"],
        (torch.tanh(torch.tensor(4.0)) + torch.tanh(torch.tensor(-2.0))) / 2.0,
    )


def test_mirror_prediction_unwraps_only_field_logits() -> None:
    class DictNetwork(nn.Module):
        def forward(self, image):
            return {"field_logits": image[:, :3], "forbidden": image[:, :1] + 99}

    predictor = object.__new__(MicroBTPFPredictor)
    predictor.network = DictNetwork()
    predictor.use_mirroring = True
    predictor.allowed_mirroring_axes = (2,)
    image = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 1, 2)

    observed = predictor._internal_maybe_mirror_and_predict(image)

    assert isinstance(observed, torch.Tensor)
    torch.testing.assert_close(observed, image)


def test_overlapping_windows_are_fused_before_one_global_decode() -> None:
    class IdentityDictNetwork(nn.Module):
        def forward(self, image):
            return {"field_logits": image}

    predictor = object.__new__(MicroBTPFPredictor)
    predictor.network = IdentityDictNetwork()
    predictor.device = torch.device("cpu")
    predictor.use_mirroring = False
    predictor.allowed_mirroring_axes = None
    predictor.use_gaussian = False
    predictor.verbose = False
    predictor.allow_tqdm = False
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=3)
    raw_volume = torch.full((3, 1, 1, 5), -4.0)
    raw_volume[:, 0, 0, 2] = 4.0
    slicers = [
        (slice(None), slice(None), slice(None), slice(0, 3)),
        (slice(None), slice(None), slice(None), slice(2, 5)),
    ]

    fused = predictor._internal_predict_sliding_window_return_logits(
        raw_volume, slicers, do_on_device=False
    )
    decoded = raw_logits_to_microbt_pf(fused.float(), a0=False)
    from nnunetv2.topology.persistent_field_0d import decode_persistent_field

    objects = decode_persistent_field(decoded["field"].numpy())
    assert objects["count"] == 1
    assert objects["instance_ids"].tolist() == [[[0, 0, 1, 0, 0]]]


def test_a0_decoder_uses_exact_wt_conversion_and_independent_regions() -> None:
    raw = torch.tensor([0.0, -2.0, 2.0])[:, None, None, None]
    converted = raw_logits_to_microbt_pf(raw, a0=True)
    torch.testing.assert_close(converted["field"], torch.zeros((1, 1, 1)))
    torch.testing.assert_close(
        converted["region_probabilities"][:, 0, 0, 0],
        torch.sigmoid(torch.tensor([0.0, -2.0, 2.0])),
    )


def test_restore_reinserts_crop_and_preserves_semantic_hierarchy() -> None:
    restored = restore_microbt_pf_prediction(
        _raw_single_lesion(),
        _properties(),
        _IdentityConfiguration(),
        _Plans(),
        a0=False,
    )
    assert restored["segmentation"].shape == (4, 4, 5)
    assert restored["field"].shape == (4, 4, 5)
    assert restored["instance_ids"].shape == (4, 4, 5)
    assert restored["segmentation"][1, 1, 2] == 3
    assert np.count_nonzero(restored["segmentation"] == 3) == 1
    assert np.all(restored["field"][:1] == -1.0)
    assert restored["instance_ids"][1, 1, 2] == 1
    assert restored["objects"][0]["centroid_zyx"] == [1, 1, 2]
    assert restored["objects"][0]["box_zyx"] == [1, 1, 2, 2, 2, 3]


def test_export_sidecars_match_instance_volume_and_store_raw_logits(
    tmp_path: Path,
) -> None:
    output = tmp_path / "case"
    restored = export_microbt_pf_prediction(
        _raw_single_lesion(),
        _properties(),
        _IdentityConfiguration(),
        _Plans(),
        {"file_ending": ".nii.gz"},
        str(output),
        save_field=True,
        a0=False,
    )

    segmentation = np.load(str(output) + ".nii.gz")
    instances = np.load(str(output) + "_instances.nii.gz")
    objects = json.loads((tmp_path / "case_objects.json").read_text(encoding="utf-8"))
    with np.load(tmp_path / "case_field.npz") as archive:
        assert set(archive.files) == {"raw_logits", "field"}
        assert archive["raw_logits"].shape == (3, 2, 2, 3)
        np.testing.assert_array_equal(archive["field"], restored["field"])
    np.testing.assert_array_equal(segmentation, restored["segmentation"])
    np.testing.assert_array_equal(instances, restored["instance_ids"])
    assert objects["count"] == 1
    assert objects["objects"][0]["volume"] == int(np.count_nonzero(instances == 1))
    assert objects["objects"][0]["centroid_zyx"] == [1, 1, 2]
