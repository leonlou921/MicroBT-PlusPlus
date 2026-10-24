from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import label, watershed_ift

try:
    from nnunetv2 import _persistent_field_0d as _backend
except (ImportError, OSError):
    _backend = None

try:
    from nnunetv2 import _persistent_field_0d_cuda as _cuda_backend
except (ImportError, OSError):
    _cuda_backend = None


@dataclass(frozen=True, eq=False)
class PersistencePairs0D:
    birth_flat: np.ndarray
    death_flat: np.ndarray
    essential: np.ndarray


@dataclass(frozen=True, eq=False)
class InducedMatching0D:
    prediction: PersistencePairs0D
    target: PersistencePairs0D
    matched_prediction: np.ndarray
    matched_target: np.ndarray
    unmatched_prediction: np.ndarray
    unmatched_target: np.ndarray


def _require_backend():
    if _backend is None:
        raise RuntimeError("MicroBT-PF H0 backend is unavailable")
    return _backend


def exact_backend_info() -> dict[str, str | bool | None]:
    cuda_build_id = None
    if _cuda_backend is not None:
        cuda_build_id = str(_cuda_backend.build_id())
        if len(cuda_build_id) != 64:
            raise RuntimeError("MicroBT-PF CUDA H0 backend returned an invalid build id")
    return {
        "cpu_available": _backend is not None,
        "cuda_available": _cuda_backend is not None,
        "cuda_build_id": cuda_build_id,
    }


def require_cuda_exact_backend():
    if _cuda_backend is None:
        raise RuntimeError(
            "MicroBT-PF exact CUDA H0 backend is required but unavailable; "
            "CPU fallback is forbidden for an admitted CUDA run"
        )
    exact_backend_info()
    return _cuda_backend


def _validated_cuda_field(field: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(field, torch.Tensor) or not field.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if field.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if field.ndim != 3 or field.numel() == 0:
        raise ValueError(f"{name} must be a nonempty three-dimensional field")
    if not field.is_contiguous():
        raise ValueError(f"{name} must be C-contiguous")
    if not bool(torch.isfinite(field).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return field


def canonical_order_cuda(field: torch.Tensor) -> torch.Tensor:
    require_cuda_exact_backend()
    field = _validated_cuda_field(field, "field")
    return torch.argsort(field.reshape(-1), descending=True, stable=True)


def persistence_pairs_0d_cuda(
    field: torch.Tensor, floor: float = -1.0
) -> PersistencePairs0D:
    if not np.isfinite(floor):
        raise ValueError("floor must be finite")
    backend = require_cuda_exact_backend()
    field = _validated_cuda_field(field, "field")
    order = canonical_order_cuda(field)
    tensors = backend.persistence_pairs(field, order)
    if not isinstance(tensors, (tuple, list)) or len(tensors) != 4:
        raise RuntimeError("MicroBT-PF CUDA H0 backend returned an invalid result")
    birth_tensor, death_tensor, essential_tensor, count_tensor = tensors
    count = int(count_tensor.item())
    if count <= 0 or count > field.numel():
        raise RuntimeError("MicroBT-PF CUDA H0 backend returned an invalid pair count")
    raw = {
        "birth_flat": birth_tensor[:count]
        .to(device="cpu", dtype=torch.int64)
        .numpy()
        .astype(np.uint32, copy=False),
        "death_flat": death_tensor[:count].to(device="cpu").numpy(),
        "essential": essential_tensor[:count]
        .to(device="cpu")
        .numpy()
        .astype(bool, copy=False),
    }
    return _validated_pairs(raw, field.numel())


def _discrete_field(field: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(field, torch.Tensor):
        array = field.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(field), dtype=np.float32)
    if array.ndim != 3 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty three-dimensional field")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _immutable_array(value, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.ndim != 1:
        raise RuntimeError(f"H0 backend returned invalid {name}")
    array = np.ascontiguousarray(array).copy()
    array.flags.writeable = False
    return array


def _validated_pairs(
    raw: dict[str, np.ndarray], size: int, prefix: str = ""
) -> PersistencePairs0D:
    key = lambda suffix: f"{prefix}_{suffix}" if prefix else suffix
    birth = _immutable_array(raw[key("birth_flat")], np.dtype(np.uint32), key("birth_flat"))
    death = _immutable_array(raw[key("death_flat")], np.dtype(np.int64), key("death_flat"))
    essential = _immutable_array(raw[key("essential")], np.dtype(bool), key("essential"))
    if not (birth.size == death.size == essential.size):
        raise RuntimeError("H0 backend returned pair arrays with unequal lengths")
    if birth.size == 0 or np.any(birth.astype(np.uint64) >= size):
        raise RuntimeError("H0 backend returned an invalid birth index")
    if np.count_nonzero(essential) != 1:
        raise RuntimeError("H0 backend must return exactly one essential branch")
    if np.any(death[essential] != -1) or np.any(death[~essential] < 0):
        raise RuntimeError("H0 backend returned an invalid essential death marker")
    if np.any(death[~essential] >= size):
        raise RuntimeError("H0 backend returned an invalid death index")
    return PersistencePairs0D(birth, death, essential)


def persistence_pairs_0d(
    field: np.ndarray | torch.Tensor, floor: float = -1.0
) -> PersistencePairs0D:
    array = _discrete_field(field, "field")
    raw = _require_backend().persistence_pairs(array, float(floor))
    return _validated_pairs(raw, array.size)


def _matching_indices(
    raw: dict[str, np.ndarray], key: str, pair_count: int
) -> np.ndarray:
    indices = _immutable_array(raw[key], np.dtype(np.int64), key)
    if np.any(indices < 0) or np.any(indices >= pair_count):
        raise RuntimeError(f"H0 backend returned invalid {key}")
    if np.unique(indices).size != indices.size:
        raise RuntimeError(f"H0 backend returned duplicate {key}")
    return indices


def _validated_matching(raw: dict, size: int) -> InducedMatching0D:
    prediction_pairs = _validated_pairs(raw, size, "prediction")
    target_pairs = _validated_pairs(raw, size, "target")
    matched_prediction = _matching_indices(
        raw, "matched_prediction", prediction_pairs.birth_flat.size
    )
    matched_target = _matching_indices(raw, "matched_target", target_pairs.birth_flat.size)
    if matched_prediction.size != matched_target.size:
        raise RuntimeError("H0 backend returned unequal matched index lengths")
    unmatched_prediction = _matching_indices(
        raw, "unmatched_prediction", prediction_pairs.birth_flat.size
    )
    unmatched_target = _matching_indices(
        raw, "unmatched_target", target_pairs.birth_flat.size
    )
    if set(matched_prediction) & set(unmatched_prediction):
        raise RuntimeError("H0 backend prediction matching is not a partition")
    if set(matched_target) & set(unmatched_target):
        raise RuntimeError("H0 backend target matching is not a partition")
    if matched_prediction.size + unmatched_prediction.size != prediction_pairs.birth_flat.size:
        raise RuntimeError("H0 backend prediction matching is incomplete")
    if matched_target.size + unmatched_target.size != target_pairs.birth_flat.size:
        raise RuntimeError("H0 backend target matching is incomplete")
    return InducedMatching0D(
        prediction=prediction_pairs,
        target=target_pairs,
        matched_prediction=matched_prediction,
        matched_target=matched_target,
        unmatched_prediction=unmatched_prediction,
        unmatched_target=unmatched_target,
    )


def _matching_inputs(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = _discrete_field(prediction, "prediction")
    target_array = _discrete_field(target, "target")
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must have equal shapes")
    return prediction_array, target_array


def induced_matching_0d(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    floor: float = -1.0,
) -> InducedMatching0D:
    prediction_array, target_array = _matching_inputs(prediction, target)
    raw = _require_backend().induced_matching(
        prediction_array, target_array, float(floor)
    )
    return _validated_matching(raw, prediction_array.size)


def induced_matching_0d_precomputed_target(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    target_pairs: PersistencePairs0D,
    floor: float = -1.0,
) -> InducedMatching0D:
    prediction_array, target_array = _matching_inputs(prediction, target)
    if not isinstance(target_pairs, PersistencePairs0D):
        raise TypeError("target_pairs must be PersistencePairs0D")
    raw = _require_backend().induced_matching_precomputed_target(
        prediction_array,
        target_array,
        target_pairs.birth_flat,
        target_pairs.death_flat,
        target_pairs.essential,
        float(floor),
    )
    return _validated_matching(raw, prediction_array.size)


def profile_induced_matching_0d(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    floor: float = -1.0,
) -> tuple[InducedMatching0D, dict[str, float]]:
    prediction_array, target_array = _matching_inputs(prediction, target)
    raw = _require_backend().induced_matching_profiled(
        prediction_array, target_array, float(floor)
    )
    matching = _validated_matching(raw, prediction_array.size)
    profile = raw.get("profile_seconds")
    expected = {
        "prediction_pairs",
        "target_pairs",
        "comparison_matching",
        "total",
    }
    if not isinstance(profile, dict) or set(profile) != expected:
        raise RuntimeError("H0 backend returned an invalid timing profile")
    profile_seconds = {key: float(profile[key]) for key in expected}
    if any(not np.isfinite(value) or value < 0 for value in profile_seconds.values()):
        raise RuntimeError("H0 backend returned an invalid timing value")
    if any(
        profile_seconds["total"] < profile_seconds[key]
        for key in expected - {"total"}
    ):
        raise RuntimeError("H0 backend returned an inconsistent timing profile")
    return matching, profile_seconds


def critical_values(
    field: torch.Tensor, pairs: PersistencePairs0D, floor: float = -1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = field.reshape(-1)
    birth_indices = torch.as_tensor(
        pairs.birth_flat.astype(np.int64), device=field.device, dtype=torch.long
    )
    birth = flat.index_select(0, birth_indices)
    death = torch.full(
        (pairs.death_flat.size,), float(floor), device=field.device, dtype=field.dtype
    )
    finite = pairs.death_flat >= 0
    if np.any(finite):
        finite_positions = torch.as_tensor(
            np.flatnonzero(finite), device=field.device, dtype=torch.long
        )
        death_indices = torch.as_tensor(
            pairs.death_flat[finite], device=field.device, dtype=torch.long
        )
        death = death.index_copy(0, finite_positions, flat.index_select(0, death_indices))
    return birth, death


def decode_persistent_field(
    field: np.ndarray,
    field_min: float = 0.25,
    foreground_level: float = 0.0,
) -> dict:
    array = _discrete_field(field, "field")
    pairs = persistence_pairs_0d(array, floor=-1.0)
    flat = array.ravel()
    birth_values = flat[pairs.birth_flat.astype(np.int64)]
    death_values = np.full(pairs.death_flat.size, -1.0, dtype=np.float32)
    finite = pairs.death_flat >= 0
    death_values[finite] = flat[pairs.death_flat[finite]]
    retained = np.flatnonzero(
        (birth_values >= float(field_min)) & (death_values < float(foreground_level))
    )
    retained = retained[np.argsort(pairs.birth_flat[retained], kind="stable")]

    foreground = array >= float(foreground_level)
    zero_components, _ = label(
        foreground, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    instance_ids = np.zeros(array.shape, dtype=np.int32)
    masks: list[np.ndarray] = []
    objects: list[dict] = []
    used_components: set[int] = set()
    for output_id, pair_index in enumerate(retained, start=1):
        birth_flat = int(pairs.birth_flat[pair_index])
        component_id = int(zero_components.ravel()[birth_flat])
        if component_id == 0:
            raise RuntimeError("retained branch birth is outside the zero-level foreground")
        if component_id in used_components:
            raise RuntimeError("retained branches share one zero-level component")
        used_components.add(component_id)
        mask = zero_components == component_id
        instance_ids[mask] = output_id
        masks.append(mask)
        maximum = float(array[mask].max())
        centroid = np.argwhere(mask & (array == maximum))[0]
        coordinates = np.argwhere(mask)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        objects.append(
            {
                "instance_id": output_id,
                "birth_flat": birth_flat,
                "birth": float(birth_values[pair_index]),
                "death": float(death_values[pair_index]),
                "centroid_zyx": [int(value) for value in centroid],
                "box_zyx": [
                    int(lower[0]),
                    int(lower[1]),
                    int(lower[2]),
                    int(upper[0]),
                    int(upper[1]),
                    int(upper[2]),
                ],
                "volume": int(mask.sum()),
            }
        )
    return {
        "count": len(objects),
        "instance_ids": instance_ids,
        "masks": tuple(masks),
        "objects": tuple(objects),
    }


def decode_support_birth_field(
    birth_field: np.ndarray,
    whole_tumor_probability: np.ndarray,
    field_min: float = 0.25,
    persistence_min: float = 0.25,
    support_threshold: float = 0.5,
) -> dict:
    """Decode SB-PF markers inside semantic support with deterministic watershed."""
    birth = _discrete_field(birth_field, "birth_field")
    support_probability = np.asarray(whole_tumor_probability, dtype=np.float32)
    if support_probability.shape != birth.shape:
        raise ValueError("birth field and whole-tumor probability must have equal shapes")
    if not np.isfinite(support_probability).all():
        raise ValueError("whole-tumor probability must be finite")

    pairs = persistence_pairs_0d(birth, floor=-1.0)
    flat = birth.ravel()
    birth_values = flat[pairs.birth_flat.astype(np.int64)]
    death_values = np.full(pairs.death_flat.size, -1.0, dtype=np.float32)
    finite = pairs.death_flat >= 0
    death_values[finite] = flat[pairs.death_flat[finite]]
    persistence = birth_values - death_values
    retained = np.flatnonzero(
        (birth_values >= float(field_min))
        & (persistence >= float(persistence_min))
    )
    retained = retained[np.argsort(pairs.birth_flat[retained], kind="stable")]

    support = support_probability >= float(support_threshold)
    retained_birth_flat = pairs.birth_flat[retained].astype(np.int64, copy=False)
    marker_outside_support = int((~support.ravel()[retained_birth_flat]).sum())
    support = support.copy()
    support.ravel()[retained_birth_flat] = True

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    support_components, component_count = label(support, structure=structure)
    markers = np.zeros(birth.shape, dtype=np.int32)
    for output_id, pair_index in enumerate(retained, start=1):
        markers.ravel()[int(pairs.birth_flat[pair_index])] = output_id

    # watershed_ift minimizes its uint gradient. High birth values must therefore
    # be low on the flooding surface; 16-bit quantization is deterministic.
    surface = np.rint((1.0 - np.clip(birth, -1.0, 1.0)) * 32767.0).astype(
        np.uint16
    )
    instance_ids = np.zeros(birth.shape, dtype=np.int32)
    for component_id in range(1, int(component_count) + 1):
        component = support_components == component_id
        if not np.any(markers[component] > 0):
            continue
        coordinates = np.argwhere(component)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        local_component = component[crop]
        local_surface = surface[crop].copy()
        local_surface[~local_component] = np.iinfo(np.uint16).max
        local_markers = markers[crop].copy()
        local_labels = watershed_ift(
            local_surface,
            local_markers,
            structure=structure,
        )
        destination = instance_ids[crop]
        destination[local_component] = local_labels[local_component]

    masks: list[np.ndarray] = []
    objects: list[dict] = []
    for output_id, pair_index in enumerate(retained, start=1):
        mask = instance_ids == output_id
        if not bool(mask.any()):
            raise RuntimeError("retained SB-PF marker received no watershed support")
        masks.append(mask)
        coordinates = np.argwhere(mask)
        lower = coordinates.min(axis=0)
        upper = coordinates.max(axis=0) + 1
        birth_flat = int(pairs.birth_flat[pair_index])
        centroid = np.unravel_index(birth_flat, birth.shape)
        objects.append(
            {
                "instance_id": output_id,
                "birth_flat": birth_flat,
                "birth": float(birth_values[pair_index]),
                "death": float(death_values[pair_index]),
                "persistence": float(persistence[pair_index]),
                "centroid_zyx": [int(value) for value in centroid],
                "box_zyx": [
                    int(lower[0]),
                    int(lower[1]),
                    int(lower[2]),
                    int(upper[0]),
                    int(upper[1]),
                    int(upper[2]),
                ],
                "volume": int(mask.sum()),
            }
        )
    return {
        "count": len(objects),
        "instance_ids": instance_ids,
        "masks": tuple(masks),
        "objects": tuple(objects),
        "marker_outside_support": marker_outside_support,
    }
