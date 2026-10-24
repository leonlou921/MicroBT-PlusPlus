from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import torch

from nnunetv2.inference.microbt_pf_predictor import MicroBTPFPredictor


def mirror_axis_combinations(
    allowed_axes: Sequence[int] | None,
    spatial_ndim: int,
) -> tuple[tuple[int, ...], ...]:
    """Return canonical tensor-axis flips, including the unmirrored task first."""
    if spatial_ndim <= 0:
        raise ValueError("spatial dimensionality must be positive")
    if allowed_axes is None:
        return ((),)
    axes = tuple(int(axis) for axis in allowed_axes)
    if len(set(axes)) != len(axes) or any(axis < 0 or axis >= spatial_ndim for axis in axes):
        raise ValueError("mirror axes do not match the input dimensionality")
    tensor_axes = tuple(axis + 2 for axis in axes)
    return ((),) + tuple(
        combination
        for count in range(1, len(tensor_axes) + 1)
        for combination in itertools.combinations(tensor_axes, count)
    )


@torch.inference_mode()
def predict_mirrored_microbatches(
    network: torch.nn.Module,
    x: torch.Tensor,
    unwrap: Callable[[Any], torch.Tensor],
    allowed_axes: Sequence[int] | None,
    microbatch_size: int,
) -> torch.Tensor:
    """Average registered mirror tasks while batching network forwards.

    Results are always unflipped and accumulated in canonical task order in
    FP32. The input batch is preserved; ``microbatch_size`` counts mirror tasks.
    """
    if int(microbatch_size) <= 0:
        raise ValueError("forward microbatch size must be positive")
    if x.ndim < 3:
        raise ValueError("input must contain batch, channel, and spatial dimensions")
    tasks = mirror_axis_combinations(allowed_axes, x.ndim - 2)
    input_batch = int(x.shape[0])
    outputs: list[torch.Tensor] = []
    for start in range(0, len(tasks), int(microbatch_size)):
        task_chunk = tasks[start : start + int(microbatch_size)]
        transformed = [torch.flip(x, axes) if axes else x for axes in task_chunk]
        raw = unwrap(network(torch.cat(transformed, dim=0)))
        if int(raw.shape[0]) != input_batch * len(task_chunk):
            raise ValueError("network changed the batch dimension during mirror inference")
        for axes, task_output in zip(task_chunk, raw.split(input_batch, dim=0)):
            outputs.append(torch.flip(task_output, axes) if axes else task_output)
    total = outputs[0].float().clone()
    for output in outputs[1:]:
        total.add_(output.float())
    total.div_(float(len(outputs)))
    return total


class MicroBTPFFastPredictor(MicroBTPFPredictor):
    """MicroBT-PF predictor with equivalent mirror-task GPU micro-batching."""

    def __init__(self, *args: Any, forward_microbatch_size: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if int(forward_microbatch_size) <= 0:
            raise ValueError("forward microbatch size must be positive")
        self.forward_microbatch_size = int(forward_microbatch_size)

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        axes = self.allowed_mirroring_axes if self.use_mirroring else None
        return predict_mirrored_microbatches(
            self.network,
            x,
            self._unwrap_raw_logits,
            axes,
            int(getattr(self, "forward_microbatch_size", 1)),
        )
