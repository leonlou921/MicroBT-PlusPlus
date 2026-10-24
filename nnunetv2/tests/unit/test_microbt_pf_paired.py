from __future__ import annotations

import pytest

from nnunetv2.evaluation.microbt_pf_paired import (
    paired_bootstrap_ci,
    paired_comparison,
)


def _rows(values: list[float]) -> list[dict]:
    return [
        {"subject_id": f"case_{index}", "metrics": {"wt_dice": value, "count_mae": 1 - value}}
        for index, value in enumerate(values)
    ]


def test_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_ci([1.0, 2.0, 3.0], samples=1000, seed=7)
    second = paired_bootstrap_ci([1.0, 2.0, 3.0], samples=1000, seed=7)
    assert first == second


def test_paired_comparison_aligns_ids_and_directions() -> None:
    result = paired_comparison(_rows([0.1, 0.2]), list(reversed(_rows([0.2, 0.1]))), samples=100)
    assert result["metrics"]["wt_dice"]["wins"] == 1
    assert result["metrics"]["wt_dice"]["ties"] == 0
    assert result["metrics"]["count_mae"]["wins"] == 1
    with pytest.raises(ValueError, match="subject"):
        paired_comparison(_rows([0.1]), _rows([0.1, 0.2]), samples=100)
