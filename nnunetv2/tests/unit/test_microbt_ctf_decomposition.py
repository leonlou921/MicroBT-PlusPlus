from __future__ import annotations

import numpy as np

from nnunetv2.evaluation.microbt_ctf_decomposition import (
    aggregate_arm_rows,
    build_oracle_transport,
    endpoint_instance_diagnostics,
    make_support_transport_matrix,
)


def _two_instances() -> tuple[np.ndarray, tuple[tuple[int, int, int], ...]]:
    labels = np.zeros((7, 7, 7), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1
    labels[5, 5, 5] = 2
    return labels, ((1, 1, 1), (5, 5, 5))


def test_oracle_transport_sends_every_instance_voxel_to_its_center() -> None:
    labels, centers = _two_instances()
    transport = build_oracle_transport(labels, centers, (7.0, 7.0, 7.0))

    for instance_id, center in enumerate(centers, start=1):
        coords = np.argwhere(labels == instance_id)
        endpoints = coords + transport[:, labels == instance_id].T * 7.0
        expected = np.repeat(np.asarray(center)[None], len(coords), axis=0)
        np.testing.assert_allclose(endpoints, expected, atol=1e-6)


def test_matrix_clips_predicted_support_to_gt_for_support_only_arm() -> None:
    labels, centers = _two_instances()
    predicted_support = np.full(labels.shape, 0.9, dtype=np.float32)
    predicted_transport = np.zeros((3, *labels.shape), dtype=np.float32)

    matrix = make_support_transport_matrix(
        predicted_support,
        predicted_transport,
        labels,
        centers,
        (7.0, 7.0, 7.0),
    )

    assert set(matrix) == {
        "gt_support_gt_transport",
        "gt_support_pred_transport",
        "pred_support_gt_transport",
        "pred_support_pred_transport",
    }
    support_only, _ = matrix["pred_support_gt_transport"]
    assert np.count_nonzero(support_only[labels == 0]) == 0
    assert np.all(support_only[labels > 0] == 0.9)
    end_to_end_support, _ = matrix["pred_support_pred_transport"]
    assert np.all(end_to_end_support == 0.9)


def test_endpoint_diagnostics_separate_mean_error_from_vote_concentration() -> None:
    labels, centers = _two_instances()
    transport = build_oracle_transport(labels, centers, (7.0, 7.0, 7.0))
    support = np.ones(labels.shape, dtype=np.float32)

    rows = endpoint_instance_diagnostics(
        labels, centers, support, transport, (7.0, 7.0, 7.0), (1.0, 1.0, 1.0)
    )

    assert len(rows) == 2
    assert all(row["endpoint_error_mean_mm"] == 0.0 for row in rows)
    assert all(row["endpoint_within_0_5_vox_rate"] == 1.0 for row in rows)
    assert all(row["rounded_endpoint_mode_fraction"] == 1.0 for row in rows)


def test_arm_aggregation_ignores_empty_micro_cases_and_weights_objects() -> None:
    rows = [
        {
            "gt_count": 1, "micro_count": 0, "predicted_count": 1,
            "proposal_hits_2mm": 1, "proposal_hits_5mm": 1,
            "micro_hits_2mm": 0, "micro_hits_5mm": 0,
            "centroid_inside_hits": 1, "nearest_center_sum_mm": 1.0,
            "micro_nearest_center_sum_mm": 0.0, "absolute_count_error": 0,
        },
        {
            "gt_count": 3, "micro_count": 2, "predicted_count": 5,
            "proposal_hits_2mm": 1, "proposal_hits_5mm": 2,
            "micro_hits_2mm": 0, "micro_hits_5mm": 1,
            "centroid_inside_hits": 1, "nearest_center_sum_mm": 9.0,
            "micro_nearest_center_sum_mm": 8.0, "absolute_count_error": 2,
        },
    ]

    summary = aggregate_arm_rows(rows)

    assert summary["proposal_recall_2mm"] == 0.5
    assert summary["proposal_recall_5mm"] == 0.75
    assert summary["micro_recall_2mm"] == 0.0
    assert summary["micro_recall_5mm"] == 0.5
    assert summary["centroid_inside"] == 0.5
    assert summary["mean_nearest_center_mm"] == 2.5
    assert summary["micro_mean_nearest_center_mm"] == 4.0
    assert summary["count_mae"] == 1.0
