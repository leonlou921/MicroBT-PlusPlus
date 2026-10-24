from nnunetv2.scripts.profile_microbt_pf_cuda_pairs import evaluate_cuda_pairs_probe


def test_cuda_pairs_probe_requires_exactness_and_latency() -> None:
    passed = evaluate_cuda_pairs_probe(
        {
            "randomized_mismatches": 0,
            "tie_heavy_mismatches": 0,
            "deterministic": True,
            "constant_fullres_median_seconds": 0.49,
            "constant_fullres_p95_seconds": 0.69,
            "protected_state_unchanged": True,
        }
    )
    assert passed == {"admitted": True, "reasons": []}

    rejected = evaluate_cuda_pairs_probe(
        {
            "randomized_mismatches": 1,
            "tie_heavy_mismatches": 0,
            "deterministic": False,
            "constant_fullres_median_seconds": 0.51,
            "constant_fullres_p95_seconds": 0.71,
            "protected_state_unchanged": False,
        }
    )
    assert rejected["admitted"] is False
    assert rejected["reasons"] == [
        "randomized_exactness_mismatch",
        "repeated_run_nondeterminism",
        "median_exceeds_0.50_seconds",
        "p95_exceeds_0.70_seconds",
        "protected_state_drift",
    ]
