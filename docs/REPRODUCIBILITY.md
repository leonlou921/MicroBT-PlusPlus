# Reproducibility

This release contains sanitized research source and owner-authorized manuscript illustrations. It does not contain the original datasets, checkpoint weights, environment lockfile, private resource locations, or concrete hash records.

## Paper versus source snapshot

| Aspect | Supplied manuscript | Released snapshot |
| --- | --- | --- |
| Density objective | Per-lesion pointwise density penalty in Algorithm 1 / Eq. (5) | Integrated lesion-mass Smooth-L1, detached scaling, background-mass penalty and hard-survival term |
| Transport objective | Compact equal-instance endpoint L1 expression | Physical-space Smooth-L1 endpoint term plus support-assignment consistency |
| Cohort protocol | Six datasets with case-level 7:1:2 train/validation/test splits | Historical final launcher retains `fold=all`, 1011-train / 251-direct-test contract |
| Micro-lesion diameter | Maximum 3D diameter | Retained evaluator uses volume-equivalent diameter for its size diagnostics |
| Counting summary | MAPE and AP_vox | Count MAE and other diagnostics, with `ap_1vox` added by the direct-test adapter |
| Environment and identity | Experimental description in manuscript | Unpinned source-derived dependencies, generic configuration, synthetic case examples and public-template seed |

The 7-channel field structure, hierarchical gating, centroid-peak decoder and SCCO continuation mechanism map to the code. The rows above prevent an unsupported claim of exact equation/protocol equivalence.

## Local reproducibility record

For a new experiment, record your code revision, dependency versions, own dataset/label definition, case split, preprocessing plans, channel order, decoder settings and trained checkpoint identity in your local records. Generated integrity records remain ignored by Git. Never add private paths, real case identifiers, concrete source/data/checkpoint hash records or patient files to this public repository.

## Verification performed here

Publication checks validate links, figure extraction, table transcription, source immutability, syntax and the small dependency-free test suite. No full GPU training/inference or six-dataset score reproduction was run. The current source includes no verified generic end-to-end pipeline equivalent to `microbt.cli` in the earlier MicroBT repository.
