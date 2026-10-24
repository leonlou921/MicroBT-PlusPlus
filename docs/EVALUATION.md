# Evaluation definitions

## Manuscript metrics

The manuscript reports whole-tumor Dice and HD95, lesion-level AP_vox, count MAPE and micro-tumor strata defined by maximum 3D diameter. AP_vox uses confidence-ranked, one-to-one matching when axis-aligned prediction and reference boxes share at least one voxel. Reference lesions are 26-connected components.

[Results](RESULTS.md) reproduces the reported values with their original scale. Lower MAPE is better; the tables' decimals are not silently converted into percentages.

## Released evaluator

The retained evaluator computes three-region Dice, count MAE, box AP at IoU thresholds 0.50/0.75, centroid-distance recall and other diagnostics. The direct-test adapter additionally supplies `ap_1vox`. Saved records and summaries support local integrity checks but are not published private run records.

Important distinctions:

- Count MAE is not MAPE; the retained summary does not provide the manuscript's MAPE tables automatically.
- `_equivalent_diameter` derives a volume-equivalent sphere diameter. It is not the manuscript's maximum 3D diameter, so micro-size stratification is not directly interchangeable.
- Decoder support threshold 0.10 and semantic-region evaluation threshold 0.5 have different roles.
- Peak count, instance-label count and connected-component count must not be silently substituted for one another.

These distinctions are documented to support correct interpretation of the available code. The documentation update does not change metric implementations or claim to reproduce the six-dataset results.
