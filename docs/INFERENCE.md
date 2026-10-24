# Inference contract

## Predicted field

[MicroBTUnitMassNetwork](../nnunetv2/nets/microbt_umctf/model.py) returns `field_logits`, `region_probabilities`, `object_density` and `transport`. The region probabilities are nested WT/TC/ET; transport has three channels.

## Centroid-peak decoding

[decode_umctf_instances](../nnunetv2/inference/microbt_umctf_decode.py) returns:

- `centers_zyx` and `center_scores`;
- `instance_labels` and `atom_map`;
- admitted input mass and accumulated atom mass;
- `count`, equal to the number of retained centroid candidates.

The default support threshold is 0.10, density threshold `1e-6`, minimum peak mass 0.25, NMS radius 2 voxels and maximum center count 64. Displacement scale and spacing conventions must match the trained model and preprocessing; these are not universal physical defaults.

## Evaluation/export wrapper

```bash
bash scripts/run_microbt_umctf_test251.sh \
  MODEL_FOLDER checkpoint_final.pth OUTPUT_DIR SOURCE_ROOT LABELS_TS
```

The five uppercase paths above are placeholders for your own resources. The wrapper loads `MODEL_FOLDER/fold_all/checkpoint_final.pth`, performs sharded inference and merge checks, then exports native-space semantic/instance NIfTI files. It requires the evaluation aliases from `.env.example`, CUDA configuration and a compatible trained model.

Saved evaluation arrays contain `region_mask` and `instance_labels`; the native exporter restores those predictions with nnU-Net geometry metadata. A successful file write is not evidence of registration or label correctness. The current wrapper preserves a fixed 251-case protocol and requires adaptation for other datasets.

Counts from the MicroBT++ decoder are peak-based. The earlier MicroBT repository's final-mask connected-component count is a different contract; label those comparisons explicitly.
