# Data format and configuration

The repository supplies no imaging dataset or trained weights. Use aligned multi-modal volumes and an explicit nnU-Net dataset definition, plans and splits.

## Generic nnU-Net layout

```text
nnUNet_raw/
  Dataset001_Example/
    dataset.json
    imagesTr/
      case-0001_0000.nii.gz
      case-0001_0001.nii.gz
      case-0001_0002.nii.gz
      case-0001_0003.nii.gz
    labelsTr/
      case-0001.nii.gz
    imagesTs/
    labelsTs/
```

These names are synthetic examples. The channel/label mapping must match your own `dataset.json`, preprocessed plans and trained model. The model expects three nested region targets in WT, TC, ET order; do not assume that every dataset uses the same integer label encoding.

## Evaluation adapter

The retained evaluator can resolve flat `<subject>_0000.nii.gz` through `<subject>_0003.nii.gz` files for generic subjects. It also retains a synthetic `EXAMPLE__` nested-case naming template. Its native suffix order is **t1c, t1n, t2f, t2w**, corresponding to contrast-enhanced T1, native T1, FLAIR and T2. Match the exact model's channel order; figure panel order is not a loader contract.

The fixed `test251` wrapper enforces 251 unique cases. It is a retained experiment template, not a general dataset converter. The lower-level evaluator has explicit expected-count options, but using a different cohort also requires adapting its labels and evaluation protocol.

## Configuration

Copy [.env.example](../.env.example) to an ignored `.env` and set your dataset number/name, raw/preprocessed roots, cache and interpreter. The source assumes the associated nnU-Net preprocessing products and MicroBT target metadata already exist; copying NIfTI files into the folders above does not complete preprocessing.

The six public benchmark labels shown in the manuscript tables describe the reported experiments. Private folder identifiers and original resource locations are not included in this release.
