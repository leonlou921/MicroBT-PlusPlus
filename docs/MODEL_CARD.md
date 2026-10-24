# Model card

## Model and intended use

MicroBT++ is a multi-modal 3D research framework for tumor segmentation and counting. It predicts nested semantic regions, lesion density and centroid transport and recovers instances by centroid-peak decoding. It extends the research direction of MicroBT.

The repository supports academic method inspection, adaptation and offline evaluation. It is not a clinically validated diagnostic system.

## Inputs and outputs

Inputs are aligned multi-channel MRI volumes with a consistent nnU-Net preprocessing and label specification. Outputs include semantic probabilities, density/transport fields, centroid candidates, instance labels, boxes and counts. The available export path also writes semantic/instance NIfTI files. Users supply their own trained weights.

## Evidence

The supplied manuscript reports experiments on six MRI datasets; [Results](RESULTS.md) transcribes its quantitative comparisons. These reported results are separate from the source snapshot's lightweight software tests. No checkpoint or independently reproduced score is distributed.

## Limitations

Very small lesions remain difficult: manuscript Figure 3 reports 50.4% recall for lesions ≤3 mm despite high precision. Results depend on imaging, labels, preprocessing, size definitions, decoder thresholds and domain shift. Peak detection is capped by a configurable maximum number of centers.

The current snapshot and manuscript have documented loss/protocol differences. Public-template identities, paths and seeds have been sanitized. See [Reproducibility](REPRODUCIBILITY.md) before interpreting the release as a reproduction package.
