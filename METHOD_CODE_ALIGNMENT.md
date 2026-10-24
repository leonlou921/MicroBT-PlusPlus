# Paper-to-code alignment

The current manuscript uses **CCS**, **TDCE**, and **SCCO**. The source retains the earlier internal names `UMCTF`, `CTF`, and `SACL` to preserve imports and trainer discovery. This is a mapping of concepts; it does not claim all manuscript equations are identical to the snapshot.

| Manuscript term | Implementation | Notes |
| --- | --- | --- |
| Count-Coupled Segmentation (CCS) | `nnunetv2/nets/microbt_umctf/model.py`; losses and decoder below | Shared semantic, density and transport representation. |
| TDCE: hierarchical gating | `nnunetv2/nets/microbt_umctf/model.py` | Seven channels; nested WT/TC/ET, WT-gated density, bounded transport. |
| TDCE: density supervision | `nnunetv2/training/loss/microbt_umctf_losses.py` | Integrated-mass Smooth-L1 with background and hard-survival terms; not the literal pointwise squared density expression of manuscript Eq. (5). |
| TDCE: centroid transport | `nnunetv2/training/loss/microbt_ctf_losses.py` | Equal-instance physical endpoint Smooth-L1 plus support-assignment consistency; not only the compact endpoint L1 formula. |
| TDCE: centroid-peak decoding | `nnunetv2/inference/microbt_umctf_decode.py` | Trilinear splatting, local peak filtering, endpoint assignment, instance boxes and peak count. |
| SCCO | `nnunetv2/training/loss/microbt_sacl_losses.py`; `nnunetv2/training/nnUNetTrainer/MicroBTSACLTrainer.py` | Legacy SACL continuation: semantic weight 1, measure `0.25*lambda`, transport `0.25*lambda^2`; endpoint epoch 29. |
| Final trainer | `MicroBTUnitMassSACLFinal300Trainer` | One 300-epoch PolyLR trajectory. |

Earlier package descriptions used LMC, APD and SAC for related components. The current repository presentation follows the supplied manuscript's terminology while preserving source names. SCCO is not the earlier MicroBT repository's SACO pipeline.

See [Reproducibility](docs/REPRODUCIBILITY.md) for cohort/split, metric and diameter-definition differences. The documentation update leaves algorithm source unchanged.
