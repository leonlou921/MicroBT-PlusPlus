# GPU smoke test

The retained script is:

```bash
python scripts/smoke_microbt_sacl_c2_cuda.py
```

It requires a compatible PyTorch/CUDA installation, prepared nnU-Net plans/data and target metadata. Its current dataset argument is the synthetic example number `1`, and it instantiates the scratch continuation trainer rather than running a full 300-epoch experiment. Adapt that local dataset selection consistently with your resources before use.

The script performs bounded training steps at continuation epochs 5 and 17, checks finite losses and scheduled weights, and verifies optimizer activity. It is not a data-free hardware check, a complete inference test, or validation of manuscript scores.

Machine-specific extension binaries are not distributed. The persistent-field C++/CUDA sources are present for historical paths that require them; a compatible build must be supplied and verified separately.

No GPU smoke success is claimed for this documentation update.
