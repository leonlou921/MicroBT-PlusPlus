# Training and recovery

## Final entry point

```bash
bash scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh --print-contract
```

The final class is `MicroBTUnitMassSACLFinal300Trainer`. The retained template uses a 300-epoch PolyLR horizon, training from scratch, `fold=all`, three nested regions, and progressive density/transport weights. It expects explicit local configuration and refuses to overwrite an existing result directory.

After configuring resources and preparing data/plans/metadata:

```bash
python scripts/build_umctf_sacl_c2_source_manifest.py write
python scripts/build_umctf_sacl_c2_source_manifest.py verify
bash scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh
```

`write` establishes a new local baseline. Use `verify` for subsequent checks; regenerating a baseline after an unexpected source change does not validate that change.

## Retained cohort contract

The launcher carries a historical contract of 1011 training cases and 251 direct-test cases. This is not an automatic implementation of the manuscript's six-dataset 7:1:2 split protocol. `fold=all` must not be reported as cross-validation or as proof of a held-out validation split. Adapt and document the full protocol before applying the launcher to another cohort.

## Recovery

```bash
bash scripts/continue_microbt_umctf_sacl_c2_final300_foldall.sh --print-contract
```

Recovery accepts an epoch-200 `checkpoint_latest.pth` with the specified optimizer-state structure and continues the 300-epoch scientific schedule. It is a restricted handoff, not arbitrary resume. The training entry point and recovery entry point should not run concurrently.

## Historical scripts

Other PF/SPP/CTF, 60/200-epoch and convergence scripts are retained as redacted research templates. Generic `/path/to/example` locations require local adaptation. They do not replace the final trainer or establish equivalence to manuscript results.
