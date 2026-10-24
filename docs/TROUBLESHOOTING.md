# Troubleshooting

| Symptom | What to check |
| --- | --- |
| Missing `MICROBT_DATASET_ID` or resource variable | Edit and load your ignored `.env`; inspect `.env.example`. |
| Dataset/plans cannot be resolved | Confirm dataset number/name, preprocessing roots and plans agree. |
| Source manifest is missing | Generate a local baseline with the manifest helper's `write` mode, then use `verify`. |
| Source manifest reports drift | Inspect the changes. Rewriting the baseline is not an integrity check. |
| Output directory already exists | The scratch launcher refuses overwrite. Select a separate run location locally or use only the compatible recovery contract. |
| Recovery rejects a checkpoint | Verify epoch 200 and optimizer structure. Arbitrary checkpoint resume is not supported by that wrapper. |
| Evaluator rejects case count | The retained shell wrapper enforces its historical 251-case protocol. Review the evaluator and cohort before adapting it. |
| Input modalities or labels do not match | Check channel order, region mapping, spacing and native geometry metadata. |
| Exact persistent-field backend unavailable | Precompiled machine-specific extensions are omitted. Build compatible sources only for paths that require them. |
| Paper scores or size strata do not match | Review loss, split, metric and diameter differences in Reproducibility; tests do not reproduce paper experiments. |

Do not paste private paths, real subject identifiers, checkpoint hashes or patient images into public issue reports.
