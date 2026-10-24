# Public source release

This repository is a sanitized derivative, not the unchanged frozen experiment bundle.

## Sanitization

- Original server, storage, user-home and cluster paths were replaced by generic placeholders or explicit environment configuration.
- Private dataset and case identifiers were replaced with illustrative dataset names and synthetic cases. Upstream converters for publicly named datasets remain part of the framework.
- Dated run/environment labels and date-based example seeds were replaced with public template values.
- Original checkpoint records, data hashes, source-archive provenance and unsanitized documentation were omitted. They remain local to the owner.
- Medical-image fixtures, platform-specific extension binaries, cluster batch jobs and private Git history were omitted.
- The method-to-code table from the supplied reference image is represented as editable Markdown. The earlier reference image is not redistributed. Owner-authorized manuscript figure panels are listed separately in `docs/assets/README.md`.
- No concrete source, data or checkpoint hash records are published. The helper can generate a relative-path source manifest locally; generated records are ignored by Git.

## What changes affect reproduction

The network and loss mathematics, final trainer, continuation schedule and scientific thresholds are retained. Source literals used for private paths, identifiers and date-based seeds have changed. The final training/recovery scripts now read local resource configuration explicitly. Other historical scripts are examples requiring review and configuration.

The original cohort-size checks, three-region labels, modality ordering and checkpoint restrictions still exist. This release does not make those contracts apply to arbitrary datasets. Missing fixtures and omitted cluster scripts also limit some historical tests.

The source-manifest verifier checks only a locally generated baseline for this source. It does not authenticate external preprocessing data, checkpoint provenance or equivalence to the unpublished original bundle.

## Verification boundary

Checks performed for the initial source publication cover text-only file selection, privacy-pattern scanning, Python and Bash syntax, locally generated source-integrity checks, final printed contracts, configuration validation and a dependency-free subset of original tests. No full training, CUDA smoke test or complete inference run is claimed. No compatible dependency lockfile was recovered.

## Publication check results

- Six final core model/loss/trainer/decoder files are byte-identical to the local original.
- Python syntax: 351 files; Bash syntax: 43 files.
- Public configuration tests: 3 passed, including locally generated checksum drift detection and missing-configuration failure before training.
- Original dependency-free protocol tests: 4 passed.
- Privacy-pattern scan found no residual matches for the private server, dataset/case, dated environment, host, address or credential patterns checked.

These checks do not establish full runtime compatibility or an absolute guarantee of anonymization.

A historical hardcoded checkpoint checksum and a test tied to an unpublished private monitor manifest were also removed. Historical checkpoint guards require user-supplied values; no original checkpoint or split checksum literals are published.

## Manuscript presentation update

The owner subsequently supplied the MicroBT++ manuscript and requested a repository presentation following MicroBT. This adds three inspected figure panels, manuscript-reported results tables, current CCS/TDCE/SCCO terminology, citation metadata and focused usage documents. No algorithm source was changed by this update. Public benchmark labels and professional author names are taken from the supplied manuscript; private local dataset identifiers, contact details and resource locations remain excluded.

The new alignment notes explicitly distinguish the manuscript's compact loss formulas, six-dataset splits and maximum-diameter strata from the retained snapshot's integrated-mass/assignment losses, cohort launcher and volume-equivalent-diameter diagnostics.
