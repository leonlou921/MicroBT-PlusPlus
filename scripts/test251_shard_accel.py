#!/usr/bin/env python3
"""Prepare, validate, and merge disjoint test251 inference shards."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_artifact_subjects(
    output: Path, checkpoint_sha256: str, evaluator_sha256: str
) -> dict[str, tuple[Path, Path]]:
    case_dir = output / "cases"
    prediction_dir = output / "predictions"
    case_paths = {path.stem: path for path in case_dir.glob("*.json")}
    prediction_paths = {path.stem: path for path in prediction_dir.glob("*.npz")}
    if set(case_paths) != set(prediction_paths):
        raise ValueError(f"case/prediction coverage mismatch in {output}")
    validated: dict[str, tuple[Path, Path]] = {}
    for subject, case_path in case_paths.items():
        record = _read_json(case_path)
        if record.get("subject_id") != subject:
            raise ValueError(f"case subject mismatch: {case_path}")
        if record.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"case checkpoint mismatch: {case_path}")
        if record.get("evaluator_sha256") != evaluator_sha256:
            raise ValueError(f"case evaluator mismatch: {case_path}")
        prediction_path = prediction_paths[subject]
        with np.load(prediction_path, allow_pickle=False) as payload:
            if set(payload.files) != {"region_mask", "instance_labels"}:
                raise ValueError(f"prediction payload mismatch: {prediction_path}")
        validated[subject] = (case_path, prediction_path)
    return validated


def validate_test_shard_plan(plan: dict[str, Any]) -> None:
    subjects = [str(value) for value in plan["subject_ids"]]
    completed = [str(value) for value in plan["completed_subject_ids"]]
    assigned = [
        str(subject)
        for shard in plan["shards"]
        for subject in shard["subject_ids"]
    ]
    if len(subjects) != len(set(subjects)):
        raise ValueError("test subject list is not unique")
    if len(completed) != len(set(completed)):
        raise ValueError("completed test subject list is not unique")
    if len(assigned) != len(set(assigned)):
        raise ValueError("test shard overlap detected")
    if set(completed) & set(assigned):
        raise ValueError("completed subjects overlap test shards")
    if set(completed) | set(assigned) != set(subjects):
        raise ValueError("test shards do not cover the frozen subject list")


def make_test_shard_plan(
    *,
    subject_ids: Sequence[str],
    main_output: Path,
    shard_root: Path,
    workers: int,
    weights: Mapping[str, int],
    checkpoint_sha256: str,
    evaluator_sha256: str,
) -> Path:
    if workers < 2:
        raise ValueError("at least two test workers are required")
    subjects = tuple(sorted(str(value) for value in subject_ids))
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("test subjects must be nonempty and unique")
    if set(weights) != set(subjects):
        raise ValueError("one workload weight is required per test subject")

    shard_root.mkdir(parents=True, exist_ok=True)
    plan_path = shard_root / "plan.json"
    if plan_path.is_file():
        existing = _read_json(plan_path)
        validate_test_shard_plan(existing)
        if (
            existing["subject_ids"] != list(subjects)
            or existing["checkpoint_sha256"] != checkpoint_sha256
            or existing["evaluator_sha256"] != evaluator_sha256
            or len(existing["shards"]) != workers
        ):
            raise RuntimeError("existing test shard plan drift")
        return plan_path

    completed_artifacts = _validated_artifact_subjects(
        main_output, checkpoint_sha256, evaluator_sha256
    )
    completed = sorted(completed_artifacts)
    remaining = [subject for subject in subjects if subject not in completed_artifacts]
    bins: list[list[str]] = [[] for _ in range(workers)]
    totals = [0 for _ in range(workers)]
    for subject in sorted(remaining, key=lambda value: (-int(weights[value]), value)):
        worker = min(range(workers), key=lambda index: (totals[index], len(bins[index])))
        bins[worker].append(subject)
        totals[worker] += int(weights[subject])

    shards = []
    for index, values in enumerate(bins):
        subject_file = shard_root / f"shard-{index}-subjects.json"
        _atomic_write_json(subject_file, sorted(values))
        shards.append(
            {
                "name": f"shard-{index}",
                "subject_ids": sorted(values),
                "expected_count": len(values),
                "weight_bytes": totals[index],
                "subject_file": str(subject_file.resolve()),
                "subject_file_sha256": _sha256(subject_file),
            }
        )
    plan = {
        "schema_version": 1,
        "status": "prepared",
        "subject_ids": list(subjects),
        "expected_count": len(subjects),
        "completed_subject_ids": completed,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_sha256": evaluator_sha256,
        "workers": workers,
        "shards": shards,
    }
    validate_test_shard_plan(plan)
    _atomic_write_json(plan_path, plan)
    return plan_path


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        os.link(source, temporary)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def recover_partial_test_shards(
    *,
    plan_path: Path,
    main_output: Path,
    shard_outputs: Sequence[Path],
    audit_path: Path,
) -> Path:
    """Atomically preserve valid completed artifacts before repartitioning.

    Each recovered subject must belong to the corresponding frozen shard in
    ``plan_path``.  The checkpoint and evaluator hashes embedded in every case
    record are validated before either artifact is linked into ``main_output``.
    """
    plan = _read_json(plan_path)
    validate_test_shard_plan(plan)
    if len(shard_outputs) != len(plan["shards"]):
        raise ValueError("one output directory is required per test shard")
    checkpoint = str(plan["checkpoint_sha256"])
    evaluator = str(plan["evaluator_sha256"])
    frozen_subjects = set(str(value) for value in plan["subject_ids"])
    existing = _validated_artifact_subjects(main_output, checkpoint, evaluator)
    if not set(existing).issubset(frozen_subjects):
        raise ValueError("main output contains a subject outside the frozen plan")

    recovered: dict[str, tuple[Path, Path]] = {}
    for shard, shard_output in zip(plan["shards"], shard_outputs, strict=True):
        assigned = set(str(value) for value in shard["subject_ids"])
        artifacts = _validated_artifact_subjects(shard_output, checkpoint, evaluator)
        unexpected = set(artifacts) - assigned
        if unexpected:
            raise ValueError(
                f"partial shard contains unassigned subjects: {sorted(unexpected)}"
            )
        overlap = set(recovered) & set(artifacts)
        if overlap:
            raise ValueError(f"partial shard overlap detected: {sorted(overlap)}")
        recovered.update(artifacts)

    for subject, (case_path, prediction_path) in sorted(recovered.items()):
        for source, destination in (
            (case_path, main_output / "cases" / case_path.name),
            (prediction_path, main_output / "predictions" / prediction_path.name),
        ):
            if destination.is_file():
                if destination.read_bytes() != source.read_bytes():
                    raise ValueError(f"conflicting recovered test artifact: {subject}")
            else:
                _atomic_copy(source, destination)

    preserved = _validated_artifact_subjects(main_output, checkpoint, evaluator)
    if not set(preserved).issubset(frozen_subjects):
        raise ValueError("recovered output contains a subject outside the frozen plan")
    audit = {
        "schema_version": 1,
        "status": "passed",
        "recovered_count": len(recovered),
        "recovered_subject_ids": sorted(recovered),
        "preserved_count": len(preserved),
        "preserved_subject_ids": sorted(preserved),
        "checkpoint_sha256": checkpoint,
        "evaluator_sha256": evaluator,
        "case_sha256": {
            subject: _sha256(paths[0]) for subject, paths in sorted(preserved.items())
        },
        "prediction_sha256": {
            subject: _sha256(paths[1]) for subject, paths in sorted(preserved.items())
        },
    }
    _atomic_write_json(audit_path, audit)
    return audit_path


def merge_test_shards(
    *,
    plan_path: Path,
    main_output: Path,
    shard_outputs: Sequence[Path],
    audit_path: Path,
) -> Path:
    plan = _read_json(plan_path)
    validate_test_shard_plan(plan)
    if len(shard_outputs) != len(plan["shards"]):
        raise ValueError("one output directory is required per test shard")
    checkpoint = str(plan["checkpoint_sha256"])
    evaluator = str(plan["evaluator_sha256"])

    for shard, shard_output in zip(plan["shards"], shard_outputs, strict=True):
        expected = set(str(value) for value in shard["subject_ids"])
        artifacts = _validated_artifact_subjects(shard_output, checkpoint, evaluator)
        if set(artifacts) != expected:
            raise ValueError(f"incomplete test shard output: {shard['name']}")
        for subject, (case_path, prediction_path) in artifacts.items():
            for source, destination in (
                (case_path, main_output / "cases" / case_path.name),
                (prediction_path, main_output / "predictions" / prediction_path.name),
            ):
                if destination.is_file():
                    if destination.read_bytes() != source.read_bytes():
                        raise ValueError(f"conflicting merged test artifact: {subject}")
                else:
                    _atomic_copy(source, destination)

    merged = _validated_artifact_subjects(main_output, checkpoint, evaluator)
    subjects = set(str(value) for value in plan["subject_ids"])
    if set(merged) != subjects:
        raise ValueError("merged test artifacts do not cover the frozen subject list")
    audit = {
        "schema_version": 1,
        "status": "passed",
        "case_count": len(merged),
        "prediction_count": len(merged),
        "unique_subject_count": len(merged),
        "checkpoint_sha256": checkpoint,
        "evaluator_sha256": evaluator,
        "case_sha256": {
            subject: _sha256(paths[0]) for subject, paths in sorted(merged.items())
        },
        "prediction_sha256": {
            subject: _sha256(paths[1]) for subject, paths in sorted(merged.items())
        },
    }
    _atomic_write_json(audit_path, audit)
    return audit_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--labels-ts", type=Path, required=True)
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--main-output", type=Path, required=True)
    prepare.add_argument("--shard-root", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=3)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--evaluator-sha256", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--main-output", type=Path, required=True)
    merge.add_argument("--shard-output", type=Path, action="append", required=True)
    merge.add_argument("--audit", type=Path, required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--plan", type=Path, required=True)
    recover.add_argument("--main-output", type=Path, required=True)
    recover.add_argument("--shard-output", type=Path, action="append", required=True)
    recover.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        from scripts.evaluate_microbt_umctf_test251 import (
            _test_evaluator_sha256,
            load_test_subjects,
            resolve_met_case_files,
        )

        subjects = load_test_subjects(args.labels_ts, expected_count=251)
        weights = {
            subject: sum(
                path.stat().st_size
                for path in resolve_met_case_files(args.source_root, subject)
            )
            for subject in subjects
        }
        actual_evaluator = _test_evaluator_sha256()
        if actual_evaluator != args.evaluator_sha256:
            raise RuntimeError("test evaluator hash drift during shard preparation")
        path = make_test_shard_plan(
            subject_ids=subjects,
            main_output=args.main_output,
            shard_root=args.shard_root,
            workers=args.workers,
            weights=weights,
            checkpoint_sha256=args.checkpoint_sha256,
            evaluator_sha256=args.evaluator_sha256,
        )
    elif args.command == "merge":
        path = merge_test_shards(
            plan_path=args.plan,
            main_output=args.main_output,
            shard_outputs=args.shard_output,
            audit_path=args.audit,
        )
    else:
        path = recover_partial_test_shards(
            plan_path=args.plan,
            main_output=args.main_output,
            shard_outputs=args.shard_output,
            audit_path=args.audit,
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
