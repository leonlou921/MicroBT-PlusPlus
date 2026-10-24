from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.test251_shard_accel import (
    make_test_shard_plan,
    merge_test_shards,
    validate_test_shard_plan,
)


def _write_artifacts(
    output: Path, subject: str, checkpoint: str, evaluator: str
) -> None:
    (output / "cases").mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    (output / "cases" / f"{subject}.json").write_text(
        json.dumps(
            {
                "subject_id": subject,
                "checkpoint_sha256": checkpoint,
                "evaluator_sha256": evaluator,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "predictions" / f"{subject}.npz",
        region_mask=np.zeros((3, 2, 2, 2), dtype=np.uint8),
        instance_labels=np.zeros((2, 2, 2), dtype=np.int16),
    )


def test_test251_shards_cover_once_and_merge_cases_with_predictions(
    tmp_path: Path,
) -> None:
    subjects = tuple(f"EXAMPLE__case-{index}" for index in range(7))
    checkpoint = "checkpoint-hash"
    evaluator = "evaluator-hash"
    main = tmp_path / "main"
    _write_artifacts(main, subjects[0], checkpoint, evaluator)

    plan_path = make_test_shard_plan(
        subject_ids=subjects,
        main_output=main,
        shard_root=tmp_path / "shards",
        workers=3,
        weights={subject: index + 1 for index, subject in enumerate(subjects)},
        checkpoint_sha256=checkpoint,
        evaluator_sha256=evaluator,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_test_shard_plan(plan)
    assigned = [
        subject for shard in plan["shards"] for subject in shard["subject_ids"]
    ]
    assert set(assigned) == set(subjects[1:])
    assert len(assigned) == len(set(assigned))

    shard_outputs = []
    for shard in plan["shards"]:
        output = tmp_path / shard["name"]
        for subject in shard["subject_ids"]:
            _write_artifacts(output, subject, checkpoint, evaluator)
        shard_outputs.append(output)

    audit_path = merge_test_shards(
        plan_path=plan_path,
        main_output=main,
        shard_outputs=shard_outputs,
        audit_path=tmp_path / "merge-audit.json",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert audit["case_count"] == len(subjects)
    assert audit["prediction_count"] == len(subjects)
    assert {path.stem for path in (main / "cases").glob("*.json")} == set(subjects)
    assert {path.stem for path in (main / "predictions").glob("*.npz")} == set(subjects)
    linked_subject = plan["shards"][0]["subject_ids"][0]
    assert (main / "predictions" / f"{linked_subject}.npz").samefile(
        shard_outputs[0] / "predictions" / f"{linked_subject}.npz"
    )
