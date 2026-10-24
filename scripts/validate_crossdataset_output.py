from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    summary = json.loads((args.output / "summary.json").read_text())
    audit = json.loads((args.output / "native_nifti_export_audit.json").read_text())
    expected = args.expected_count
    subjects = tuple(summary.get("subject_ids", ()))
    if summary.get("status") != "complete" or len(subjects) != expected or len(set(subjects)) != expected:
        raise ValueError("test summary coverage failure")
    if int(summary["metrics"].get("nonfinite_case_count", -1)) != 0:
        raise ValueError("nonfinite test case")
    if int(summary.get("prediction_file_count", -1)) != expected:
        raise ValueError("prediction NPZ coverage failure")
    if audit.get("status") != "passed" or int(audit.get("subject_count", -1)) != expected:
        raise ValueError("native export audit failure")
    for folder in ("semantic", "instances"):
        if len(tuple((args.output / "native_nifti" / folder).glob("*.nii.gz"))) != expected:
            raise ValueError(f"native {folder} coverage failure")
    print(json.dumps({"status": "passed", "subject_count": expected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
