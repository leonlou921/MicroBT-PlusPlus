from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_COMPONENT_KEYS = (
    "loss",
    "region",
    "field",
    "betti_0",
    "persistent_field",
    "betti_0_raw",
    "h0_active",
)


def _scalar(output: Mapping[str, Any], key: str) -> float:
    if key not in output:
        raise KeyError(f"component output is missing {key}")
    value = np.asarray(output[key])
    if value.size != 1:
        raise ValueError(f"component {key} must be scalar")
    number = float(value.reshape(-1)[0])
    if not math.isfinite(number):
        raise ValueError(f"nonfinite {key}")
    return number


def summarize_component_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    epoch: int,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    if phase not in {"train", "val"}:
        raise ValueError("phase must be train or val")
    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    if not outputs:
        raise ValueError("component outputs must not be empty")

    columns = {
        key: [_scalar(output, key) for output in outputs]
        for key in _COMPONENT_KEYS
    }
    active = np.asarray(columns["h0_active"], dtype=np.float64)
    if not np.isin(active, (0.0, 1.0)).all():
        raise ValueError("h0_active must be binary")
    active_mask = active.astype(bool)
    raw_betti = np.asarray(columns["betti_0_raw"], dtype=np.float64)
    raw_active_mean = (
        float(raw_betti[active_mask].mean()) if bool(active_mask.any()) else 0.0
    )
    summary = {
        "schema_version": 1,
        "phase": phase,
        "epoch": int(epoch),
        "steps": len(outputs),
        "h0_active_steps": int(active_mask.sum()),
        "h0_inactive_steps": int((~active_mask).sum()),
        "loss_mean": float(np.mean(columns["loss"])),
        "region_mean": float(np.mean(columns["region"])),
        "field_mean": float(np.mean(columns["field"])),
        "betti_0_raw_active_mean": raw_active_mean,
        "betti_0_weighted_all_mean": float(np.mean(columns["betti_0"])),
        "persistent_field_mean": float(np.mean(columns["persistent_field"])),
        "schedule": {str(key): schedule[key] for key in sorted(schedule)},
    }
    optional = (
        "birth_completeness",
        "birth_completeness_raw",
        "object_morse",
    )
    presence = {
        key: [key in output for output in outputs]
        for key in optional
    }
    for key, flags in presence.items():
        if any(flags) and not all(flags):
            raise KeyError(f"component {key} must be present in every output")
    if all(all(presence[key]) for key in optional):
        birth = np.asarray(
            [_scalar(output, "birth_completeness") for output in outputs],
            dtype=np.float64,
        )
        raw_birth = np.asarray(
            [_scalar(output, "birth_completeness_raw") for output in outputs],
            dtype=np.float64,
        )
        object_morse = np.asarray(
            [_scalar(output, "object_morse") for output in outputs],
            dtype=np.float64,
        )
        summary.update(
            {
                "birth_completeness_raw_active_mean": (
                    float(raw_birth[active_mask].mean())
                    if bool(active_mask.any())
                    else 0.0
                ),
                "birth_completeness_weighted_all_mean": float(birth.mean()),
                "object_morse_mean": float(object_morse.mean()),
            }
        )
    obr_optional = (
        "birth_reservation",
        "birth_reservation_raw",
        "birth_existence",
        "birth_existence_raw",
        "birth_prominence",
        "birth_prominence_raw",
        "object_morse",
    )
    obr_presence = {
        key: [key in output for output in outputs]
        for key in obr_optional
    }
    for key, flags in obr_presence.items():
        if any(flags) and not all(flags):
            raise KeyError(f"component {key} must be present in every output")
    if all(all(obr_presence[key]) for key in obr_optional):
        obr_columns = {
            key: np.asarray(
                [_scalar(output, key) for output in outputs], dtype=np.float64
            )
            for key in obr_optional
        }
        summary.update(
            {
                "birth_reservation_weighted_all_mean": float(
                    obr_columns["birth_reservation"].mean()
                ),
                "birth_reservation_raw_all_mean": float(
                    obr_columns["birth_reservation_raw"].mean()
                ),
                "birth_existence_weighted_all_mean": float(
                    obr_columns["birth_existence"].mean()
                ),
                "birth_existence_raw_all_mean": float(
                    obr_columns["birth_existence_raw"].mean()
                ),
                "birth_prominence_weighted_all_mean": float(
                    obr_columns["birth_prominence"].mean()
                ),
                "birth_prominence_raw_all_mean": float(
                    obr_columns["birth_prominence_raw"].mean()
                ),
                "object_morse_mean": float(
                    obr_columns["object_morse"].mean()
                ),
            }
        )
    false_presence = ["false_persistence" in output for output in outputs]
    if any(false_presence) and not all(false_presence):
        raise KeyError("component false_persistence must be present in every output")
    if all(false_presence):
        false_persistence = np.asarray(
            [_scalar(output, "false_persistence") for output in outputs],
            dtype=np.float64,
        )
        summary["false_persistence_weighted_all_mean"] = float(
            false_persistence.mean()
        )
    return summary


def upsert_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
    key_fields: Sequence[str],
) -> None:
    destination = Path(path)
    if not key_fields:
        raise ValueError("key_fields must not be empty")
    missing = [key for key in key_fields if key not in record]
    if missing:
        raise KeyError(f"record is missing key fields: {missing}")
    rows: list[dict[str, Any]] = []
    if destination.is_file():
        rows = [
            json.loads(line)
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    identity = tuple(record[key] for key in key_fields)
    replacement = dict(record)
    replaced = False
    for index, row in enumerate(rows):
        if tuple(row.get(key) for key in key_fields) == identity:
            rows[index] = replacement
            replaced = True
            break
    if not replaced:
        rows.append(replacement)
    rows.sort(key=lambda row: tuple(row.get(key) for key in key_fields))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
