from __future__ import annotations

import re
from pathlib import Path


_CHANNEL_SUFFIX = re.compile(r"_(\d{4})\.nii\.gz$")


def resolve_case_files(source_root: str | Path, subject_id: str) -> tuple[Path, ...]:
    """Resolve contiguous nnU-Net channels from one dataset's own imagesTs."""
    source_root = Path(source_root)
    files = tuple(sorted(source_root.glob(f"{subject_id}_????.nii.gz")))
    indices: list[int] = []
    for path in files:
        match = _CHANNEL_SUFFIX.search(path.name)
        if match is None:
            raise ValueError(f"invalid nnU-Net channel filename: {path}")
        indices.append(int(match.group(1)))
    if not files or indices != list(range(len(files))):
        raise FileNotFoundError(
            f"missing or non-contiguous test modalities for {subject_id}: {indices}"
        )
    return files
