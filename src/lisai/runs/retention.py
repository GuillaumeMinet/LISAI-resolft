from __future__ import annotations

from pathlib import Path

from .io import read_run_metadata, write_run_metadata_atomic
from .schema import RunMetadata, utc_now


def set_run_kept(run_dir: str | Path, *, kept: bool) -> RunMetadata:
    """Set the lightweight retention flag for a training run.

    Keeping a run does not move or copy it.  The flag is persisted in the
    run metadata and is used by listing/filtering and pruning.
    """
    metadata = read_run_metadata(run_dir)
    if metadata.kept == kept:
        return metadata

    updated = metadata.model_copy(
        update={
            "kept": kept,
            "updated_at": utc_now(),
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


__all__ = ["set_run_kept"]
