from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.infra.paths.run_location import (
    InferredRunLocation,
    infer_run_location as infer_run_location_from_paths,
    iter_run_metadata_paths,
)

from .io import read_run_metadata
from .schema import RUN_METADATA_FILENAME, RunMetadata, normalize_posix_path


@dataclass(frozen=True)
class DiscoveredRun:
    metadata: RunMetadata
    metadata_path: Path
    run_dir: Path
    dataset: str
    model_subfolder: str
    group_path: str | None
    path: str
    path_consistent: bool = True
    consistency_issues: tuple[str, ...] = ()



    @property
    def last_seen(self) -> datetime:
        return self.metadata.last_heartbeat_at


@dataclass(frozen=True)
class InvalidRunMetadata:
    metadata_path: Path
    kind: str
    message: str


@dataclass(frozen=True)
class ScanResults:
    runs: tuple[DiscoveredRun, ...]
    invalid: tuple[InvalidRunMetadata, ...]


_PATHS = Paths(settings)


def default_datasets_root() -> Path:
    """Return the canonical root scanned for training runs."""
    return _PATHS.training_datasets_root()


def scan_runs(datasets_root: str | Path | None = None) -> ScanResults:
    use_canonical_root = datasets_root is None
    root = default_datasets_root() if use_canonical_root else Path(datasets_root)
    root = root.resolve()
    stored_path_root = _PATHS.datasets_root().parent if use_canonical_root else root.parent
    if not root.exists():
        return ScanResults(runs=(), invalid=())

    runs: list[DiscoveredRun] = []
    invalid: list[InvalidRunMetadata] = []

    # retrieve metadata paths of all saved runs folder
    for meta_path in iter_run_metadata_paths(
        root,
        metadata_filename=RUN_METADATA_FILENAME,
        paths=_PATHS,
    ):
        try:
            inferred = infer_run_location(
                meta_path,
                root,
                stored_path_root=stored_path_root,
            )
            metadata = read_run_metadata(meta_path)
            mismatches = metadata_path_mismatches(metadata, inferred)

            runs.append(
                DiscoveredRun(
                    metadata=metadata,
                    metadata_path=meta_path,
                    run_dir=inferred.run_dir,
                    dataset=inferred.dataset,
                    model_subfolder=inferred.model_subfolder,
                    group_path=inferred.group_path,
                    path=inferred.path,
                    path_consistent=not mismatches,
                    consistency_issues=tuple(mismatches),
                )
            )
        except json.JSONDecodeError as exc:
            invalid.append(
                InvalidRunMetadata(
                    metadata_path=meta_path,
                    kind="json_parse_error",
                    message=str(exc),
                )
            )
        except ValidationError as exc:
            invalid.append(
                InvalidRunMetadata(
                    metadata_path=meta_path,
                    kind="schema_validation_error",
                    message=str(exc),
                )
            )
        except (OSError, ValueError) as exc:
            invalid.append(
                InvalidRunMetadata(
                    metadata_path=meta_path,
                    kind="scan_error",
                    message=str(exc),
                )
            )

    runs.sort(key=lambda run: run.last_seen, reverse=True)
    return ScanResults(runs=tuple(runs), invalid=tuple(invalid))


def infer_run_location(
    metadata_path: str | Path,
    datasets_root: str | Path,
    *,
    stored_path_root: str | Path | None = None,
) -> InferredRunLocation:
    return infer_run_location_from_paths(
        metadata_path,
        datasets_root,
        metadata_filename=RUN_METADATA_FILENAME,
        run_container_dirname=_PATHS.run_container_dirname(),
        stored_path_root=stored_path_root,
    )


def metadata_path_mismatches(metadata: RunMetadata, inferred: InferredRunLocation) -> list[str]:
    mismatches: list[str] = []
    if metadata.dataset != inferred.dataset:
        mismatches.append(
            f"dataset_mismatch: metadata={metadata.dataset!r}, filesystem={inferred.dataset!r}"
        )

    normalized_model_subfolder = normalize_posix_path(metadata.model_subfolder)
    if normalized_model_subfolder != inferred.model_subfolder:
        mismatches.append(
            "model_subfolder_mismatch: "
            f"metadata={normalized_model_subfolder!r}, filesystem={inferred.model_subfolder!r}"
        )

    normalized_group_path = (
        None if metadata.group_path is None else normalize_posix_path(metadata.group_path)
    )
    if normalized_group_path != inferred.group_path:
        mismatches.append(
            f"group_path_mismatch: metadata={normalized_group_path!r}, filesystem={inferred.group_path!r}"
        )

    normalized_path = normalize_posix_path(metadata.path)
    if normalized_path != inferred.path:
        mismatches.append(f"path_mismatch: metadata={normalized_path!r}, filesystem={inferred.path!r}")

    return mismatches


__all__ = [
    "DiscoveredRun",
    "InferredRunLocation",
    "InvalidRunMetadata",
    "ScanResults",
    "default_datasets_root",
    "infer_run_location",
    "metadata_path_mismatches",
    "scan_runs",
]
