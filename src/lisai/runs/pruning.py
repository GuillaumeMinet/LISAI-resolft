from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from lisai.infra.paths.run_location import RUN_ARCHIVE_DIRNAME

from .scanner import DiscoveredRun
from .schema import RUN_TERMINAL_STATUSES, utc_now


@dataclass(frozen=True)
class PrunePlan:
    matched: tuple[DiscoveredRun, ...]
    candidates: tuple[DiscoveredRun, ...]
    kept: tuple[DiscoveredRun, ...]
    non_terminal: tuple[DiscoveredRun, ...]


def build_prune_plan(runs: Iterable[DiscoveredRun]) -> PrunePlan:
    matched = tuple(runs)
    kept: list[DiscoveredRun] = []
    non_terminal: list[DiscoveredRun] = []
    candidates: list[DiscoveredRun] = []

    for run in matched:
        if run.metadata.kept:
            kept.append(run)
        elif run.metadata.status not in RUN_TERMINAL_STATUSES:
            non_terminal.append(run)
        else:
            candidates.append(run)

    return PrunePlan(
        matched=matched,
        candidates=tuple(candidates),
        kept=tuple(kept),
        non_terminal=tuple(non_terminal),
    )


def archive_run_directory(
    run_dir: str | Path,
    *,
    archived_at: datetime | None = None,
) -> Path:
    source = Path(run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {source}")

    timestamp = utc_now() if archived_at is None else archived_at
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("archived_at must be timezone-aware.")
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%SZ")

    archive_dir = source.parent / RUN_ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_archive_destination(
        archive_dir / f"{source.name}_archived_{stamp}"
    )
    source.rename(destination)
    return destination


def delete_run_directory(run_dir: str | Path) -> None:
    source = Path(run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {source}")
    shutil.rmtree(source)


def _unique_archive_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    index = 2
    while True:
        candidate = destination.with_name(f"{destination.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


__all__ = [
    "PrunePlan",
    "archive_run_directory",
    "build_prune_plan",
    "delete_run_directory",
]
