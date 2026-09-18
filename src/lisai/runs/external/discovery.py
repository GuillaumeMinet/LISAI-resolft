from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lisai.config import load_yaml, settings
from lisai.infra.paths import Paths

from .schema import EXTERNAL_RUN_METADATA_FILENAME, ExternalRunMetadata


@dataclass(frozen=True)
class DiscoveredExternalRun:
    run_dir: Path
    metadata: ExternalRunMetadata

    @property
    def dataset(self) -> str:
        return self.metadata.trained_on.dataset

    @property
    def name(self) -> str:
        return self.metadata.run_name

    @property
    def kind(self) -> str:
        return "external"


@dataclass(frozen=True)
class InvalidExternalRun:
    metadata_path: Path
    message: str


@dataclass(frozen=True)
class ExternalScanResults:
    runs: tuple[DiscoveredExternalRun, ...]
    invalid: tuple[InvalidExternalRun, ...]


def read_external_run_metadata(path: str | Path) -> ExternalRunMetadata:
    path = Path(path)
    return ExternalRunMetadata.model_validate(load_yaml(path))


def scan_external_runs(*, paths: Paths | None = None) -> ExternalScanResults:
    paths = Paths(settings) if paths is None else paths
    runs: list[DiscoveredExternalRun] = []
    invalid: list[InvalidExternalRun] = []
    root = paths.training_datasets_root()
    if not root.is_dir():
        return ExternalScanResults((), ())

    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        container = paths.dataset_external_runs_dir_from_dataset_dir(dataset_dir)
        if not container.is_dir():
            continue
        for run_dir in sorted(path for path in container.iterdir() if path.is_dir()):
            metadata_path = run_dir / EXTERNAL_RUN_METADATA_FILENAME
            if not metadata_path.is_file():
                continue
            try:
                metadata = read_external_run_metadata(metadata_path)
            except Exception as exc:  # invalid imported metadata should not break listing
                invalid.append(InvalidExternalRun(metadata_path, str(exc)))
                continue
            if metadata.trained_on.dataset != dataset_dir.name:
                invalid.append(
                    InvalidExternalRun(
                        metadata_path,
                        "trained_on.dataset does not match the containing training dataset folder.",
                    )
                )
                continue
            runs.append(DiscoveredExternalRun(run_dir.resolve(), metadata))

    runs.sort(key=lambda run: (run.dataset.casefold(), run.name.casefold()))
    return ExternalScanResults(tuple(runs), tuple(invalid))


def filter_external_runs(
    runs: Iterable[DiscoveredExternalRun],
    *,
    dataset: str | None = None,
    run_name: str | None = None,
) -> list[DiscoveredExternalRun]:
    query = None if run_name is None else run_name.strip().casefold()
    return [
        run
        for run in runs
        if (dataset is None or run.dataset == dataset)
        and (query is None or query in run.name.casefold())
    ]


__all__ = [
    "DiscoveredExternalRun",
    "ExternalScanResults",
    "InvalidExternalRun",
    "filter_external_runs",
    "read_external_run_metadata",
    "scan_external_runs",
]
