from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lisai.promoted_models.sources.zenodo import (
    ZenodoSourceError,
    resolve_record_file_download_url,
)


DATASET_CATALOG_SCHEMA_VERSION = 1
DEFAULT_DATASET_RECORD_ID = "22877255"
DEFAULT_DATASET_CATALOG_FILENAME = "dataset_catalog.json"
DEFAULT_CATALOG_TIMEOUT_SECONDS = 30.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DatasetCatalogUnavailableError(RuntimeError):
    """Raised when the downloadable-dataset catalog cannot be retrieved."""


def _clean_name(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"{field_name} must be a simple name, not a path.")
    return text


class DatasetArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    size_bytes: int = Field(ge=0)
    sha256: str

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        filename = _clean_name(value, field_name="archive filename")
        if not filename.lower().endswith(".zip"):
            raise ValueError("archive filename must end with '.zip'.")
        return filename

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        digest = str(value).strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("sha256 must be a 64-character SHA256 hex digest.")
        return digest


class DatasetDependencies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_datasets: list[str] = Field(default_factory=list)
    noise_models: list[str] = Field(default_factory=list)

    @field_validator("evaluation_datasets", "noise_models")
    @classmethod
    def _validate_names(cls, values: list[str]) -> list[str]:
        return [_clean_name(value, field_name="dependency name") for value in values]


class DownloadableDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usage: str
    install_path: str
    archives: list[DatasetArchive]
    registry_entry: dict[str, Any]
    dependencies: DatasetDependencies = Field(default_factory=DatasetDependencies)

    @field_validator("usage")
    @classmethod
    def _validate_usage(cls, value: str) -> str:
        usage = str(value).strip().lower()
        if usage not in {"training", "evaluation"}:
            raise ValueError("usage must be 'training' or 'evaluation'.")
        return usage

    @field_validator("install_path")
    @classmethod
    def _validate_install_path(cls, value: str) -> str:
        text = str(value).strip().replace("\\", "/")
        if text not in {"datasets/training", "datasets/evaluation"}:
            raise ValueError(
                "install_path must be 'datasets/training' or 'datasets/evaluation'."
            )
        return text

    @field_validator("archives")
    @classmethod
    def _validate_archives(cls, value: list[DatasetArchive]) -> list[DatasetArchive]:
        if not value:
            raise ValueError("at least one archive is required.")
        filenames = [archive.filename for archive in value]
        if len(set(filenames)) != len(filenames):
            raise ValueError("archive filenames must be unique within a dataset.")
        return value


class NoiseModelBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    archive: DatasetArchive
    available: list[str] = Field(default_factory=list)

    @field_validator("available")
    @classmethod
    def _validate_available(cls, values: list[str]) -> list[str]:
        return [_clean_name(value, field_name="noise model name") for value in values]


class SharedDownloads(BaseModel):
    model_config = ConfigDict(extra="forbid")

    noise_models: NoiseModelBundle | None = None


class DatasetCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    datasets: dict[str, DownloadableDataset] = Field(default_factory=dict)
    shared: SharedDownloads | None = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != DATASET_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset catalog schema_version {value!r}. "
                f"Expected {DATASET_CATALOG_SCHEMA_VERSION}."
            )
        return value

    @field_validator("datasets")
    @classmethod
    def _validate_dataset_names(
        cls,
        value: dict[str, DownloadableDataset],
    ) -> dict[str, DownloadableDataset]:
        normalized: dict[str, DownloadableDataset] = {}
        for raw_name, entry in value.items():
            name = _clean_name(raw_name, field_name="dataset name")
            expected_path = f"datasets/{entry.usage}"
            if entry.install_path != expected_path:
                raise ValueError(
                    f"Dataset {name!r} has install_path {entry.install_path!r}, "
                    f"expected {expected_path!r} for usage {entry.usage!r}."
                )
            normalized[name] = entry
        return normalized


def _fetch_catalog_bytes(url: str, *, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "LISAI"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (HTTPError, URLError, OSError) as exc:
        raise DatasetCatalogUnavailableError(
            f"Could not retrieve the downloadable-dataset catalog from {url}: {exc}"
        ) from exc


def load_catalog(
    *,
    record_id: str = DEFAULT_DATASET_RECORD_ID,
    catalog_url: str | None = None,
    timeout: float = DEFAULT_CATALOG_TIMEOUT_SECONDS,
) -> DatasetCatalog:
    """Fetch and validate the downloadable-dataset catalog."""
    if catalog_url is None:
        try:
            catalog_url = resolve_record_file_download_url(
                record_id,
                DEFAULT_DATASET_CATALOG_FILENAME,
                timeout=timeout,
            )
        except ZenodoSourceError as exc:
            raise DatasetCatalogUnavailableError(str(exc)) from exc

    raw = _fetch_catalog_bytes(catalog_url, timeout=timeout)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid downloadable-dataset catalog at {catalog_url}: {exc}") from exc

    try:
        return DatasetCatalog.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid downloadable-dataset catalog at {catalog_url}: {exc}") from exc


def dataset_download_size(dataset: DownloadableDataset) -> int:
    return sum(archive.size_bytes for archive in dataset.archives)


def list_datasets(
    *,
    record_id: str = DEFAULT_DATASET_RECORD_ID,
    catalog_url: str | None = None,
    timeout: float = DEFAULT_CATALOG_TIMEOUT_SECONDS,
) -> list[tuple[str, DownloadableDataset]]:
    catalog = load_catalog(record_id=record_id, catalog_url=catalog_url, timeout=timeout)
    return [(name, catalog.datasets[name]) for name in sorted(catalog.datasets, key=str.casefold)]


__all__ = [
    "DATASET_CATALOG_SCHEMA_VERSION",
    "DEFAULT_CATALOG_TIMEOUT_SECONDS",
    "DEFAULT_DATASET_CATALOG_FILENAME",
    "DEFAULT_DATASET_RECORD_ID",
    "DatasetArchive",
    "DatasetCatalog",
    "DatasetCatalogUnavailableError",
    "DatasetDependencies",
    "DownloadableDataset",
    "NoiseModelBundle",
    "SharedDownloads",
    "dataset_download_size",
    "list_datasets",
    "load_catalog",
]
