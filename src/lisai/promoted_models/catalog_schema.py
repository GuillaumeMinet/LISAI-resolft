from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lisai.config.models.training import TaskName


DOWNLOADABLE_MODELS_CATALOG_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _clean_text(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


class ZenodoDownloadSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["zenodo"]
    record_id: str
    filename: str

    @field_validator("record_id")
    @classmethod
    def _validate_record_id(cls, value: str) -> str:
        return _clean_text(value, field_name="record_id")

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        filename = _clean_text(value, field_name="filename")
        if "/" in filename or "\\" in filename:
            raise ValueError("filename must be a file name, not a path.")
        if not filename.endswith(".lisai.zip"):
            raise ValueError("filename must use the '.lisai.zip' suffix.")
        return filename


class DownloadableModelDownload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    source: ZenodoDownloadSource | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = _clean_text(value, field_name="url")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute HTTP(S) URL.")
        return url

    @model_validator(mode="after")
    def _validate_exactly_one_location(self):
        if (self.url is None) == (self.source is None):
            raise ValueError("download must define exactly one of 'url' or 'source'.")
        return self


class DownloadableModelCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: TaskName
    description: str
    archive_sha256: str
    download: DownloadableModelDownload

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        return _clean_text(value, field_name="description")

    @field_validator("archive_sha256")
    @classmethod
    def _validate_archive_sha256(cls, value: str) -> str:
        digest = _clean_text(value, field_name="archive_sha256").lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("archive_sha256 must be a 64-character SHA256 hex digest.")
        return digest


class DownloadableModelCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    models: dict[str, DownloadableModelCatalogEntry] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != DOWNLOADABLE_MODELS_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported downloadable-model catalog schema_version {value!r}. "
                f"Expected {DOWNLOADABLE_MODELS_CATALOG_SCHEMA_VERSION}."
            )
        return value

    @field_validator("models", mode="before")
    @classmethod
    def _normalize_empty_models(cls, value):
        return {} if value is None else value

    @field_validator("models")
    @classmethod
    def _validate_model_names(
        cls,
        value: dict[str, DownloadableModelCatalogEntry],
    ) -> dict[str, DownloadableModelCatalogEntry]:
        normalized: dict[str, DownloadableModelCatalogEntry] = {}
        for raw_name, entry in value.items():
            name = _clean_text(raw_name, field_name="model name")
            if "/" in name or "\\" in name or name in {".", ".."}:
                raise ValueError("model name must not contain path separators or be '.' or '..'.")
            normalized[name] = entry
        return normalized


__all__ = [
    "DOWNLOADABLE_MODELS_CATALOG_SCHEMA_VERSION",
    "DownloadableModelCatalog",
    "DownloadableModelCatalogEntry",
    "DownloadableModelDownload",
    "ZenodoDownloadSource",
]
