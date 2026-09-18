from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


EXTERNAL_RUN_METADATA_FILENAME = "external_run.yaml"
EXTERNAL_RUN_SCHEMA_VERSION = 1


class ExternalDatasetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    data_type: str | None = None
    input: str | None = None
    gt: str | None = None
    snr_idx: int | list[int] | Literal["last", "random"] | None = None

    @field_validator("dataset")
    @classmethod
    def _dataset_not_empty(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("dataset must not be empty.")
        return text


class ExternalRunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = EXTERNAL_RUN_SCHEMA_VERSION
    kind: Literal["external"] = "external"
    run_name: str
    trained_on: ExternalDatasetSelection
    checkpoint: str | None = None
    config: str | None = None
    uses_current_split: bool = True
    imported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = None

    @field_validator("run_name")
    @classmethod
    def _run_name_not_empty(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("run_name must not be empty.")
        return text

    @field_serializer("imported_at", when_used="json")
    def _serialize_imported_at(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "EXTERNAL_RUN_METADATA_FILENAME",
    "EXTERNAL_RUN_SCHEMA_VERSION",
    "ExternalDatasetSelection",
    "ExternalRunMetadata",
]
