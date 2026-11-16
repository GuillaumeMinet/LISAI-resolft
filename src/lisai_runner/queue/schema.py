from __future__ import annotations

from datetime import datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from lisai.runs.identifiers import is_valid_run_id
from lisai.runs.schema import TrainingSignature, format_timestamp, parse_timestamp

QUEUE_SCHEMA_VERSION = 1
JOB_STATUSES = ("queued", "running", "blocked", "done", "failed")
JOB_PRIORITIES = ("high", "normal", "low")
RESOURCE_CLASSES = ("light", "medium", "heavy")
SELECTOR_PREFIX = "q"
SELECTOR_MIN_WIDTH = 4

_SELECTOR_RE = re.compile(r"^q(\d+)$", re.IGNORECASE)

JobStatus = Literal["queued", "running", "blocked", "done", "failed"]
JobPriority = Literal["high", "normal", "low"]
ResourceClass = Literal["light", "medium", "heavy"]


class QueueJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=QUEUE_SCHEMA_VERSION)
    job_id: str
    selector: str | None = None
    config: str
    source_config: str | None = None
    # Both roots are captured at submission; config_root remains the writable
    # experiment YAML root, not the workspace containing global settings.
    workspace_root: str | None = None
    config_root: str | None = None
    status: JobStatus
    priority: JobPriority = "normal"
    device: str
    submitted_at: datetime
    updated_at: datetime
    resource_class: ResourceClass
    run_id: str | None = None
    dataset: str | None = None
    model_subfolder: str | None = None
    run_name: str | None = None
    training_signature: TrainingSignature | None = None
    launched_at: datetime | None = None
    finished_at: datetime | None = None
    pid: int | None = Field(default=None, ge=1)
    exit_code: int | None = None
    log_path: str | None = None
    error: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != QUEUE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported queue schema_version {value!r}. Expected {QUEUE_SCHEMA_VERSION}."
            )
        return value

    @field_validator("job_id", "config", "device")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Value must not be empty.")
        return text

    @field_validator("selector", mode="before")
    @classmethod
    def _normalize_selector(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("selector must be a string.")
        text = value.strip().lower()
        if not text:
            return None
        match = _SELECTOR_RE.fullmatch(text)
        if match is None:
            raise ValueError("selector must match q<digits>, e.g. q0001.")
        number = int(match.group(1))
        if number <= 0:
            raise ValueError("selector index must be >= 1.")
        return format_queue_selector(number)

    @field_validator("source_config", "workspace_root", "config_root", "dataset", "model_subfolder", "run_name", "log_path", "error")
    @classmethod
    def _validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return None if not text else text

    @field_validator("run_id", mode="before")
    @classmethod
    def _normalize_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("run_id must be a string.")
        normalized = value.strip().upper()
        if not normalized:
            return None
        if not is_valid_run_id(normalized):
            raise ValueError("run_id must be a ULID-like 26-character identifier.")
        return normalized

    @field_validator(
        "submitted_at",
        "updated_at",
        "launched_at",
        "finished_at",
        mode="before",
    )
    @classmethod
    def _parse_timestamps(cls, value: datetime | str | None):
        return parse_timestamp(value)

    @model_validator(mode="after")
    def _validate_status_consistency(self):
        if self.workspace_root is not None:
            if self.config_root is None:
                raise ValueError("workspace_root requires config_root for two-root worker handoff")
            from pathlib import Path
            if not Path(self.workspace_root).is_absolute() or not Path(self.config_root).is_absolute():
                raise ValueError("Submitted workspace_root and config_root must be absolute paths")
        if self.submitted_at is None:
            raise ValueError("submitted_at must be set.")
        if self.updated_at is None:
            raise ValueError("updated_at must be set.")
        if self.updated_at < self.submitted_at:
            raise ValueError("updated_at must be >= submitted_at.")
        if self.launched_at is not None and self.launched_at < self.submitted_at:
            raise ValueError("launched_at must be >= submitted_at.")
        if self.finished_at is not None:
            reference = self.launched_at if self.launched_at is not None else self.submitted_at
            if self.finished_at < reference:
                raise ValueError("finished_at must be >= launched_at/submitted_at.")
        if self.status == "queued":
            if self.finished_at is not None:
                raise ValueError("queued jobs cannot have finished_at.")
        if self.status == "running":
            if self.launched_at is None:
                raise ValueError("running jobs must have launched_at.")
            if self.finished_at is not None:
                raise ValueError("running jobs cannot have finished_at.")
        if self.status == "blocked":
            if self.launched_at is not None:
                raise ValueError("blocked jobs cannot have launched_at.")
        if self.status in {"blocked", "done", "failed"}:
            if self.finished_at is None:
                raise ValueError("terminal jobs must have finished_at.")
        return self

    @field_serializer(
        "submitted_at",
        "updated_at",
        "launched_at",
        "finished_at",
        when_used="json",
    )
    def _serialize_timestamps(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_timestamp(value)


__all__ = [
    "JOB_STATUSES",
    "JOB_PRIORITIES",
    "JobPriority",
    "JobStatus",
    "QUEUE_SCHEMA_VERSION",
    "QueueJob",
    "RESOURCE_CLASSES",
    "ResourceClass",
    "SELECTOR_MIN_WIDTH",
    "SELECTOR_PREFIX",
    "format_queue_selector",
    "is_queue_selector",
    "parse_queue_selector",
]


def format_queue_selector(index: int, *, width: int = SELECTOR_MIN_WIDTH) -> str:
    if index <= 0:
        raise ValueError("Selector index must be >= 1.")
    if width <= 0:
        raise ValueError("Selector width must be >= 1.")
    return f"{SELECTOR_PREFIX}{index:0{width}d}"


def parse_queue_selector(value: str) -> int | None:
    if not isinstance(value, str):
        return None
    match = _SELECTOR_RE.fullmatch(value.strip())
    if match is None:
        return None
    number = int(match.group(1))
    return number if number > 0 else None


def is_queue_selector(value: str) -> bool:
    return parse_queue_selector(value) is not None
