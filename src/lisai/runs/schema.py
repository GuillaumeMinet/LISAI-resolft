from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from statistics import median
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from lisai.infra.paths.model_subfolder import group_path_from_model_subfolder

from .identifiers import is_valid_run_id

RUN_METADATA_FILENAME = ".lisai_run_meta.json"
SCHEMA_VERSION = 2
RUN_STATUSES = (
    "running",
    "pause_requested",
    "paused",
    "resuming",
    "completed",
    "stopped",
    "failed",
)
RUN_TERMINAL_STATUSES = ("completed", "stopped", "failed")
RUN_NON_TERMINAL_STATUSES = ("running", "pause_requested", "paused", "resuming")
RunStatus = Literal["running", "pause_requested", "paused", "resuming", "completed", "stopped", "failed"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        dt = datetime.fromisoformat(normalized)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return dt.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_timestamp_local(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d - %H:%M")


def normalize_posix_path(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text:
        raise ValueError("Path must not be empty.")
    normalized = PurePosixPath(text).as_posix()
    if normalized == ".":
        raise ValueError("Path must not be '.'.")
    return normalized


class TrainingSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str
    train_batch_size: int = Field(default=1, ge=1)
    train_patch_size: int | list[int] | None = None
    val_patch_size: int | list[int] | None = None
    input_channels: int | None = Field(default=None, ge=1)
    upsampling_factor: int | None = Field(default=None, ge=1)
    trainable_params: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_fields(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        legacy_batch = payload.pop("batch_size", None)
        if "train_batch_size" not in payload and legacy_batch is not None:
            payload["train_batch_size"] = legacy_batch
        legacy_patch = payload.pop("patch_size", None)
        if "train_patch_size" not in payload and legacy_patch is not None:
            payload["train_patch_size"] = legacy_patch
        if "val_patch_size" not in payload and legacy_patch is not None:
            payload["val_patch_size"] = legacy_patch
        payload.pop("total_training_time_sec", None)
        payload.pop("training_time_per_epoch_sec", None)
        return payload

    @field_validator("architecture")
    @classmethod
    def _validate_architecture(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("architecture must not be empty.")
        return text

    @field_validator("train_patch_size", "val_patch_size", mode="before")
    @classmethod
    def _normalize_patch_size(cls, value):
        if isinstance(value, tuple):
            return list(value)
        return value

    @field_validator("train_patch_size", "val_patch_size")
    @classmethod
    def _validate_patch_size(cls, value: int | list[int] | None):
        if value is None:
            return None
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("patch size must be > 0.")
            return value
        if isinstance(value, list):
            if not value:
                raise ValueError("patch size list must not be empty.")
            normalized: list[int] = []
            for item in value:
                if not isinstance(item, int) or item <= 0:
                    raise ValueError("patch size list values must be integers > 0.")
                normalized.append(item)
            return normalized
        raise TypeError("patch size must be an int, a list of ints, or null.")

    @model_validator(mode="after")
    def _default_val_patch_size(self):
        if self.val_patch_size is None:
            self.val_patch_size = self.train_patch_size
        return self

    @property
    def batch_size(self) -> int:
        """Backward-compatible alias for legacy callers."""
        return int(self.train_batch_size)

    @property
    def patch_size(self) -> int | list[int] | None:
        """Backward-compatible alias for legacy callers."""
        return self.train_patch_size


class RuntimeStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peak_gpu_mem_mb: int | None = Field(default=None, ge=0)
    total_training_time_sec: float | None = Field(default=None, ge=0.0)
    training_time_per_epoch_sec: float | None = Field(default=None, ge=0.0)


class RunProvenance(BaseModel):
    """Optional origin metadata for runs imported from outside the current trainer."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["legacy_import"] = "legacy_import"
    source_path: str
    source_config: str = "config_train.json"
    imported_at: datetime = Field(default_factory=utc_now)
    notes: str | None = None

    @field_validator("source_path", "source_config")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Value must not be empty.")
        return text

    @field_validator("imported_at", mode="before")
    @classmethod
    def _parse_imported_at(cls, value: datetime | str) -> datetime:
        parsed = parse_timestamp(value)
        if parsed is None:
            raise ValueError("Timestamp must not be null.")
        return parsed

    @field_serializer("imported_at", when_used="json")
    def _serialize_imported_at(self, value: datetime) -> str:
        return format_timestamp(value)


class LiveRuntimeStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    last_epoch_duration_s: float | None = Field(default=None, ge=0.0)
    recent_epoch_durations_s: list[float] = Field(default_factory=list)
    median_epoch_duration_s: float | None = Field(default=None, ge=0.0)

    @field_validator("recent_epoch_durations_s", mode="before")
    @classmethod
    def _normalize_recent_epoch_durations(cls, value):
        if value is None:
            return []
        if isinstance(value, tuple):
            return list(value)
        return value

    @field_validator("recent_epoch_durations_s")
    @classmethod
    def _validate_recent_epoch_durations(cls, value: list[float]) -> list[float]:
        normalized: list[float] = []
        for item in value:
            duration = float(item)
            if duration < 0:
                raise ValueError("recent_epoch_durations_s values must be >= 0.")
            normalized.append(duration)
        if len(normalized) > 3:
            normalized = normalized[-3:]
        return normalized

    @model_validator(mode="after")
    def _recompute_median(self):
        if self.recent_epoch_durations_s:
            self.median_epoch_duration_s = float(median(self.recent_epoch_durations_s))
            self.last_epoch_duration_s = float(self.recent_epoch_durations_s[-1])
            return self
        if self.last_epoch_duration_s is not None:
            self.median_epoch_duration_s = float(self.last_epoch_duration_s)
        else:
            self.median_epoch_duration_s = None
        return self


class CodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_remote: str | None = None
    lisai_version: str | None = None

    @field_validator("git_commit", "git_branch", "git_remote", "lisai_version")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = Field(default=SCHEMA_VERSION)
    run_id: str
    run_name: str
    run_index: int = Field(ge=0)
    dataset: str
    model_subfolder: str
    kept: bool = False

    status: RunStatus
    closed_cleanly: bool
    created_at: datetime
    updated_at: datetime
    ended_at: datetime | None
    last_heartbeat_at: datetime
    last_epoch: int | None = None
    max_epoch: int | None = None
    best_val_loss: float | None = None
    path: str
    group_path: str | None = None
    training_signature: TrainingSignature | None = None
    runtime_stats: RuntimeStats | None = None
    live_runtime_stats: LiveRuntimeStats | None = None
    code: CodeState | None = None
    provenance: RunProvenance | None = None

    failure_reason: str | None = None
    recovery_checkpoint_filename: str | None = None
    recovery_strategy: str | None = None
    last_safe_epoch: int | None = None
    last_safe_batch_id: int | None = None
    safe_resume_fail_count: int = Field(default=0, ge=0)


    @model_validator(mode="before")
    @classmethod
    def _strip_legacy_retry_orchestration_fields(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        payload.pop("retry_attempt", None)
        payload.pop("max_retry_attempts", None)
        payload.pop("retry_state", None)
        return payload



    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {value!r}. Expected {SCHEMA_VERSION}.")
        return value

    @field_validator("run_id", mode="before")
    @classmethod
    def _normalize_run_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("run_id must be a string.")
        text = value.strip().upper()
        if not text:
            raise ValueError("run_id must not be empty.")
        if not is_valid_run_id(text):
            raise ValueError("run_id must be a ULID-like 26-character identifier.")
        return text

    @field_validator("run_name", "dataset", "model_subfolder")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Value must not be empty.")
        return text

    @field_validator("created_at", "updated_at", "last_heartbeat_at", mode="before")
    @classmethod
    def _parse_required_timestamps(cls, value: datetime | str) -> datetime:
        parsed = parse_timestamp(value)
        if parsed is None:
            raise ValueError("Timestamp must not be null.")
        return parsed

    @field_validator("ended_at", mode="before")
    @classmethod
    def _parse_optional_timestamp(cls, value: datetime | str | None) -> datetime | None:
        return parse_timestamp(value)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Path must be a string.")
        return normalize_posix_path(value)

    @field_validator("model_subfolder", "group_path", mode="before")
    @classmethod
    def _normalize_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Path value must be a string.")
        normalized = normalize_posix_path(value)
        return None if normalized in {"", "."} else normalized

    @field_validator("last_epoch")
    @classmethod
    def _validate_last_epoch(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("last_epoch must be >= 0.")
        return value

    @field_validator("max_epoch")
    @classmethod
    def _validate_max_epoch(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_epoch must be >= 0.")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self):
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at.")
        if self.last_heartbeat_at < self.created_at:
            raise ValueError("last_heartbeat_at must be >= created_at.")
        if self.ended_at is not None and self.ended_at < self.created_at:
            raise ValueError("ended_at must be >= created_at.")

        if self.status in RUN_NON_TERMINAL_STATUSES:
            if self.ended_at is not None:
                raise ValueError("ended_at must be null while status is non-terminal.")
            if self.closed_cleanly:
                raise ValueError("closed_cleanly must be false while status is non-terminal.")
        else:
            if self.status not in RUN_TERMINAL_STATUSES:
                raise ValueError(f"Unsupported terminal status: {self.status!r}.")
            if self.ended_at is None:
                raise ValueError("ended_at must be set for terminal statuses.")
            if not self.closed_cleanly:
                raise ValueError("closed_cleanly must be true for terminal statuses.")

        if self.last_epoch is not None and self.max_epoch is not None and self.last_epoch > self.max_epoch:
            raise ValueError("last_epoch must be <= max_epoch.")

        expected_group = group_path_from_model_subfolder(self.model_subfolder)
        if self.group_path != expected_group:
            raise ValueError("group_path must match model_subfolder.")

        return self

    @field_serializer("created_at", "updated_at", "last_heartbeat_at", when_used="json")
    def _serialize_required_timestamps(self, value: datetime) -> str:
        return format_timestamp(value)

    @field_serializer("ended_at", when_used="json")
    def _serialize_optional_timestamp(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_timestamp(value)


__all__ = [
    "RUN_METADATA_FILENAME",
    "RUN_NON_TERMINAL_STATUSES",
    "RUN_STATUSES",
    "RUN_TERMINAL_STATUSES",
    "SCHEMA_VERSION",
    "CodeState",
    "LiveRuntimeStats",
    "RuntimeStats",
    "TrainingSignature",
    "RunMetadata",
    "RunProvenance",
    "RunStatus",
    "format_timestamp",
    "format_timestamp_local",
    "normalize_posix_path",
    "parse_timestamp",
    "utc_now",
]
