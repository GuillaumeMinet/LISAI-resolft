from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from lisai.config.models.training import TaskName

PROMOTED_MODEL_SCHEMA_VERSION = 1
PROMOTED_MODEL_REGISTRY_SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    return text


def _public_identifier(value: str, *, field_name: str) -> str:
    text = _clean_text(value, field_name=field_name)
    if "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must not contain path separators.")
    if text in {".", ".."}:
        raise ValueError(f"{field_name} must not be '.' or '..'.")
    return text


def _relative_package_path(value: str) -> str:
    text = str(value).replace("\\", "/").strip()
    if not text:
        raise ValueError("Package artifact path must not be empty.")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Package artifact paths must be relative and must not contain '..'.")
    normalized = path.as_posix()
    if normalized == ".":
        raise ValueError("Package artifact path must not be '.'.")
    return normalized


class PromotedModelCodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_remote: str | None = None
    lisai_version: str | None = None


class PromotedModelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_name: str
    run_status: Literal["completed", "stopped"]
    checkpoint_selector: Literal["best", "last"]
    checkpoint_filename: str
    code: PromotedModelCodeState | None = None

    @field_validator("run_id", "run_name", "checkpoint_filename")
    @classmethod
    def _validate_required_text(cls, value: str, info):
        return _clean_text(value, field_name=info.field_name)


class PromotedModelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: TaskName
    architecture: str

    @field_validator("task", "architecture")
    @classmethod
    def _validate_required_text(cls, value: str, info):
        return _clean_text(value, field_name=info.field_name)


class PromotedTrainingData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str

    @field_validator("dataset")
    @classmethod
    def _validate_dataset(cls, value: str) -> str:
        return _clean_text(value, field_name="dataset")


class PromotedTrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_val_loss: float | None = None
    last_epoch: int | None = Field(default=None, ge=0)
    max_epoch: int | None = Field(default=None, ge=0)
    training_signature: dict[str, Any] | None = None
    runtime_stats: dict[str, Any] | None = None


class PromotedNoiseModelArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    norm_prm: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _clean_text(value, field_name="name")

    @field_validator("model", "norm_prm")
    @classmethod
    def _validate_package_path(cls, value: str) -> str:
        return _relative_package_path(value)


class PromotedModelArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str = "config_train.yaml"
    inference_config: str | None = None
    weights: str = "weights.pt"
    loss: str | None = None
    loss_plot: str | None = None
    noise_model: PromotedNoiseModelArtifacts | None = None

    @field_validator("config", "inference_config", "weights", "loss", "loss_plot")
    @classmethod
    def _validate_package_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _relative_package_path(value)


class PromotedModelManifest(BaseModel):
    """Machine-readable identity and provenance for one promoted LISAI model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=PROMOTED_MODEL_SCHEMA_VERSION)
    name: str
    created_at: datetime = Field(default_factory=_utc_now)
    model: PromotedModelDefinition
    training_data: PromotedTrainingData
    source: PromotedModelSource
    training: PromotedTrainingSummary = Field(default_factory=PromotedTrainingSummary)
    artifacts: PromotedModelArtifacts = Field(default_factory=PromotedModelArtifacts)
    checksums: dict[str, str] = Field(default_factory=dict)

    @field_validator("checksums")
    @classmethod
    def _validate_checksums(cls, value: dict[str, str]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for package_path, digest in value.items():
            normalized_path = _relative_package_path(package_path)
            normalized_digest = str(digest).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", normalized_digest):
                raise ValueError(
                    f"Checksum for {normalized_path!r} must be a 64-character SHA256 hex digest."
                )
            validated[normalized_path] = normalized_digest
        return validated

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != PROMOTED_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported promoted-model schema_version {value!r}. "
                f"Expected {PROMOTED_MODEL_SCHEMA_VERSION}."
            )
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _public_identifier(value, field_name="name")

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_created_at(cls, value: datetime | str) -> datetime:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = f"{normalized[:-1]}+00:00"
            value = datetime.fromisoformat(normalized)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @field_serializer("created_at", when_used="json")
    def _serialize_created_at(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class PromotedModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_run_id: str
    path: str
    created_at: datetime
    origin: Literal["promoted", "installed"] = "promoted"
    installed_at: datetime | None = None
    task: TaskName | None = None

    @field_validator("source_run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        return _clean_text(value, field_name="source_run_id")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _relative_package_path(value)

    @field_validator("created_at", "installed_at", mode="before")
    @classmethod
    def _parse_registry_datetime(cls, value: datetime | str | None, info) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = f"{normalized[:-1]}+00:00"
            value = datetime.fromisoformat(normalized)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(timezone.utc)

    @field_serializer("created_at", "installed_at", when_used="json")
    def _serialize_registry_datetime(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class PromotedModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PROMOTED_MODEL_REGISTRY_SCHEMA_VERSION
    models: dict[str, PromotedModelRegistryEntry] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != PROMOTED_MODEL_REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported promoted-model registry schema_version {value!r}. "
                f"Expected {PROMOTED_MODEL_REGISTRY_SCHEMA_VERSION}."
            )
        return value

    @field_validator("models", mode="before")
    @classmethod
    def _normalize_empty_models(cls, value):
        # ``models:`` in YAML is parsed as ``None``. Treat that the same as an
        # omitted/empty mapping so a manually emptied registry remains usable.
        return {} if value is None else value

    @field_validator("models")
    @classmethod
    def _validate_model_names(
        cls, value: dict[str, PromotedModelRegistryEntry]
    ) -> dict[str, PromotedModelRegistryEntry]:
        return {
            _public_identifier(name, field_name="model name"): entry
            for name, entry in value.items()
        }


__all__ = [
    "PROMOTED_MODEL_REGISTRY_SCHEMA_VERSION",
    "PROMOTED_MODEL_SCHEMA_VERSION",
    "PromotedModelArtifacts",
    "PromotedModelCodeState",
    "PromotedModelDefinition",
    "PromotedModelManifest",
    "PromotedModelRegistry",
    "PromotedModelRegistryEntry",
    "PromotedModelSource",
    "PromotedNoiseModelArtifacts",
    "PromotedTrainingData",
    "PromotedTrainingSummary",
]
