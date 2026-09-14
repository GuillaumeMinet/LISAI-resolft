from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LocalInfrastructureConfig(BaseModel):
    """Machine-specific filesystem settings."""

    model_config = ConfigDict(extra="forbid")

    data_root: str


class LocalInferenceConfig(BaseModel):
    """User-specific defaults for where `lisai apply` writes predictions."""

    model_config = ConfigDict(extra="forbid")

    output_mode: Literal["default", "in_place"] = Field(default="default")
    inference_dir: str = Field(default="default")

    @field_validator("inference_dir", mode="before")
    @classmethod
    def _normalize_inference_dir(cls, value):
        text = str(value).strip()
        if not text:
            raise ValueError("local inference.inference_dir must not be empty.")
        return text


class LocalConfig(BaseModel):
    """Local, untracked LISAI configuration."""

    model_config = ConfigDict(extra="forbid")

    infrastructure: LocalInfrastructureConfig
    inference: LocalInferenceConfig = Field(default_factory=LocalInferenceConfig)


__all__ = [
    "LocalConfig",
    "LocalInfrastructureConfig",
    "LocalInferenceConfig",
]
