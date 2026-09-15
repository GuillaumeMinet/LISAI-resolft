from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .inference import ApplyOutputMode, SaveInputMode


class LocalInfrastructureConfig(BaseModel):
    """Machine-specific filesystem settings."""

    model_config = ConfigDict(extra="forbid")

    data_root: str


class LocalInferenceOutputConfig(BaseModel):
    """User-specific defaults controlling where and what `lisai apply` saves."""

    model_config = ConfigDict(extra="forbid")

    mode: ApplyOutputMode = Field(
        default="default",
        description=(
            "Default output placement for `lisai apply`: default uses inference_dir, "
            "in_place writes directly with the input data, folder_inside creates a "
            "dedicated folder inside the source folder, and folder_outside creates "
            "a dedicated folder beside the source folder."
        ),
    )
    save_input_mode: SaveInputMode = Field(
        default="if_not_in_place",
        description=(
            "Default policy for saving apply inputs: always saves them, never omits them, "
            "and if_not_in_place saves them unless predictions are written directly with "
            "the source data."
        ),
    )


class LocalInferenceConfig(BaseModel):
    """User-specific defaults for `lisai apply`."""

    model_config = ConfigDict(extra="forbid")

    output: LocalInferenceOutputConfig = Field(default_factory=LocalInferenceOutputConfig)
    inference_dir: str = Field(
        default="default",
        description=(
            "Root used by inference.output.mode=default. Use default to resolve to the "
            "project inference root, or provide an explicit local path."
        ),
    )

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
    "LocalInferenceOutputConfig",
]
