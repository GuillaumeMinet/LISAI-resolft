from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class LocalConsoleConfig(BaseModel):
    """Machine-local console display preferences."""

    model_config = ConfigDict(extra="forbid")

    progress_bar: bool | None = Field(
        default=None,
        description=(
            "Machine-local default for live progress bars. Null preserves each "
            "command's normal default; true/false enables or disables tqdm when "
            "the CLI does not specify --progress-bar or --no-progress-bar."
        ),
    )


class LocalConfig(BaseModel):
    """Effective runtime preferences; never owns the workspace location."""

    model_config = ConfigDict(extra="forbid")

    inference: LocalInferenceConfig = Field(default_factory=LocalInferenceConfig)
    console: LocalConsoleConfig = Field(default_factory=LocalConsoleConfig)


class LegacyLocalConfig(LocalConfig):
    """Read old local_config.yml only through explicit migration/testing loading."""

    infrastructure: LocalInfrastructureConfig


class DeveloperModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(default=False, strict=True)


class DeveloperWorkspaceSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str | None = None

    @field_validator("root")
    @classmethod
    def _absolute_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("workspace.root in local.yml must be absolute")
        return str(path.resolve())


class DeveloperConfigSelector(BaseModel):
    """Persisted experiment-config ownership; independent of workspace YAMLs."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["repo", "workspace", "custom"]
    root: str | None = None

    @model_validator(mode="after")
    def _validate_root(self):
        if self.mode == "custom":
            if not self.root or not self.root.strip():
                raise ValueError("config.root is required when config.mode is custom")
            path = Path(self.root).expanduser()
            if not path.is_absolute():
                raise ValueError("config.root must be an absolute path")
            self.root = str(path.resolve())
        elif self.root is not None:
            raise ValueError("config.root is only valid when config.mode is custom")
        return self


class DeveloperLocalConfig(LocalConfig):
    """Strict external local.yml: a selector plus whitelisted preferences."""

    dev: DeveloperModeConfig = Field(default_factory=DeveloperModeConfig)
    workspace: DeveloperWorkspaceSelector = Field(default_factory=DeveloperWorkspaceSelector)
    config: DeveloperConfigSelector | None = None

    # The developer file may deliberately be incomplete after `dev init`.
    # CLI selection checks readiness through `developer_setup_status`; ordinary
    # commands must not treat an incomplete developer setup as standalone.


__all__ = [
    "LocalConfig",
    "LocalInfrastructureConfig",
    "LegacyLocalConfig",
    "DeveloperLocalConfig",
    "DeveloperConfigSelector",
    "LocalInferenceConfig",
    "LocalInferenceOutputConfig",
    "LocalConsoleConfig",
]
