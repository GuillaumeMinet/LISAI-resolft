from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .tasks import CustomTaskSection, ExperimentTaskSection, normalize_task_value

Mode = Literal["train", "continue_training", "retrain"]
PositiveTilingSize: TypeAlias = Annotated[int, Field(gt=0)]
RunDefaultTilingSize: TypeAlias = PositiveTilingSize | Literal["auto", "off"] | None


class ExperimentSection(BaseModel):
    """High-level experiment metadata controlling how training is launched."""

    model_config = ConfigDict(extra="allow")

    mode: Mode = Field(
        default="train",
        description="Training lifecycle mode: start a new run, continue an existing one, or retrain from loaded weights.",
    )
    exp_name: str = Field(
        default="unnamed_experiment",
        description="Experiment name used to create the output run directory and derived artifact names.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether an existing experiment directory with the same name may be overwritten.",
    )
    post_training_inference: bool = Field(
        default=True,
        description="Whether to trigger automatic post-training evaluation when training completes or stops early.",
    )
    task: ExperimentTaskSection = Field(
        default_factory=CustomTaskSection,
        description=(
            "Optional high-level task preset. Use 'custom' to keep the low-level "
            "data and model sections exactly as authored."
        ),
    )

    @field_validator("task", mode="before")
    @classmethod
    def _normalize_task(cls, value):
        return normalize_task_value(value)


class ResolvedExperimentSection(ExperimentSection):
    """Resolved experiment metadata enriched with runtime-only bookkeeping."""

    origin_run_dir: Optional[str] = Field(
        default=None,
        description="Original run directory used as the source when continuing training or retraining.",
    )


class RoutingSection(BaseModel):
    """Subfolder routing used to place datasets, checkpoints, logs, and inference outputs."""

    model_config = ConfigDict(extra="allow")

    data_subfolder: str = Field(
        default="",
        description="Dataset subfolder selected under the configured data root.",
    )
    models_subfolder: str = Field(
        default="",
        description="Subfolder under the models root where checkpoints and training artifacts are stored.",
    )
    tensorboard_subfolder: Optional[str] = Field(
        default=None,
        description="Subfolder under the tensorboard root. Defaults to the models_subfolder when omitted.",
    )
    inference_subfolder: str = Field(
        default="",
        description="Subfolder under the inference root where evaluation or apply outputs are written.",
    )

    @model_validator(mode="after")
    def _defaults(self):
        if self.tensorboard_subfolder is None:
            self.tensorboard_subfolder = self.models_subfolder
        return self


class InferenceSection(BaseModel):
    """Inference defaults saved with a training run."""

    model_config = ConfigDict(extra="forbid")

    default_tiling_size: RunDefaultTilingSize = Field(
        default=None,
        description=(
            "Optional saved default for inference tiling. Use null or 'auto' to "
            "fall back to the architecture default, a positive integer to force "
            "a run-specific tile size, or 'off' to disable tiling by default."
        ),
    )

    @field_validator("default_tiling_size", mode="before")
    @classmethod
    def _normalize_default_tiling_size(cls, value):
        if value is False:
            return "off"
        if value is True:
            raise ValueError(
                "default_tiling_size must be a positive integer, 'auto', 'off', or null."
            )
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "auto"}:
                return "auto"
            if normalized in {"off", "none", "disable", "disabled"}:
                return "off"
            return normalized
        return value


class TrainingSection(BaseModel):
    """Core optimization settings for the training loop."""

    model_config = ConfigDict(extra="allow")

    n_epochs: int = Field(
        default=1,
        description="Maximum number of training epochs to run.",
    )
    batch_size: int = Field(
        default=1,
        description="Fallback batch size used by the trainer when the data section does not provide one.",
    )
    learning_rate: float = Field(
        default=1e-4,
        description="Initial learning rate passed to the optimizer.",
    )
    optimizer: str = Field(
        default="Adam",
        description="Optimizer name used to update the model parameters.",
    )
    scheduler: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional learning-rate scheduler. "
            "Use a string name (legacy) or an object with `name` plus scheduler kwargs."
        ),
    )
    progress_bar: bool = Field(
        default=False,
        description="Whether to show a live progress bar during training iterations.",
    )
    early_stop: bool = Field(
        default=False,
        description=(
            "Legacy debug-stop flag. Keeps historical truncated-training behavior for backward compatibility."
        ),
    )
    debug_stop: bool = Field(
        default=False,
        description="Debug helper: when true, stop after 3 full epochs.",
    )
    pos_encod: bool = Field(
        default=False,
        description="Whether positional encoding should be enabled for models that support it.",
    )
    max_grad_norm: float | None = Field(
        default=None,
        description="To clip grad norm value. Set to None to disable gradient clipping.",
    )
    val_loss_patience: int | None = Field(
        default=None,
        ge=0,
        deprecated=True,
        description=(
            "Deprecated and ignored. Use `training.auto_stop.patience` instead."
        ),
    )
    warmup: "WarmupSection" = Field(
        default_factory=lambda: WarmupSection(),
        description=(
            "Optional optimizer-step warmup. Active only when scheduler is ReduceLROnPlateau."
        ),
    )
    auto_stop: "AutoStopSection" = Field(
        default_factory=lambda: AutoStopSection(),
        description="Optional metric-based automatic stopping.",
    )

class WarmupSection(BaseModel):
    """Manual optimizer-step warmup settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable optimizer-step LR warmup.",
    )
    steps: int = Field(
        default=500,
        ge=0,
        description="Number of optimizer steps used for warmup.",
    )
    start_factor: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Initial LR factor relative to base LR at warmup start.",
    )


class AutoStopSection(BaseModel):
    """Metric-based automatic stopping."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable metric-based auto-stop.",
    )
    metrics: Literal["loss", "val_loss"] = Field(
        default="val_loss",
        description="Metric monitored for auto-stop decisions.",
    )
    patience: int = Field(
        default=30,
        ge=0,
        description="Number of non-improving epochs tolerated before stop.",
    )

class SavingSection(BaseModel):
    """Checkpoint and validation-image saving behavior."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Whether checkpoints and other training artifacts should be written to disk.",
    )
    canonical_save: bool = Field(
        default=True,
        description="Whether checkpoint/output paths should be resolved through the canonical project routing rules.",
    )
    validation_images: bool = Field(
        default=True,
        description="Whether to save validation image previews during training.",
    )
    validation_freq: int = Field(
        default=10,
        description="Number of epochs between validation-image saves.",
    )
    state_dict: bool = Field(
        default=False,
        description="Whether checkpoints should be saved as state_dict files.",
    )
    entire_model: bool = Field(
        default=False,
        description="Whether checkpoints should be saved as serialized full-model files.",
    )
    overwrite_best: bool = Field(
        default=True,
        description="Whether the current best checkpoint may overwrite the previously saved best checkpoint.",
    )


class TensorboardSection(BaseModel):
    """TensorBoard logging settings."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=False,
        description="Whether TensorBoard summaries should be written during training.",
    )


class NoiseModelSection(BaseModel):
    """Optional auxiliary noise-model configuration."""

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(
        default=None,
        description="Registered noise-model name to attach to the experiment. Use null to disable it.",
    )
