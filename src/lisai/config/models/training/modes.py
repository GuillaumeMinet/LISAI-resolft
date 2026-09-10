from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..project_config import RecoveryConfig
from .data import ExperimentDataSection
from .loading import ExperimentLoadModelSection
from .loss import LossFunctionConfig
from .model import ModelSection
from .normalization import NormalizationSection
from .sections import (
    ExperimentSection,
    InferenceSection,
    NoiseModelSection,
    RoutingSection,
    SavingSection,
    TensorboardSection,
    TrainingSection,
)
from .tasks import (
    CustomTaskSection,
    ExperimentTaskSection,
    apply_experiment_task_overrides,
    normalize_task_value,
)
from .validation import validate_cross_section_consistency


class TrainExperimentSection(ExperimentSection):
    """Experiment metadata for a fresh training run that creates a new run directory."""

    mode: Literal["train"] = Field(
        default="train",
        description="Launch a fresh training run with an explicitly configured model and dataset.",
    )


class ContinueTrainingExperimentSection(BaseModel):
    """Experiment metadata for resuming optimization inside an existing run directory."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["continue_training"] = Field(
        default="continue_training",
        description="Resume training in-place from an existing run. The original experiment name and run directory are reused automatically.",
    )
    post_training_inference: bool = Field(
        default=True,
        description="Whether to trigger automatic post-training evaluation when the resumed training completes or stops early.",
    )


class RetrainExperimentSection(BaseModel):
    """Experiment metadata for starting a new run from weights loaded from an earlier experiment."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["retrain"] = Field(
        default="retrain",
        description="Start a new run initialized from a previous checkpoint while allowing selected training, data, and normalization overrides.",
    )
    exp_name: str = Field(
        default="unnamed_experiment",
        description="Name of the new retrain run to create under the models directory.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether an existing retrain run directory with the same name may be overwritten.",
    )
    post_training_inference: bool = Field(
        default=True,
        description="Whether to trigger automatic post-training evaluation when the retrain run completes or stops early.",
    )
    task: ExperimentTaskSection = Field(
        default_factory=CustomTaskSection,
        description=(
            "Optional high-level task preset. Use 'custom' to keep the low-level "
            "data and model sections exactly as authored."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_task(cls, value):
        if isinstance(value, dict) and "task" in value:
            value = dict(value)
            value["task"] = normalize_task_value(value["task"])
        return value


class ExperimentConfig(BaseModel):
    """User-authored schema for a fresh training experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment: TrainExperimentSection = Field(
        default_factory=TrainExperimentSection,
        description="High-level metadata for a fresh training run.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Subfolder routing that determines where data, models, logs, and inference outputs live.",
    )
    data: ExperimentDataSection = Field(
        default_factory=ExperimentDataSection,
        description="User-authored data-loading, patching, and preprocessing settings.",
    )
    model: ModelSection | None = Field(
        default=None,
        description="Model architecture selection and typed constructor parameters.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Core optimization settings for the training loop.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Dataset normalization settings applied before or during data loading.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Optional model-output normalization settings kept separate from the main data normalization block.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Typed loss-function configuration used to instantiate the trainer loss.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Optional noise-model configuration or shortcut name.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Checkpoint and validation-image saving behavior.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="TensorBoard logging settings.",
    )
    inference: InferenceSection = Field(
        default_factory=InferenceSection,
        description="Inference defaults to save with the run.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_task_overrides(cls, value):
        return apply_experiment_task_overrides(value)

    @model_validator(mode="after")
    def _validate_cross_section_rules(self):
        return validate_cross_section_consistency(self, emit_warnings=False)


class ContinueTrainingConfig(BaseModel):
    """Sparse authoring schema for resuming a previous run in-place."""

    model_config = ConfigDict(extra="forbid")

    experiment: ContinueTrainingExperimentSection = Field(
        default_factory=ContinueTrainingExperimentSection,
        description="Mode-specific metadata for resuming training in the original run directory.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Optional optimization overrides to apply while continuing the original run.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Optional checkpoint and validation-image saving overrides for the resumed run.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="Optional TensorBoard logging overrides for the resumed run.",
    )
    load_model: ExperimentLoadModelSection = Field(
        ...,
        description="Required source-run selector identifying which existing run should be resumed.",
    )
    recovery: RecoveryConfig = Field(
        default_factory=RecoveryConfig,
        description="Recovery behavior for safe resume on continue_training.",
    )
    inference: InferenceSection = Field(
        default_factory=InferenceSection,
        description="Optional inference default overrides saved with the resumed run.",
    )


class RetrainConfig(BaseModel):
    """Sparse authoring schema for starting a new run from weights loaded from an earlier experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment: RetrainExperimentSection = Field(
        default_factory=RetrainExperimentSection,
        description="Mode-specific metadata for a new retrain run.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Optional routing overrides for where the new retrain run should be saved.",
    )
    data: ExperimentDataSection = Field(
        default_factory=ExperimentDataSection,
        description="Optional data overrides, including switching to a different dataset for transfer learning.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Optional optimization overrides for the retrain run.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Optional normalization overrides for the retrain run, useful when switching datasets.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Optional model-output normalization overrides for the retrain run.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Optional loss-function override for the retrain run.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Optional noise-model override for the retrain run.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Optional checkpoint and validation-image saving overrides for the retrain run.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="Optional TensorBoard logging overrides for the retrain run.",
    )
    inference: InferenceSection = Field(
        default_factory=InferenceSection,
        description="Optional inference defaults to save with the retrain run.",
    )
    load_model: ExperimentLoadModelSection = Field(
        ...,
        description="Required source-run selector identifying which existing run should provide the starting weights.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_task_overrides(cls, value):
        return apply_experiment_task_overrides(value)
