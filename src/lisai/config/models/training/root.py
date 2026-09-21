from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..project_config import RecoveryConfig
from .data import DataSection
from .loading import LoadModelSection
from .loss import LossFunctionConfig
from .model import ModelSection
from .normalization import NormalizationSection
from .sections import (
    InferenceSection,
    NoiseModelSection,
    ResolvedExperimentSection,
    RoutingSection,
    SavingSection,
    TensorboardSection,
    TrainingSection,
)
from .tasks import apply_experiment_task_overrides
from .validation import validate_cross_section_consistency


class ResolvedExperiment(BaseModel):
    """Runtime training config after merge and normalization."""

    model_config = ConfigDict(extra="allow")

    experiment: ResolvedExperimentSection = Field(
        default_factory=ResolvedExperimentSection,
        description="Resolved experiment metadata enriched with runtime-only bookkeeping.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Resolved routing configuration for datasets, models, logs, and inference outputs.",
    )
    data: DataSection = Field(
        default_factory=DataSection,
        description="Resolved data-loading and preprocessing settings used by the runtime.",
    )
    model: ModelSection | None = Field(
        default=None,
        description="Resolved model architecture selection and typed constructor parameters.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Resolved optimization settings for the training loop.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Resolved dataset normalization settings used during training data preparation.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Resolved model-output normalization settings.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Resolved typed loss-function configuration.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Resolved optional noise-model configuration or shortcut name.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Resolved checkpoint and artifact-saving behavior.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="Resolved TensorBoard logging settings.",
    )
    load_model: LoadModelSection = Field(
        default_factory=LoadModelSection,
        description="Resolved source-run and checkpoint selection used when loading a prior model.",
    )
    recovery: RecoveryConfig = Field(
        default_factory=RecoveryConfig,
        description="Resolved recovery behavior for continue/restart flows.",
    )
    inference: InferenceSection = Field(
        default_factory=InferenceSection,
        description="Inference defaults saved with this training run.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_task_overrides(cls, value):
        return apply_experiment_task_overrides(value)

    @model_validator(mode="after")
    def _validate_cross_section_rules(self):
        return validate_cross_section_consistency(self, emit_warnings=True)
