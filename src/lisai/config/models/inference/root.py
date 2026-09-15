from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .sections import ApplyDefaults, ApplyOverrides, EvaluateDefaults, EvaluateOverrides


class ResolvedInferenceConfig(BaseModel):
    """Fully resolved inference settings used at runtime.

    This object defines the canonical LISAI inference defaults in the same
    nested layout used by inference YAML files. Runtime apply/evaluate options
    are flattened by the inference resolver.
    """

    model_config = ConfigDict(extra="forbid")

    apply: ApplyDefaults = Field(
        default_factory=ApplyDefaults,
        description="Fully resolved settings used by the `lisai apply` flow.",
    )
    evaluate: EvaluateDefaults = Field(
        default_factory=EvaluateDefaults,
        description="Fully resolved settings used by the `lisai evaluate` flow.",
    )


class InferenceOverrides(BaseModel):
    """Sparse user-authored inference YAML overrides.

    Any omitted section or field means "leave the resolved default as-is".
    Local configs may stay sparse; the resolver separately enforces completeness
    for standalone configs outside `configs/inference/local/`.
    """

    model_config = ConfigDict(extra="forbid")

    apply: ApplyOverrides | None = Field(
        default=None,
        description="Optional overrides for the `lisai apply` flow.",
    )
    evaluate: EvaluateOverrides | None = Field(
        default=None,
        description="Optional overrides for the `lisai evaluate` flow.",
    )


# Backward-compatible aliases kept during the inference model naming cleanup.
InferenceDefaults = ResolvedInferenceConfig
InferenceConfig = InferenceOverrides


__all__ = [
    "InferenceOverrides",
    "ResolvedInferenceConfig",
    "InferenceConfig",
    "InferenceDefaults",
]
