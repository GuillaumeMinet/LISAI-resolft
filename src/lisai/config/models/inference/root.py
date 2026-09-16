from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .sections import ApplyDefaults, ApplyOverrides, EvaluateDefaults, EvaluateOverrides


class ResolvedInferenceConfig(BaseModel):
    """Fully resolved inference settings used at runtime.

    This object defines the canonical LISAI inference defaults in the same
    nested layout used by inference YAML files. Apply/evaluate runtime code is
    expected to consume these typed nested sections directly.
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
    All inference configs may stay sparse: the resolver layers them over the
    local inference defaults, while explicitly authored values always win.
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
