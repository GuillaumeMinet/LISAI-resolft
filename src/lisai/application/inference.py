"""Host-independent model selection and sparse inference options for GUI clients.

This module does not load model weights, execute prediction, or depend on Qt.
The public LISAI ``predict`` API remains the sole inference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

from lisai.config.models.inference import (
    ApplyInferenceOverrides,
    ApplyInputOverrides,
    ApplyOverrides,
    CheckpointOverrides,
    InputNormalization,
)
from lisai.config.settings import Settings
from lisai.model_sources import RunRef


@dataclass(frozen=True)
class InferenceSelection:
    """Validated source plus *sparse* explicit overrides for ``lisai.predict``."""

    model: str | RunRef
    overrides: ApplyOverrides

    @property
    def source(self) -> str:
        return "run" if isinstance(self.model, RunRef) else "installed"

    @property
    def identifier(self) -> str:
        return self.model.run_id if isinstance(self.model, RunRef) else self.model


def _tiling_override(value: str | int | None) -> str | int | None:
    """Translate UI choices; ``None`` means use the model's saved policy."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Tiling must be model settings, auto, off, or a positive integer.")
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "model settings"}:
            return None
        if raw == "auto":
            return "auto"  # saved model default, NOT GPU memory-based autotiling
        if raw in {"off", "none"}:
            return "off"
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid tiling size {value!r}.") from exc
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Tiling size must be a positive integer, 'auto', or 'off'.")
    return value


def _optional_choice(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return None if normalized in {"", "model settings"} else normalized


def resolve_inference_selection(
    *,
    settings: Settings,
    source: str,
    identifier: str,
    checkpoint: str = "best",
    tiling_size: str | int | None = None,
    crop_size: int | None = None,
    normalization_mode: str | None = None,
    manual_gain: float | None = None,
    manual_offset: float | None = None,
    temporal_edge_mode: str | None = None,
) -> InferenceSelection:
    """Validate one workspace-bound model/run and construct sparse ``ApplyOverrides``.

    No global workspace is consulted. Catalogs are checked afresh so a stale GUI
    selection cannot silently select a different model or unavailable checkpoint.
    """
    if not isinstance(settings, Settings):
        raise TypeError("An explicitly constructed LISAI Settings instance is required.")
    if source not in {"installed", "run"}:
        raise ValueError("Model source must be 'installed' or 'run'.")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Select an installed model or completed training run first.")

    # The plugin and the future standalone app share the *same* source catalogs.
    from lisai.application.catalogs import (
        get_installed_model_names,
        list_usable_training_runs,
    )

    if source == "installed":
        model_name = identifier.strip()
        if model_name not in get_installed_model_names(settings):
            raise ValueError(f"LISAI model {model_name!r} is not installed in this workspace.")
        # The registry may outlive its local model folder; do not load weights.
        from lisai.infra.paths import Paths
        from lisai.promoted_models.registry import resolve_registered_model_dir

        resolve_registered_model_dir(model_name, paths=Paths(settings))
        model: str | RunRef = model_name
    else:
        model = RunRef(identifier)
        candidates = [
            run for run in list_usable_training_runs(settings)
            if run["run_id"].upper() == model.run_id
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Completed training run {model.run_id!r} with checkpoints is not available "
                "in this workspace."
            )
        if checkpoint not in {"best", "last"}:
            raise ValueError("Checkpoint must be 'best' or 'last'.")
        if not candidates[0]["checkpoints"].get(checkpoint, False):
            raise FileNotFoundError(
                f"Training run {model.run_id} has no {checkpoint!r} checkpoint."
            )

    inference_fields: dict = {}
    tiling = _tiling_override(tiling_size)
    if tiling is not None:
        inference_fields["tiling_size"] = tiling
    if crop_size is not None:
        if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size < 0:
            raise ValueError("crop_size must be a nonnegative integer.")
        if crop_size:
            inference_fields["crop_size"] = crop_size
    edge = _optional_choice(temporal_edge_mode)
    if edge is not None:
        if edge not in {"drop", "black_output", "black_input"}:
            raise ValueError(f"Unsupported temporal edge mode {edge!r}.")
        inference_fields["temporal_edge_mode"] = edge

    input_fields: dict = {}
    norm = _optional_choice(normalization_mode)
    if norm is not None:
        if norm not in {"training", "image", "manual"}:
            raise ValueError(f"Unsupported input normalization mode {norm!r}.")
        normalization: dict = {"mode": norm}
        if norm == "manual":
            normalization["gain"] = 1.0 if manual_gain is None else manual_gain
            normalization["offset"] = 0.0 if manual_offset is None else manual_offset
        input_fields["normalization"] = InputNormalization.model_validate(normalization)

    overrides = ApplyOverrides(
        **({"input": ApplyInputOverrides(**input_fields)} if input_fields else {}),
        **({"inference": ApplyInferenceOverrides(**inference_fields)} if inference_fields else {}),
        **({"checkpoint": CheckpointOverrides(best_or_last=checkpoint)}
           if source == "run" else {}),
    )
    return InferenceSelection(model=model, overrides=overrides)


__all__ = ["InferenceSelection", "resolve_inference_selection"]
