from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from lisai.config import load_yaml
from lisai.config.io import deep_merge
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.models.inference import (
    ApplyDefaults,
    ApplyOutputMode,
    ApplyOverrides,
    EvaluateDefaults,
    EvaluateOverrides,
    InferenceOverrides,
    ResolvedInferenceConfig,
)
from lisai.config.models.inference.presets import POST_TRAINING_OVERRIDES

inference_config_paths = ConfigPathResolver("inference")
POST_TRAINING_CONFIG_NAME = "post_training"

# Backward-compatible aliases kept while the clearer inference model names settle in.
InferenceConfig = InferenceOverrides
InferenceDefaults = ResolvedInferenceConfig


@dataclass(frozen=True)
class ApplyOutputPolicy:
    """Resolved destination policy for one `apply` invocation."""

    mode: ApplyOutputMode | Literal["folder"]
    save_folder: Path | None = None


def resolve_inference_config_path(config_arg: str | Path | None) -> Path | None:
    return inference_config_paths.resolve(config_arg)


def load_inference_config(
    config_arg: str | Path | None = None,
) -> tuple[InferenceOverrides, Path | None]:
    cfg_path = resolve_inference_config_path(config_arg)
    if cfg_path is None:
        return InferenceOverrides(), None
    return InferenceOverrides.model_validate(load_yaml(cfg_path)), cfg_path


def _section_overrides(
    cfg: InferenceOverrides,
    section: Literal["apply", "evaluate"],
) -> dict[str, Any] | None:
    raw = cfg.model_dump(exclude_unset=True)
    section_raw = raw.get(section)
    return None if section_raw is None else dict(section_raw)


def _is_post_training_name(config: str | Path | None, path: Path | None = None) -> bool:
    if path is not None and path.stem == POST_TRAINING_CONFIG_NAME:
        return True
    if config is None:
        return False
    config_path = Path(config)
    return config_path.parent == Path(".") and config_path.stem == POST_TRAINING_CONFIG_NAME


def _merge_section_config(
    section: Literal["apply", "evaluate"],
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    base = deepcopy(base)
    if section == "apply":
        saving = overrides.get("saving")
        if isinstance(saving, Mapping) and any(
            key in saving for key in ("mode", "save_folder", "in_place")
        ):
            base_saving = dict(base.get("saving") or {})
            base_saving.update({"mode": None, "save_folder": None, "in_place": None})
            base["saving"] = base_saving
    return deep_merge(base, overrides)


def _validate_resolved_section(
    section: Literal["apply", "evaluate"],
    value: dict[str, Any],
) -> ApplyDefaults | EvaluateDefaults:
    if section == "apply":
        return ApplyDefaults.model_validate(value)
    return EvaluateDefaults.model_validate(value)


def _load_local_defaults_section(section: Literal["apply", "evaluate"]) -> dict[str, Any] | None:
    cfg, cfg_path = load_inference_config(None)
    if cfg_path is None:
        return None
    return _section_overrides(cfg, section)


def _load_selected_config(
    config: str | Path,
) -> tuple[InferenceOverrides | None, Path | None]:
    try:
        return load_inference_config(config)
    except FileNotFoundError:
        if _is_post_training_name(config):
            return None, None
        raise


def _resolve_section_nested(
    section: Literal["apply", "evaluate"],
    *,
    config: str | Path | None = None,
) -> ApplyDefaults | EvaluateDefaults:
    # Base config precedence is intentionally one-way:
    # canonical typed defaults < local/defaults.yml < built-in preset (if any)
    # < selected config. Typed invocation overrides are layered by the public
    # section resolvers below. A selected config may therefore stay sparse;
    # local defaults only fill values that it does not explicitly provide.
    resolved_section = deepcopy(ResolvedInferenceConfig().model_dump()[section])

    local_defaults = _load_local_defaults_section(section)
    if local_defaults:
        resolved_section = _merge_section_config(section, resolved_section, local_defaults)

    if config is None:
        return _validate_resolved_section(section, resolved_section)

    selected_cfg, cfg_path = _load_selected_config(config)

    if cfg_path is None:
        # Built-in post-training preset remains usable even if the local file was removed.
        preset_cfg = InferenceOverrides.model_validate(POST_TRAINING_OVERRIDES)
        preset_section = _section_overrides(preset_cfg, section)
        if preset_section is None:
            raise ValueError(f"Built-in post-training preset does not define a '{section}' section.")
        resolved_section = _merge_section_config(section, resolved_section, preset_section)
        return _validate_resolved_section(section, resolved_section)

    selected_section = _section_overrides(selected_cfg, section) if selected_cfg is not None else None

    if _is_post_training_name(config, cfg_path):
        preset_cfg = InferenceOverrides.model_validate(POST_TRAINING_OVERRIDES)
        preset_section = _section_overrides(preset_cfg, section)
        if preset_section:
            resolved_section = _merge_section_config(section, resolved_section, preset_section)

    if selected_section is None:
        raise ValueError(f"Inference config '{cfg_path}' does not define a '{section}' section.")

    resolved_section = _merge_section_config(section, resolved_section, selected_section)
    return _validate_resolved_section(section, resolved_section)


def _finalize_apply_local_fallbacks(
    resolved: ApplyDefaults,
    *,
    stg=None,
) -> ApplyDefaults:
    """Fill apply saving policies that intentionally fall back to local_config.

    Inference YAML owns model/workflow-specific saving overrides, while
    local_config owns the machine-local baseline for output routing and input
    saving. This finalization happens only after all inference-config and typed
    invocation overrides have been merged, so any explicit inference choice
    keeps priority over the local fallback.
    """
    if stg is None:
        from lisai.config.settings import settings as stg

    values = resolved.model_dump()
    saving = values["saving"]
    if all(saving[key] is None for key in ("mode", "save_folder", "in_place")):
        saving["mode"] = stg.INFERENCE_OUTPUT_MODE
    if saving["save_input_mode"] is None:
        saving["save_input_mode"] = stg.INFERENCE_SAVE_INPUT_MODE
    return ApplyDefaults.model_validate(values)


def resolve_apply_config(
    *,
    config: str | Path | None = None,
    overrides: ApplyOverrides | None = None,
    stg=None,
) -> ApplyDefaults:
    """Resolve one apply invocation to the complete typed runtime config."""
    resolved = _resolve_section_nested("apply", config=config)
    if not isinstance(resolved, ApplyDefaults):
        raise TypeError("Internal error: apply resolution did not produce ApplyDefaults.")

    if overrides is not None:
        override_values = overrides.model_dump(exclude_unset=True)
        if override_values:
            merged = _merge_section_config("apply", resolved.model_dump(), override_values)
            validated = _validate_resolved_section("apply", merged)
            if not isinstance(validated, ApplyDefaults):
                raise TypeError("Internal error: apply overrides did not produce ApplyDefaults.")
            resolved = validated

    return _finalize_apply_local_fallbacks(resolved, stg=stg)


def resolve_apply_output_policy(cfg: ApplyDefaults) -> ApplyOutputPolicy:
    """Interpret the already-resolved apply saving route."""
    saving = cfg.saving
    if saving.save_folder is not None:
        return ApplyOutputPolicy(mode="folder", save_folder=Path(saving.save_folder))
    if saving.mode is not None:
        return ApplyOutputPolicy(mode=saving.mode)
    if saving.in_place is not None:
        return ApplyOutputPolicy(mode="in_place" if saving.in_place else "default")
    raise ValueError("Resolved apply config does not define an output route.")


def resolve_apply_save_input(
    cfg: ApplyDefaults,
    *,
    output_policy: ApplyOutputPolicy,
) -> bool:
    """Interpret the resolved input-saving policy for the chosen output route."""
    save_input_mode = cfg.saving.save_input_mode
    if save_input_mode == "always":
        return True
    if save_input_mode == "never":
        return False
    if save_input_mode == "if_not_in_place":
        return output_policy.mode != "in_place"
    raise ValueError(f"Resolved apply config has no valid save_input_mode: {save_input_mode!r}")


def resolve_evaluate_config(
    *,
    config: str | Path | None = None,
    overrides: EvaluateOverrides | None = None,
) -> EvaluateDefaults:
    """Resolve one evaluate invocation to the canonical typed nested config."""
    resolved = _resolve_section_nested("evaluate", config=config)
    if not isinstance(resolved, EvaluateDefaults):
        raise TypeError("Internal error: evaluate resolution did not produce EvaluateDefaults.")

    if overrides is None:
        return resolved

    override_values = overrides.model_dump(exclude_unset=True)
    if not override_values:
        return resolved

    merged = _merge_section_config("evaluate", resolved.model_dump(), override_values)
    validated = _validate_resolved_section("evaluate", merged)
    if not isinstance(validated, EvaluateDefaults):
        raise TypeError("Internal error: evaluate overrides did not produce EvaluateDefaults.")
    return validated


def load_inference_defaults(path: str | Path | None = None) -> ResolvedInferenceConfig:
    resolved = ResolvedInferenceConfig().model_dump()
    if path is None:
        cfg, cfg_path = load_inference_config(None)
    else:
        cfg_path = Path(path)
        cfg = InferenceOverrides.model_validate(load_yaml(cfg_path))

    if cfg_path is not None:
        for section in ("apply", "evaluate"):
            overrides = _section_overrides(cfg, section)
            if overrides:
                resolved[section] = deep_merge(resolved[section], overrides)
    return ResolvedInferenceConfig.model_validate(resolved)


__all__ = [
    "ApplyOutputPolicy",
    "InferenceConfig",
    "InferenceDefaults",
    "load_inference_config",
    "load_inference_defaults",
    "resolve_apply_config",
    "resolve_apply_output_policy",
    "resolve_apply_save_input",
    "resolve_evaluate_config",
    "resolve_inference_config_path",
]
