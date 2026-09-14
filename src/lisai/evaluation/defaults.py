from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from lisai.config import load_yaml
from lisai.config.io import deep_merge
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.models.inference import (
    InferenceOverrides,
    ResolvedInferenceConfig,
)

inference_config_paths = ConfigPathResolver("inference")

# Backward-compatible aliases kept while the clearer inference model names settle in.
InferenceConfig = InferenceOverrides
InferenceDefaults = ResolvedInferenceConfig


class UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()


@dataclass(frozen=True)
class ApplyOutputPolicy:
    """Resolved destination policy for one `apply` invocation."""

    mode: Literal["default", "in_place", "folder"]
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


def _merge_value(default: Any, override: Any) -> Any:
    if override is UNSET:
        return deepcopy(default)
    if isinstance(default, Mapping) and isinstance(override, Mapping):
        return deep_merge(dict(default), dict(override))
    return override


def _resolve_task_options(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise KeyError(f"Unknown inference default override(s): {', '.join(unknown)}")
    return {key: _merge_value(default, overrides.get(key, UNSET)) for key, default in defaults.items()}


def _processing_section_overrides(
    cfg: InferenceOverrides,
    section: Literal["apply", "evaluate"],
) -> dict[str, Any] | None:
    raw = cfg.model_dump(exclude_unset=True)
    section_raw = raw.get(section)
    if section_raw is None:
        return None
    section_raw = dict(section_raw)
    if section == "apply":
        # Output routing has its own precedence chain and must never be deep-merged
        # with inference-processing defaults.
        section_raw.pop("output", None)
    return section_raw


def _resolve_section_defaults(
    section: Literal["apply", "evaluate"],
    *,
    config: str | Path | None = None,
) -> dict[str, Any]:
    resolved_section = dict(ResolvedInferenceConfig().model_dump()[section])

    defaults_cfg, _ = load_inference_config(None)
    defaults_overrides = _processing_section_overrides(defaults_cfg, section)
    if defaults_overrides:
        resolved_section = deep_merge(resolved_section, defaults_overrides)

    if config is None:
        return resolved_section

    named_cfg, cfg_path = load_inference_config(config)
    named_overrides = _processing_section_overrides(named_cfg, section)
    if named_overrides is None:
        raise ValueError(
            f"Inference config '{cfg_path}' does not define a '{section}' section."
        )
    return deep_merge(resolved_section, named_overrides)


def resolve_apply_options(
    *,
    defaults: ResolvedInferenceConfig | None = None,
    defaults_path: str | Path | None = None,
    config: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if defaults is not None or defaults_path is not None:
        loaded_defaults = load_inference_defaults(defaults_path) if defaults is None else defaults
        section_defaults = loaded_defaults.apply.model_dump()
    else:
        section_defaults = _resolve_section_defaults("apply", config=config)
    return _resolve_task_options(section_defaults, overrides)


def resolve_apply_output_policy(
    *,
    config: str | Path | None = None,
    save_folder: str | Path | None | UnsetType = UNSET,
    in_place: bool | UnsetType = UNSET,
    stg=None,
) -> ApplyOutputPolicy:
    """Resolve apply output routing as CLI > named config > local config > project default."""
    if save_folder is not UNSET and in_place is not UNSET:
        raise ValueError("save_folder and in_place are mutually exclusive output overrides.")

    # Explicit CLI output choice is authoritative. False means "force default routing".
    if save_folder is not UNSET:
        if save_folder is None or not str(save_folder).strip():
            raise ValueError("save_folder must not be empty when explicitly provided.")
        return ApplyOutputPolicy(mode="folder", save_folder=Path(save_folder))
    if in_place is not UNSET:
        return ApplyOutputPolicy(mode="in_place" if in_place else "default")

    # Only an explicitly selected named config participates in output routing.
    # defaults.yml provides processing defaults, while local_config owns normal
    # output behavior.
    if config is not None:
        named_cfg, _ = load_inference_config(config)
        if named_cfg.apply is not None and named_cfg.apply.output is not None:
            output = named_cfg.apply.output
            if output.save_folder is not None:
                return ApplyOutputPolicy(mode="folder", save_folder=Path(output.save_folder))
            if output.in_place is not None:
                return ApplyOutputPolicy(mode="in_place" if output.in_place else "default")

    if stg is None:
        from lisai.config.settings import settings as stg

    if stg.INFERENCE_OUTPUT_MODE == "in_place":
        return ApplyOutputPolicy(mode="in_place")
    return ApplyOutputPolicy(mode="default")


def resolve_evaluate_options(
    *,
    defaults: ResolvedInferenceConfig | None = None,
    defaults_path: str | Path | None = None,
    config: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if defaults is not None or defaults_path is not None:
        loaded_defaults = load_inference_defaults(defaults_path) if defaults is None else defaults
        section_defaults = loaded_defaults.evaluate.model_dump()
    else:
        section_defaults = _resolve_section_defaults("evaluate", config=config)
    return _resolve_task_options(section_defaults, overrides)


def load_inference_defaults(path: str | Path | None = None) -> ResolvedInferenceConfig:
    resolved = ResolvedInferenceConfig().model_dump()
    cfg_path = Path(path) if path is not None else resolve_inference_config_path(None)
    if cfg_path is not None:
        cfg = InferenceOverrides.model_validate(load_yaml(cfg_path))
        for section in ("apply", "evaluate"):
            overrides = _processing_section_overrides(cfg, section)
            if overrides:
                resolved[section] = deep_merge(resolved[section], overrides)
    return ResolvedInferenceConfig.model_validate(resolved)


__all__ = [
    "UNSET",
    "UnsetType",
    "ApplyOutputPolicy",
    "InferenceConfig",
    "InferenceDefaults",
    "load_inference_config",
    "load_inference_defaults",
    "resolve_apply_options",
    "resolve_apply_output_policy",
    "resolve_evaluate_options",
    "resolve_inference_config_path",
]
