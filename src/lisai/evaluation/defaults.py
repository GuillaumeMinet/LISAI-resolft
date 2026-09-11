from __future__ import annotations

from copy import deepcopy
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


def _resolve_section_defaults(
    section: Literal["apply", "evaluate"],
    *,
    config: str | Path | None = None,
) -> dict[str, Any]:
    resolved = ResolvedInferenceConfig().model_dump()
    defaults_cfg, _ = load_inference_config(None)
    resolved = deep_merge(resolved, defaults_cfg.model_dump(exclude_unset=True))

    if config is None:
        return dict(resolved[section])

    named_cfg, cfg_path = load_inference_config(config)
    named_cfg_dict = named_cfg.model_dump(exclude_unset=True)
    if section not in named_cfg_dict:
        raise ValueError(
            f"Inference config '{cfg_path}' does not define a '{section}' section."
        )
    resolved = deep_merge(resolved, named_cfg_dict)
    return dict(resolved[section])


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
        cfg = load_yaml(cfg_path)
        raw = InferenceOverrides.model_validate(cfg).model_dump(exclude_unset=True)
        resolved = deep_merge(resolved, raw)
    return ResolvedInferenceConfig.model_validate(resolved)


__all__ = [
    "UNSET",
    "UnsetType",
    "InferenceConfig",
    "InferenceDefaults",
    "load_inference_config",
    "load_inference_defaults",
    "resolve_apply_options",
    "resolve_evaluate_options",
    "resolve_inference_config_path",
]
