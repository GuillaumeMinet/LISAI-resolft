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
    EvaluateDefaults,
    InferenceOverrides,
    ResolvedInferenceConfig,
    SaveInputMode,
)
from lisai.config.models.inference.presets import POST_TRAINING_OVERRIDES

inference_config_paths = ConfigPathResolver("inference")
POST_TRAINING_CONFIG_NAME = "post_training"

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
) -> dict[str, Any]:
    if section == "apply":
        return ApplyDefaults.model_validate(value).model_dump()
    return EvaluateDefaults.model_validate(value).model_dump()


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
) -> dict[str, Any]:
    # Precedence is intentionally one-way:
    # canonical typed defaults < local/defaults.yml < built-in preset (if any)
    # < selected config < CLI.
    # A selected config may therefore stay sparse; local defaults only fill
    # values that it does not explicitly provide.
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


def _flatten_apply_section(section: dict[str, Any]) -> dict[str, Any]:
    checkpoint = section["checkpoint"]
    input_cfg = section["input"]
    inference = section["inference"]
    postprocess = section["postprocess"]
    color_code = dict(postprocess["color_code"])
    saving = section["saving"]

    enabled = color_code.pop("enabled")
    return {
        "epoch_number": checkpoint["epoch_number"],
        "best_or_last": checkpoint["best_or_last"],
        "filters": input_cfg["filters"],
        "skip_if_contain": input_cfg["skip_if_contain"],
        "crop_size": inference["crop_size"],
        "keep_original_shape": inference["keep_original_shape"],
        "tiling_size": inference["tiling_size"],
        "stack_selection_idx": input_cfg["stack_selection_idx"],
        "limit_n_imgs": input_cfg["limit_n_imgs"],
        "timelapse_max": input_cfg["timelapse_max"],
        "lvae_num_samples": inference["lvae_num_samples"],
        "lvae_save_samples": saving["lvae_save_samples"],
        "denormalize_output": postprocess["denormalize"],
        "downsamp": inference["downsamp"],
        "fill_factor": inference["fill_factor"],
        "apply_color_code": enabled,
        "color_code_prm": color_code,
        "dark_frame_context_length": inference["dark_frame_context_length"],
    }


def _flatten_evaluate_section(section: dict[str, Any]) -> dict[str, Any]:
    checkpoint = section["checkpoint"]
    data = section["data"]
    inference = section["inference"]
    saving = section["saving"]
    return {
        "best_or_last": checkpoint["best_or_last"],
        "epoch_number": checkpoint["epoch_number"],
        "tiling_size": inference["tiling_size"],
        "crop_size": inference["crop_size"],
        "metrics_list": section["metrics"],
        "lvae_num_samples": inference["lvae_num_samples"],
        "results": None,
        "save_folder": saving["save_folder"],
        "overwrite": saving["overwrite"],
        "eval_gt": data["eval_gt"],
        "data_prm_update": data["overrides"],
        "ch_out": inference["ch_out"],
        "split": data["split"],
        "limit_n_imgs": data["limit_n_imgs"],
        "timelapse_max": data["timelapse_max"],
    }


def resolve_apply_options(
    *,
    defaults: ResolvedInferenceConfig | None = None,
    defaults_path: str | Path | None = None,
    config: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if defaults is not None or defaults_path is not None:
        loaded_defaults = load_inference_defaults(defaults_path) if defaults is None else defaults
        section_defaults = _flatten_apply_section(loaded_defaults.apply.model_dump())
    else:
        section_defaults = _flatten_apply_section(_resolve_section_nested("apply", config=config))
    return _resolve_task_options(section_defaults, overrides)


def _resolved_apply_saving(config: str | Path | None) -> dict[str, Any]:
    return _resolve_section_nested("apply", config=config)["saving"]


def resolve_apply_output_policy(
    *,
    config: str | Path | None = None,
    save_folder: str | Path | None | UnsetType = UNSET,
    output_mode: ApplyOutputMode | UnsetType = UNSET,
    in_place: bool | UnsetType = UNSET,
    stg=None,
) -> ApplyOutputPolicy:
    """Resolve apply output routing as CLI > inference saving config > local_config."""
    cli_choices = [
        save_folder is not UNSET,
        output_mode is not UNSET,
        in_place is not UNSET,
    ]
    if sum(cli_choices) > 1:
        raise ValueError(
            "save_folder, output_mode, and in_place are mutually exclusive output overrides."
        )

    if save_folder is not UNSET:
        if save_folder is None or not str(save_folder).strip():
            raise ValueError("save_folder must not be empty when explicitly provided.")
        return ApplyOutputPolicy(mode="folder", save_folder=Path(save_folder))
    if output_mode is not UNSET:
        return ApplyOutputPolicy(mode=output_mode)
    if in_place is not UNSET:
        return ApplyOutputPolicy(mode="in_place" if in_place else "default")

    saving = _resolved_apply_saving(config)
    if saving["save_folder"] is not None:
        return ApplyOutputPolicy(mode="folder", save_folder=Path(saving["save_folder"]))
    if saving["mode"] is not None:
        return ApplyOutputPolicy(mode=saving["mode"])
    if saving["in_place"] is not None:
        return ApplyOutputPolicy(mode="in_place" if saving["in_place"] else "default")

    if stg is None:
        from lisai.config.settings import settings as stg

    return ApplyOutputPolicy(mode=stg.INFERENCE_OUTPUT_MODE)


def resolve_apply_save_input(
    *,
    output_policy: ApplyOutputPolicy,
    config: str | Path | None = None,
    save_input: bool | UnsetType = UNSET,
    stg=None,
) -> bool:
    """Resolve whether apply saves its input as CLI > inference saving config > local_config."""
    if save_input is not UNSET:
        return bool(save_input)

    save_input_mode: SaveInputMode | None = _resolved_apply_saving(config)["save_input_mode"]

    if stg is None:
        from lisai.config.settings import settings as stg

    if save_input_mode is None:
        save_input_mode = stg.INFERENCE_SAVE_INPUT_MODE

    if save_input_mode == "always":
        return True
    if save_input_mode == "never":
        return False
    if save_input_mode == "if_not_in_place":
        return output_policy.mode != "in_place"
    raise ValueError(f"Unknown save_input_mode: {save_input_mode!r}")


def resolve_evaluate_options(
    *,
    defaults: ResolvedInferenceConfig | None = None,
    defaults_path: str | Path | None = None,
    config: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if defaults is not None or defaults_path is not None:
        loaded_defaults = load_inference_defaults(defaults_path) if defaults is None else defaults
        section_defaults = _flatten_evaluate_section(loaded_defaults.evaluate.model_dump())
    else:
        section_defaults = _flatten_evaluate_section(_resolve_section_nested("evaluate", config=config))
    return _resolve_task_options(section_defaults, overrides)


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
    "UNSET",
    "UnsetType",
    "ApplyOutputPolicy",
    "InferenceConfig",
    "InferenceDefaults",
    "load_inference_config",
    "load_inference_defaults",
    "resolve_apply_options",
    "resolve_apply_output_policy",
    "resolve_apply_save_input",
    "resolve_evaluate_options",
    "resolve_inference_config_path",
]
