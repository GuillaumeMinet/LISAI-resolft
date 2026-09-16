"""Compatibility-only migration for legacy flat inference YAML layouts.

New inference code must use the nested apply/evaluate schema directly.  The
helpers in this module exist only so inference configs written before the
nested schema was introduced continue to load.  They can be removed together
with flat-config compatibility in a future cleanup.
"""

from __future__ import annotations

from typing import Any


_APPLY_LEGACY_GROUPS = {
    "checkpoint": {"epoch_number", "best_or_last"},
    "input": {"filters", "skip_if_contain", "stack_selection_idx", "limit_n_imgs", "timelapse_max"},
    "inference": {
        "crop_size",
        "keep_original_shape",
        "tiling_size",
        "lvae_num_samples",
        "downsamp",
        "fill_factor",
        "dark_frame_context_length",
    },
}

_EVALUATE_LEGACY_GROUPS = {
    "checkpoint": {"best_or_last", "epoch_number"},
    "data": {"split", "eval_gt", "limit_n_imgs", "timelapse_max"},
    "inference": {"tiling_size", "crop_size", "lvae_num_samples", "ch_out"},
    "saving": {"save_folder", "overwrite"},
}

_APPLY_LEGACY_KEYS = (
    set().union(*_APPLY_LEGACY_GROUPS.values())
    | {
        "denormalize_output",
        "color_code_prm",
        "apply_color_code",
        "output",
        "lvae_save_samples",
    }
)
_EVALUATE_LEGACY_KEYS = (
    set().union(*_EVALUATE_LEGACY_GROUPS.values())
    | {"metrics_list", "data_prm_update"}
)


def _sparse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(exclude_unset=True))
    return dict(value)


def migrate_legacy_flat_apply_config(value: Any) -> Any:
    """Translate the former flat ``apply`` layout into the nested schema."""
    if not isinstance(value, dict):
        return value

    raw = dict(value)
    if not (_APPLY_LEGACY_KEYS & raw.keys()):
        return value

    migrated: dict[str, Any] = {}

    for section, keys in _APPLY_LEGACY_GROUPS.items():
        nested = raw.pop(section, None)
        if nested is not None:
            migrated[section] = _sparse_mapping(nested)
        for key in keys:
            if key in raw:
                if section in migrated and key in migrated[section]:
                    raise ValueError(f"apply config defines both '{key}' and '{section}.{key}'.")
                migrated.setdefault(section, {})[key] = raw.pop(key)

    postprocess = raw.pop("postprocess", None)
    if postprocess is not None:
        migrated["postprocess"] = _sparse_mapping(postprocess)
    if "denormalize_output" in raw:
        migrated.setdefault("postprocess", {})["denormalize"] = raw.pop("denormalize_output")
    legacy_color = raw.pop("color_code_prm", None)
    if legacy_color is not None:
        migrated.setdefault("postprocess", {}).setdefault("color_code", {}).update(
            dict(legacy_color)
        )
    if "apply_color_code" in raw:
        migrated.setdefault("postprocess", {}).setdefault("color_code", {})["enabled"] = raw.pop(
            "apply_color_code"
        )

    saving = raw.pop("saving", None)
    legacy_output = raw.pop("output", None)
    if saving is not None and legacy_output is not None:
        raise ValueError("apply config cannot define both 'saving' and legacy 'output'.")
    if saving is not None:
        migrated["saving"] = _sparse_mapping(saving)
    elif legacy_output is not None:
        migrated["saving"] = _sparse_mapping(legacy_output)
    if "lvae_save_samples" in raw:
        migrated.setdefault("saving", {})["lvae_save_samples"] = raw.pop("lvae_save_samples")

    migrated.update(raw)
    return migrated


def migrate_legacy_flat_evaluate_config(value: Any) -> Any:
    """Translate the former flat ``evaluate`` layout into the nested schema."""
    if not isinstance(value, dict):
        return value

    raw = dict(value)
    if not (_EVALUATE_LEGACY_KEYS & raw.keys()):
        return value

    migrated: dict[str, Any] = {}

    for section, keys in _EVALUATE_LEGACY_GROUPS.items():
        nested = raw.pop(section, None)
        if nested is not None:
            migrated[section] = _sparse_mapping(nested)
        for key in keys:
            if key in raw:
                if section in migrated and key in migrated[section]:
                    raise ValueError(f"evaluate config defines both '{key}' and '{section}.{key}'.")
                migrated.setdefault(section, {})[key] = raw.pop(key)

    if "metrics_list" in raw:
        if "metrics" in raw:
            raise ValueError(
                "evaluate config cannot define both 'metrics' and legacy 'metrics_list'."
            )
        migrated["metrics"] = raw.pop("metrics_list")
    elif "metrics" in raw:
        migrated["metrics"] = raw.pop("metrics")

    if "data_prm_update" in raw:
        migrated.setdefault("data", {})["overrides"] = raw.pop("data_prm_update")

    migrated.update(raw)
    return migrated


__all__ = [
    "migrate_legacy_flat_apply_config",
    "migrate_legacy_flat_evaluate_config",
]
