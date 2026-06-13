from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from lisai.data.dataset_registry import (
    registry_data_format_for_output,
    registry_data_types,
    registry_mapping_for_data_type,
    registry_paths_for_data_type,
)

from .tasks import (
    DenoisingCARETaskSection,
    DenoisingHDNTaskSection,
    DenoisingUNetRCANTaskSection,
)


def _timelapse_context_length(data: Any) -> int | None:
    timelapse_prm = getattr(data, "timelapse_prm", None)
    if timelapse_prm is None:
        return None
    context_length = getattr(timelapse_prm, "context_length", None)
    return int(context_length) if context_length is not None else None


def _resolved_data_format(data: Any) -> str | None:
    value = getattr(data, "data_format", None)
    if value is not None:
        return str(value)
    dataset_info = getattr(data, "dataset_info", None)
    if isinstance(dataset_info, dict) and dataset_info.get("data_format") is not None:
        return str(dataset_info["data_format"])
    return None


def _data_value(data: Any, *names: str) -> Any:
    for name in names:
        value = getattr(data, name, None)
        if value is not None:
            return value
    return None


def _registry_default_value(dataset_info: Mapping[str, Any], data_type: str | None, name: str) -> str | None:
    defaults = registry_mapping_for_data_type(dataset_info, "defaults", data_type)
    if not isinstance(defaults, Mapping):
        return None
    value = defaults.get(name)
    return str(value) if value is not None else None


def _is_evaluation_only_dataset(dataset_info: Mapping[str, Any]) -> bool:
    if dataset_info.get("for_training") is False:
        return True
    usage = dataset_info.get("usage")
    return isinstance(usage, str) and usage.lower() in {"eval", "evaluation", "test"}


def _validate_registry_member(
    *,
    dataset_info: Mapping[str, Any],
    data_type: str | None,
    field_name: str,
    value: Any,
) -> None:
    if value is None:
        return

    paths = registry_paths_for_data_type(dataset_info, data_type)
    if not paths:
        return

    text = str(value)
    if text not in paths:
        dataset_name = dataset_info.get("name")
        suffix = f" for dataset {dataset_name!r}" if dataset_name else ""
        allowed = ", ".join(repr(path) for path in sorted(paths))
        raise ValueError(
            f"`{field_name}`={text!r} is not registered{suffix}; expected one of: {allowed}."
        )


def _validate_registry_consistency(data: Any) -> None:
    if not bool(getattr(data, "registry_checked", False)):
        return

    dataset_name = getattr(data, "dataset_name", None)
    dataset_info = getattr(data, "dataset_info", None)
    prep_before = bool(getattr(data, "prep_before", True))

    if not isinstance(dataset_info, Mapping):
        if prep_before:
            raise ValueError(
                f"Dataset {dataset_name!r} is not registered; set `data.prep_before=false` "
                "for unprepared training or preprocess/register the dataset first."
            )
        return

    if _is_evaluation_only_dataset(dataset_info):
        raise ValueError(f"Dataset {dataset_name!r} is marked as evaluation-only and cannot be used for training.")

    data_type = getattr(data, "registry_data_type", None)
    known_data_types = registry_data_types(dataset_info)
    if data_type is not None and known_data_types and str(data_type) not in known_data_types:
        allowed = ", ".join(repr(value) for value in sorted(known_data_types))
        raise ValueError(
            f"Registry data type {data_type!r} is not available for dataset {dataset_name!r}; "
            f"expected one of: {allowed}."
        )

    if not prep_before:
        return

    input_value = _data_value(data, "input", "inp")
    if input_value is None:
        default_input = _registry_default_value(dataset_info, data_type, "input")
        hint = (
            f"registry default is {default_input!r}; add it explicitly to the config."
            if default_input is not None
            else "set `data.input` explicitly in the config."
        )
        raise ValueError(
            f"`data.input` is required for prepared training on dataset {dataset_name!r}; "
            f"{hint}"
        )

    registry_format = dataset_info.get("data_format")
    explicit_format = getattr(data, "data_format", None)
    if explicit_format is not None:
        allowed_formats = {
            str(value)
            for value in (
                registry_format,
                registry_data_format_for_output(dataset_info, data_type, input_value),
            )
            if value is not None
        }
        if allowed_formats and str(explicit_format) not in allowed_formats:
            allowed = ", ".join(repr(value) for value in sorted(allowed_formats))
            raise ValueError(
                f"`data.data_format`={explicit_format!r} is not compatible with registry metadata "
                f"for dataset {dataset_name!r}; expected one of: {allowed}."
            )

    target_value = _data_value(data, "target", "gt")
    if bool(getattr(data, "paired", False)) and target_value is None:
        default_target = _registry_default_value(dataset_info, data_type, "target")
        hint = (
            f"registry default is {default_target!r}; add it explicitly to the config."
            if default_target is not None
            else "set `data.target` explicitly in the config."
        )
        raise ValueError(
            f"`data.target` is required for paired training on dataset {dataset_name!r}; "
            f"{hint}"
        )

    _validate_registry_member(
        dataset_info=dataset_info,
        data_type=data_type,
        field_name="data.input",
        value=input_value,
    )
    if target_value is not None:
        _validate_registry_member(
            dataset_info=dataset_info,
            data_type=data_type,
            field_name="data.target",
            value=target_value,
        )


def _expected_input_channels(data: Any) -> tuple[int | None, str]:
    downsampling = getattr(data, "downsampling", None)
    context_length = _timelapse_context_length(data)

    if downsampling is not None and downsampling.downsamp_method == "multiple":
        if context_length is not None and context_length > 1:
            raise ValueError(
                "`data.downsampling.multiple_prm` is incompatible with `data.timelapse_prm.context_length > 1`."
            )
        n_ch = downsampling.multiple_input_channels()
        if n_ch < 1:
            raise ValueError(
                "`data.downsampling.multiple_prm.fill_factor` yields zero input channels with the current `downsamp_factor`."
            )
        return n_ch, f"`int(downsamp_factor**2 * fill_factor)` = {n_ch}"

    if context_length is not None:
        return context_length, f"`data.timelapse_prm.context_length` ({context_length})"

    if getattr(data, "timelapse_prm", None) is not None:
        return 1, "`data.timelapse_prm.context_length=null` (single-frame timelapse)"

    if _resolved_data_format(data) == "timelapse":
        return None, "timelapse input without explicit `context_length`"

    return 1, "default single-frame input"


def _experiment_task(cfg: Any) -> Any:
    experiment = getattr(cfg, "experiment", None)
    return getattr(experiment, "task", None)


def _require_no_context_window(data: Any, *, task_name: str) -> None:
    context_length = _timelapse_context_length(data)
    if context_length is not None:
        raise ValueError(
            f"`experiment.task.name='{task_name}'` does not support `data.timelapse_prm.context_length`; "
            "remove it or set it to null."
        )


def _effective_upsampling_factor(params: Any) -> int | None:
    factor = getattr(params, "effective_upsampling_factor", None)
    if callable(factor):
        return int(factor())
    return None


def _validate_denoising_task_consistency(
    cfg: Any,
    *,
    architecture: str,
    params: Any,
    data: Any,
) -> None:
    task = _experiment_task(cfg)

    if isinstance(task, DenoisingHDNTaskSection):
        if architecture != "lvae":
            raise ValueError("`experiment.task.name='denoising_hdn'` requires `model.architecture='lvae'`.")
        if bool(data.paired) != bool(task.supervised):
            raise ValueError(
                "`data.paired` must match `experiment.task.supervised` for `denoising_hdn`."
            )
        return

    if isinstance(task, DenoisingCARETaskSection):
        _validate_supervised_denoising_task(
            task_name="denoising_care",
            expected_architecture="unet",
            architecture=architecture,
            params=params,
            data=data,
        )
        return

    if isinstance(task, DenoisingUNetRCANTaskSection):
        _validate_supervised_denoising_task(
            task_name="denoising_unetrcan",
            expected_architecture="unet_rcan",
            architecture=architecture,
            params=params,
            data=data,
        )


def _validate_supervised_denoising_task(
    *,
    task_name: str,
    expected_architecture: str,
    architecture: str,
    params: Any,
    data: Any,
) -> None:
    if architecture != expected_architecture:
        raise ValueError(
            f"`experiment.task.name='{task_name}'` requires `model.architecture='{expected_architecture}'`."
        )
    if not bool(data.paired):
        raise ValueError(f"`experiment.task.name='{task_name}'` requires `data.paired=true`.")
    _require_no_context_window(data, task_name=task_name)
    if getattr(data, "downsampling", None) is not None:
        raise ValueError(f"`experiment.task.name='{task_name}'` does not support `data.downsampling`.")

    upsampling_factor = _effective_upsampling_factor(params)
    if upsampling_factor not in {None, 1}:
        raise ValueError(
            f"`experiment.task.name='{task_name}'` requires model effective upsampling factor 1, "
            f"got {upsampling_factor}."
        )


def validate_cross_section_consistency(cfg: Any, *, emit_warnings: bool) -> Any:
    data = getattr(cfg, "data", None)
    if data is not None:
        _validate_registry_consistency(data)

    model = getattr(cfg, "model", None)
    if model is None:
        return cfg

    architecture = model.architecture
    params = model.parameters
    context_length = _timelapse_context_length(data)

    _validate_denoising_task_consistency(
        cfg,
        architecture=architecture,
        params=params,
        data=data,
    )

    if architecture == "lvae" and context_length not in {None, 1}:
        raise ValueError(
            "`model.architecture='lvae'` does not support timelapse context windows; `data.timelapse_prm.context_length` must be null or 1."
        )

    expected_n_ch, source = _expected_input_channels(data)

    if architecture == "lvae" and expected_n_ch not in {None, 1}:
        raise ValueError(
            f"`model.architecture='lvae'` does not support multi-channel generated inputs; got {source}."
        )

    if architecture in {"unet", "unet3d", "rcan"}:
        if expected_n_ch is not None and params.in_channels != expected_n_ch:
            raise ValueError(
                f"`model.parameters.in_channels` must match {source}; expected {expected_n_ch}, got {params.in_channels}."
            )
        if params.out_channels != 1:
            raise ValueError(
                f"`model.parameters.out_channels` must be 1 for architecture '{architecture}', got {params.out_channels}."
            )

    elif architecture == "unet_rcan":
        if expected_n_ch is not None:
            if params.UNet_prm.in_channels != expected_n_ch:
                raise ValueError(
                    f"`model.parameters.UNet_prm.in_channels` must match {source}; expected {expected_n_ch}, got {params.UNet_prm.in_channels}."
                )
            if params.UNet_prm.out_channels != expected_n_ch:
                raise ValueError(
                    f"`model.parameters.UNet_prm.out_channels` must match {source}; expected {expected_n_ch}, got {params.UNet_prm.out_channels}."
                )
        if params.RCAN_prm.out_channels != 1:
            raise ValueError(
                f"`model.parameters.RCAN_prm.out_channels` must be 1 for architecture 'unet_rcan', got {params.RCAN_prm.out_channels}."
            )

    if data.downsampling is not None:
        data_factor = int(data.downsampling.downsamp_factor)
        model_factor = int(params.effective_upsampling_factor())
        if model_factor != data_factor:
            if not bool(data.paired):
                raise ValueError(
                    f"`data.downsampling.downsamp_factor` ({data_factor}) must match the model effective upsampling factor ({model_factor}) when `data.paired=false`."
                )
            if emit_warnings:
                warnings.warn(
                    f"`data.downsampling.downsamp_factor` ({data_factor}) does not match the model effective upsampling factor ({model_factor}).",
                    stacklevel=3,
                )

    return cfg
