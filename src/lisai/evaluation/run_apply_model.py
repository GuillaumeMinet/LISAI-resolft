"""High-level entrypoint for applying a saved model to arbitrary input files.

This module orchestrates the evaluation pipeline for prediction-only usage:
resolve the saved run, initialize the inference runtime, prepare input files,
run stack inference, and save outputs.
"""

import warnings
from pathlib import Path
from typing import Union

import numpy as np
from tifffile import imread

from lisai.config.progress import resolve_progress_bar
from lisai.config.models.inference import ApplyOutputMode
from lisai.data.utils import center_pad, crop_center
from lisai.evaluation.defaults import (
    UNSET,
    UnsetType,
    resolve_apply_options,
    resolve_apply_output_policy,
    resolve_apply_save_input,
)
from lisai.evaluation.inference.normalization import denormalize_pred, normalize_inp
from lisai.evaluation.inference.progress import InferenceProgress
from lisai.evaluation.inference.shape import inverse_make_4d, make_4d
from lisai.evaluation.inference.stack import predict_4d_stack
from lisai.evaluation.io import create_save_folder, resolve_prediction_inputs, save_outputs
from lisai.evaluation.runtime import TilingSizePolicy, initialize_runtime
from lisai.evaluation.saved_run import load_saved_run, resolve_run_dir
from lisai.evaluation.visualization.z_projection import (
    add_colorbar,
    create_color_coded_image,
    enhance_contrast,
)
from lisai.infra.paths import Paths
from lisai.lib.upsamp.inp_generators import (
    _deterministic_mltpl_sampling,
    generate_downsamp_inp,
)


def _ensure_shape(img: np.ndarray, downsamp_factor: int) -> np.ndarray:
    """Pad spatial dimensions so they are divisible by `downsamp_factor`.

    Padding is applied only on the bottom/right borders to preserve the
    top-left sampling grid used by deterministic multiple downsampling.
    """
    pad_h = (-img.shape[-2]) % downsamp_factor
    pad_w = (-img.shape[-1]) % downsamp_factor
    if pad_h == 0 and pad_w == 0:
        return img

    pad_width = [(0, 0)] * img.ndim
    pad_width[-2] = (0, pad_h)
    pad_width[-1] = (0, pad_w)
    return np.pad(img, pad_width=tuple(pad_width), mode="constant", constant_values=0)


def _resolve_fill_factor_for_multiple_apply_downsampling(
    *,
    downsamp: int,
    fill_factor: float
) -> float:

    resolved_fill_factor = float(fill_factor)
    if resolved_fill_factor <= 0 or resolved_fill_factor > 1:
        raise ValueError(
            f"`apply.fill_factor` must be in the interval (0, 1], got {fill_factor!r}."
        )

    n_ch = int(downsamp**2 * resolved_fill_factor)
    if n_ch < 1:
        raise ValueError(
            "Computed zero channels for deterministic `multiple` apply downsampling. "
            "Increase `apply.fill_factor` or `apply.downsamp`."
        )

    supported = _deterministic_mltpl_sampling.get(downsamp, {})
    if n_ch not in supported:
        raise ValueError(
            "Deterministic `multiple` downsampling is not implemented for "
            f"`downsamp_factor={downsamp}` and `n_ch={n_ch}`. ")
    return resolved_fill_factor


def _format_tiling_size_for_display(requested: TilingSizePolicy, effective: int | None) -> str:
    if effective is None:
        return "off"
    if requested is None or requested == "auto":
        return f"{effective} (auto)"
    return str(effective)


def _source_name(data_path: Path) -> str:
    """Return a compact source identifier suitable for output-folder naming."""
    data_path = Path(data_path)
    if data_path.is_dir():
        return data_path.name or "input"

    parts = [data_path.parent.name, data_path.stem]
    source_name = "_".join(part for part in parts if part)
    return source_name or "input"


def _prediction_folder_name(
    *,
    source_name: str | None,
    model_subfolder: str,
    model_name: str,
) -> str:
    parts = ["Predict"]
    if source_name:
        parts.append(source_name)
    parts.extend([model_subfolder, model_name])
    return "_".join(parts)


def run_apply_model(model_dataset: str,
                model_subfolder: str,
                model_name: str,
                data_path: Path,
                save_folder: str | Path | None | UnsetType = UNSET,
                output_mode: ApplyOutputMode | UnsetType = UNSET,
                in_place: bool | UnsetType = UNSET,
                epoch_number: int | None | UnsetType = UNSET,
                best_or_last: str | UnsetType = UNSET,
                filters: list[str] | str | UnsetType = UNSET,
                skip_if_contain: list[str] | None | UnsetType = UNSET,
                crop_size: Union[int, tuple[int, int], None, UnsetType] = UNSET,
                keep_original_shape: bool | UnsetType = UNSET,
                tiling_size: TilingSizePolicy | UnsetType = UNSET,
                stack_selection_idx: int | None | UnsetType = UNSET,
                limit_n_imgs: int | None | UnsetType = UNSET,
                timelapse_max: int | None | UnsetType = UNSET,
                lvae_num_samples: int | None | UnsetType = UNSET,
                lvae_save_samples: bool | UnsetType = UNSET,
                denormalize_output: bool | UnsetType = UNSET,
                save_input: bool | UnsetType = UNSET,
                downsamp: int | None | UnsetType = UNSET,
                fill_factor: float | None | UnsetType = UNSET,
                apply_color_code: bool | UnsetType = UNSET,
                color_code_prm: dict | None | UnsetType = UNSET,
                dark_frame_context_length: bool | UnsetType = UNSET,
                config: str | Path | None = None,
                promoted_model_name: str | None = None,
                progress_bar: bool | None = None):
    """Apply a saved model checkpoint to one file or a directory of files.

    Omitted processing options are resolved from inference defaults or the named
    inference config. Output placement and input-saving policies additionally use
    local inference settings.
    """
    options = resolve_apply_options(
        config=config,
        epoch_number=epoch_number,
        best_or_last=best_or_last,
        filters=filters,
        skip_if_contain=skip_if_contain,
        crop_size=crop_size,
        keep_original_shape=keep_original_shape,
        tiling_size=tiling_size,
        stack_selection_idx=stack_selection_idx,
        limit_n_imgs=limit_n_imgs,
        timelapse_max=timelapse_max,
        lvae_num_samples=lvae_num_samples,
        lvae_save_samples=lvae_save_samples,
        denormalize_output=denormalize_output,
        downsamp=downsamp,
        fill_factor=fill_factor,
        apply_color_code=apply_color_code,
        color_code_prm=color_code_prm,
        dark_frame_context_length=dark_frame_context_length,
    )
    color_code_prm = options["color_code_prm"] or {}
    output_policy = resolve_apply_output_policy(
        config=config,
        save_folder=save_folder,
        output_mode=output_mode,
        in_place=in_place,
    )
    save_input = resolve_apply_save_input(
        config=config,
        output_policy=output_policy,
        save_input=save_input,
    )
    progress = InferenceProgress(
        enabled=resolve_progress_bar(True, progress_bar)
    )

    data_path = Path(data_path)
    if promoted_model_name is not None:
        from lisai.promoted_models import load_promoted_model

        promoted = load_promoted_model(promoted_model_name)
        saved_run = promoted.saved_run
        model_dataset = saved_run.dataset_name
        model_subfolder = "promoted"
        model_name = promoted.manifest.name
        runtime = initialize_runtime(
            saved_run=saved_run,
            tiling_size=options["tiling_size"],
            checkpoint_path=promoted.weights_path,
            noise_model_path=promoted.noise_model_path,
            noise_model_norm_prm_path=promoted.noise_model_norm_prm_path,
        )
    else:
        run_dir = resolve_run_dir(dataset_name=model_dataset, subfolder=model_subfolder, exp_name=model_name)
        saved_run = load_saved_run(run_dir)
        runtime = initialize_runtime(
            saved_run=saved_run,
            best_or_last=options["best_or_last"],
            epoch_number=options["epoch_number"],
            tiling_size=options["tiling_size"],
        )
    if saved_run.is_lvae:
        assert options["lvae_num_samples"] is not None, (
            "for LVAE prediction, number of samples needs to be specified"
        )

    data_norm = saved_run.data_norm_prm
    clip = False
    if isinstance(data_norm, dict):
        clip = data_norm.get("clip", False)
        if isinstance(clip, bool) and clip is True:
            clip = 0
    model_norm = saved_run.model_norm_prm

    tiling_size = runtime.tiling_size
    upsamp = saved_run.upsampling_factor
    print(f"Found upsampling factor to be: {upsamp}\n")
    print(f"Tiling size: {_format_tiling_size_for_display(options['tiling_size'], tiling_size)}\n")

    context_length = saved_run.context_length
    if context_length is not None:
        print(f"Found context length to be: {context_length}\n")

    data_path, list_files, name_file = resolve_prediction_inputs(
        data_path,
        filters=options["filters"],
        skip_if_contain=options["skip_if_contain"],
    )
    if options["limit_n_imgs"] is not None:
        list_files = list_files[: options["limit_n_imgs"]]
    print(f"Found #{len(list_files)} files.")

    input_dir = data_path if data_path.is_dir() else data_path.parent
    source_name = _source_name(data_path)

    if output_policy.mode == "in_place":
        save_folder = input_dir
    elif output_policy.mode == "folder_inside":
        # Directory inputs already provide their own source context. Single-file
        # inputs need source identity in the generated folder name so multiple
        # files from the same parent remain distinguishable.
        folder_source_name = source_name if data_path.is_file() else None
        prediction_folder_name = _prediction_folder_name(
            source_name=folder_source_name,
            model_subfolder=model_subfolder,
            model_name=model_name,
        )
        save_folder = create_save_folder(path=input_dir / prediction_folder_name)
    elif output_policy.mode == "folder_outside":
        prediction_folder_name = _prediction_folder_name(
            source_name=source_name,
            model_subfolder=model_subfolder,
            model_name=model_name,
        )
        save_folder = create_save_folder(path=input_dir.parent / prediction_folder_name)
    elif output_policy.mode == "folder":
        assert output_policy.save_folder is not None
        save_folder = create_save_folder(path=output_policy.save_folder)
    else:
        save_folder = create_save_folder(
            path=Paths().inference_output_dir(
                source_name=source_name,
                model_name=model_name,
            )
        )

    for idx, file in enumerate(list_files):
        print(f"File {idx+1}/{max(1, len(list_files))}: {file}")

        file_path = data_path / file
        img = imread(file_path)
        img = normalize_inp(img, clip, data_norm, model_norm)
        img, timelapse, volumetric = make_4d(img, options["stack_selection_idx"], options["timelapse_max"])
        print(img.shape)

        crop_size = options["crop_size"]
        if crop_size is not None:
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)
            original_size = img.shape[-2:]
            img = crop_center(img, crop_size)

        if options["fill_factor"] is not None and options["downsamp"] is None:
            raise ValueError(
                "`apply.fill_factor` requires `apply.downsamp` to be set."
            )

        if options["downsamp"] is not None:
            img = _ensure_shape(img, options["downsamp"])
            if options["fill_factor"] is None:
                img = img[..., :: options["downsamp"], :: options["downsamp"]]
            else:
                resolved_fill_factor = _resolve_fill_factor_for_multiple_apply_downsampling(
                    downsamp=options["downsamp"],
                    fill_factor=options["fill_factor"],
                )
                downsampling_prm = {
                    "downsamp_factor": int(options["downsamp"]),
                    "downsamp_method": "multiple",
                    "multiple_prm": {
                        "fill_factor": resolved_fill_factor,
                        "random": False,
                    },
                }
                img, _ = generate_downsamp_inp(img, downsampling_prm)

        resolved_ch_out = None
        if img.ndim >= 4 and img.shape[1] > 1:
            # Align apply behavior with evaluate: multi-channel inputs predict one target channel by default.
            resolved_ch_out = 1

        pred_stack, samples_stack = predict_4d_stack(
            runtime.model,
            img,
            timelapse=timelapse,
            ch_out=resolved_ch_out,
            device=runtime.device,
            is_lvae=saved_run.is_lvae,
            tiling_size=tiling_size,
            lvae_num_samples=options["lvae_num_samples"],
            lvae_save_samples=options["lvae_save_samples"],
            upsamp=upsamp,
            context_length=context_length,
            dark_frame_context_length=options["dark_frame_context_length"],
            verbose=True,
            progress=progress,
        )

        if crop_size is not None and options["keep_original_shape"]:
            pad_width = (
                max(0, original_size[0] - crop_size[0]),
                max(0, original_size[1] - crop_size[1]),
            )
            pred_stack = center_pad(pred_stack, pad_width)

            if saved_run.is_lvae and options["lvae_save_samples"] and samples_stack is not None:
                samples_stack = center_pad(samples_stack, pad_width)

        if options["denormalize_output"]:
            pred_stack = denormalize_pred(pred_stack, data_norm, model_norm)
            if saved_run.is_lvae and options["lvae_save_samples"] and samples_stack is not None:
                for sample_id in range(samples_stack.shape[0]):
                    samples_stack[sample_id] = denormalize_pred(samples_stack[sample_id], data_norm, model_norm)
        pred_stack = inverse_make_4d(pred_stack, volumetric, timelapse, lvae_samples=False)
        tosave = {"pred": pred_stack.astype(np.float32)}

        if options["apply_color_code"] and volumetric:
            try:
                if context_length is not None and not options["dark_frame_context_length"]:
                    pred_stack = pred_stack[:, context_length // 2 : -context_length // 2]
                pred_stack_color_coded = create_color_coded_image(
                    pred_stack,
                    colormap=color_code_prm.get("colormap", "turbo"),
                    stack_order="ZTYX",
                )
                pred_stack_color_coded = enhance_contrast(
                    pred_stack_color_coded,
                    color_code_prm.get("saturation", 0.35),
                )
                if color_code_prm.get("add_colorbar", True):
                    zmax = (pred_stack.shape[0] - 1) * color_code_prm.get("zstep", 0)
                    pred_stack_color_coded = add_colorbar(pred_stack_color_coded, zmax=zmax)
                tosave["pred_colorCoded"] = pred_stack_color_coded

            except Exception as e:
                warnings.warn(f"Failed to apply color coding: {e}")

        if saved_run.is_lvae and options["lvae_save_samples"] and samples_stack is not None:
            samples_stack = inverse_make_4d(samples_stack, volumetric, timelapse, lvae_samples=True)
            tosave["samples"] = samples_stack.astype(np.float32)

        if name_file is None:
            img_name = file.split('.')[0]
        else:
            img_name = name_file.split('.')[0]
        if save_input:
            tosave["inp"] = img.astype(np.float32)

        save_outputs(tosave, save_folder, img_name)
