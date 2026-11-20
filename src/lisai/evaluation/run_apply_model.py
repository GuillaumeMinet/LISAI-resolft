"""High-level entrypoint for applying a saved model to arbitrary input files.

This module orchestrates the evaluation pipeline for prediction-only usage:
resolve the saved run, initialize the inference runtime, prepare input files,
run stack inference, and save outputs.
"""

import warnings
from typing import Mapping
from pathlib import Path

import numpy as np
from tifffile import imread

from lisai.config.models.inference import ApplyDefaults, InputNormalization, TilingSizePolicy
from lisai.config.progress import resolve_progress_bar
from lisai.data.utils import center_pad, crop_center
from lisai.evaluation.defaults import resolve_apply_output_policy, resolve_apply_save_input
from lisai.evaluation.inference.normalization import denormalize_pred, normalize_apply_input
from lisai.evaluation.inference.progress import InferenceProgress
from lisai.evaluation.inference.results import PredictionInfo, PredictionResult
from lisai.evaluation.inference.shape import inverse_make_4d, make_4d
from lisai.evaluation.inference.stack import predict_4d_stack
from lisai.evaluation.io import resolve_prediction_inputs, save_outputs
from lisai.evaluation.runtime import initialize_runtime
from lisai.evaluation.saved_run import load_saved_run, resolve_run_dir
from lisai.infra.fs import prepare_output_folder
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


def _create_apply_save_folder(
    path: Path,
    *,
    overwrite: bool,
    reuse_folder: bool = False,
    progress: InferenceProgress,
) -> Path:
    requested = Path(path)
    if overwrite and reuse_folder:
        raise ValueError("--overwrite cannot be combined with --reuse-folder or --skip-existing.")

    if overwrite:
        if_exists_policy = "overwrite"
    elif reuse_folder:
        if_exists_policy = "reuse"
    else:
        if_exists_policy = "numbered"

    resolution = prepare_output_folder(
        requested,
        if_exists_policy=if_exists_policy,
    )
    progress.write(f"\n{resolution.message()}\n")
    return resolution.path


def _apply_img_name(file: str, name_file: str | None) -> str:
    source = name_file if name_file is not None else file
    return Path(source).stem


def _existing_apply_outputs(save_folder: Path, img_name: str) -> list[Path]:
    save_folder = Path(save_folder)
    if not save_folder.exists():
        return []
    prefix = f"{img_name}_"
    return sorted(
        path
        for path in save_folder.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() == ".tif"
    )


def _prediction_output_path(save_folder: Path, img_name: str) -> Path:
    return Path(save_folder) / f"{img_name}_pred.tif"


def _format_existing_output_summary(collisions: list[tuple[str, list[Path]]]) -> str:
    formatted = []
    for img_name, paths in collisions[:5]:
        formatted.append(f"{img_name} ({len(paths)} file{'s' if len(paths) != 1 else ''})")
    if len(collisions) > 5:
        formatted.append(f"... and {len(collisions) - 5} more")
    return ", ".join(formatted)


def _raise_if_reuse_folder_would_overwrite(
    list_files: list[str],
    *,
    save_folder: Path,
    name_file: str | None,
) -> None:
    collisions = []
    for file in list_files:
        img_name = _apply_img_name(file, name_file)
        existing = _existing_apply_outputs(save_folder, img_name)
        if existing:
            collisions.append((img_name, existing))

    if collisions:
        summary = _format_existing_output_summary(collisions)
        raise FileExistsError(
            f"Existing apply outputs found in {save_folder} for selected input(s): {summary}. "
            "Use --skip-existing to continue, --overwrite to replace the folder, "
            "or choose another save folder."
        )


def _filter_existing_apply_outputs(
    list_files: list[str],
    *,
    save_folder: Path,
    name_file: str | None,
    progress: InferenceProgress,
) -> list[str]:
    remaining = []
    skipped = []
    partial = []

    for file in list_files:
        img_name = _apply_img_name(file, name_file)
        existing = _existing_apply_outputs(save_folder, img_name)
        if not existing:
            remaining.append(file)
            continue
        if _prediction_output_path(save_folder, img_name).exists():
            skipped.append(file)
            continue
        partial.append((img_name, existing))

    if partial:
        summary = _format_existing_output_summary(partial)
        raise FileExistsError(
            f"Partial existing apply outputs found in {save_folder}: {summary}. "
            "Refusing to guess whether these inputs are complete. Remove those files, "
            "use --overwrite, or choose another save folder."
        )

    if skipped:
        progress.write(f"Skipping {len(skipped)} file(s) with existing predictions.")
    return remaining


def _resolve_apply_files_for_output(
    list_files: list[str],
    *,
    save_folder: Path,
    name_file: str | None,
    limit_n_imgs: int | None,
    reuse_folder: bool,
    skip_existing: bool,
    progress: InferenceProgress,
) -> list[str]:
    selected = list(list_files)
    if skip_existing:
        progress.write(f"Found #{len(selected)} candidate files.")
        selected = _filter_existing_apply_outputs(
            selected,
            save_folder=save_folder,
            name_file=name_file,
            progress=progress,
        )
        if limit_n_imgs is not None:
            selected = selected[:limit_n_imgs]
        if not selected:
            progress.write(f"All candidate files already have predictions in {save_folder}.")
        else:
            progress.write(f"Processing {len(selected)} remaining file(s).")
        return selected

    if limit_n_imgs is not None:
        selected = selected[:limit_n_imgs]
    if reuse_folder:
        _raise_if_reuse_folder_would_overwrite(
            selected,
            save_folder=save_folder,
            name_file=name_file,
        )
    return selected


def compute_prediction_array(
    img: np.ndarray, *, saved_run, runtime, cfg: ApplyDefaults,
    normalization: InputNormalization | None = None,
    progress: InferenceProgress | None = None, verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, PredictionInfo, np.ndarray]:
    """The single array computation path used by public predict and legacy apply.

    Returns final prediction, optional samples, processed input, alignment info,
    and internal ZTYX prediction for optional CLI color visualization.
    """
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3, 4):
        raise ValueError("Prediction expects an image with axes YX, TYX or TZYX.")
    selected = cfg.input.stack_selection_idx
    if img.ndim >= 3:
        source_indices = list(range(img.shape[0]))
        if selected is not None:
            if isinstance(selected, int) and not 0 <= selected < img.shape[0]:
                raise ValueError(f"stack_selection_idx {selected} outside input time axis 0..{img.shape[0]-1}.")
            source_indices = source_indices[selected:selected+1] if isinstance(selected, int) else [source_indices[i] for i in selected]
        elif cfg.input.timelapse_max is not None:
            source_indices = source_indices[:cfg.input.timelapse_max]
    else:
        source_indices = [0]

    prepared, timelapse, volumetric = make_4d(img, selected, cfg.input.timelapse_max)
    context_length = saved_run.context_length
    if context_length is not None and len(source_indices) < context_length:
        raise ValueError(
            f"Model requires at least {context_length} input frames, but received {len(source_indices)}."
        )
    data_norm = saved_run.data_norm_prm
    model_norm = saved_run.model_norm_prm
    clip = data_norm.get("clip", False) if isinstance(data_norm, dict) else False
    if clip is True:
        clip = 0
    mode = normalization if normalization is not None else cfg.input.normalization
    normalized = normalize_apply_input(prepared, clip, data_norm, model_norm, mode)
    # A raw-zero input must be transformed consistently with real observations.
    # Image-mode statistics are calculated only on real selected pixels.
    normalized_black = float(normalize_apply_input(
        np.zeros((1, 1), dtype=np.float32), clip, data_norm, model_norm,
        mode, reference_image=prepared,
    )[0, 0]) if context_length is not None and cfg.inference.temporal_edge_mode == "black_input" else 0.0

    crop_size = cfg.inference.crop_size
    if crop_size is not None:
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        original_size = normalized.shape[-2:]
        normalized = crop_center(normalized, crop_size)

    if cfg.inference.fill_factor is not None and cfg.inference.downsamp is None:
        raise ValueError("`apply.fill_factor` requires `apply.downsamp` to be set.")
    if cfg.inference.downsamp is not None:
        normalized = _ensure_shape(normalized, cfg.inference.downsamp)
        if cfg.inference.fill_factor is None:
            normalized = normalized[..., ::cfg.inference.downsamp, ::cfg.inference.downsamp]
        else:
            fill_factor = _resolve_fill_factor_for_multiple_apply_downsampling(
                downsamp=cfg.inference.downsamp, fill_factor=cfg.inference.fill_factor,
            )
            normalized, _ = generate_downsamp_inp(normalized, {
                "downsamp_factor": int(cfg.inference.downsamp),
                "downsamp_method": "multiple",
                "multiple_prm": {"fill_factor": fill_factor, "random": False},
            })

    ch_out = 1 if normalized.ndim >= 4 and normalized.shape[1] > 1 else None
    pred_stack, samples_stack = predict_4d_stack(
        runtime.model, normalized, timelapse=timelapse, ch_out=ch_out,
        device=runtime.device, is_lvae=saved_run.is_lvae,
        tiling_size=runtime.tiling_size, lvae_num_samples=cfg.inference.lvae_num_samples,
        lvae_save_samples=cfg.saving.lvae_save_samples,
        upsamp=saved_run.upsampling_factor, context_length=context_length,
        temporal_edge_mode=cfg.inference.temporal_edge_mode,
        normalized_black_value=normalized_black,
        verbose=verbose, progress=progress,
    )
    if crop_size is not None and cfg.inference.keep_original_shape:
        pad = (max(0, original_size[0] - crop_size[0]), max(0, original_size[1] - crop_size[1]))
        pred_stack = center_pad(pred_stack, pad)
        if samples_stack is not None:
            samples_stack = center_pad(samples_stack, pad)

    if cfg.postprocess.denormalize:
        pred_stack = denormalize_pred(pred_stack, data_norm, model_norm)
        if samples_stack is not None:
            samples_stack = denormalize_pred(samples_stack, data_norm, model_norm)

    black_positions: tuple[int, ...] = ()
    if context_length is not None:
        half = context_length // 2
        if cfg.inference.temporal_edge_mode == "drop":
            if half:
                source_indices = source_indices[half:-half]
        elif cfg.inference.temporal_edge_mode == "black_output":
            black_positions = tuple(list(range(half)) + list(range(len(source_indices)-half, len(source_indices))))
            # Do this AFTER denormalization; zeros in model space are not raw black.
            pred_stack[:, list(black_positions), ...] = 0
            if samples_stack is not None:
                samples_stack[:, :, list(black_positions), ...] = 0

    internal_pred = np.asarray(pred_stack, dtype=np.float32)
    prediction = inverse_make_4d(internal_pred, volumetric, timelapse)
    samples = (
        inverse_make_4d(np.asarray(samples_stack, dtype=np.float32), volumetric, timelapse, lvae_samples=True)
        if samples_stack is not None else None
    )
    return (
        np.asarray(prediction, dtype=np.float32), samples,
        np.asarray(normalized, dtype=np.float32),
        PredictionInfo(tuple(source_indices), black_positions), internal_pred,
    )


def run_apply_model(
    cfg: ApplyDefaults,
    *,
    model_dataset: str,
    model_subfolder: str,
    model_name: str,
    data_path: Path,
    paths: Paths,
    promoted_model_name: str | None = None,
    saved_run_dir: Path | None = None,
    progress_bar: bool | None = None,
    overwrite: bool = False,
    reuse_folder: bool = False,
    skip_existing: bool = False,
    input_normalization: InputNormalization | Mapping[str, object] | None = None,
    device: str | None = None,
    return_result: bool = False,
):
    """Apply a saved model using a fully resolved typed apply config.

    ``input_normalization`` replaces the YAML normalization choice for this
    invocation only; model weights and saved normalization remain unchanged.
    """
    output_policy = resolve_apply_output_policy(cfg)
    save_input = resolve_apply_save_input(cfg, output_policy=output_policy)
    color_code_cfg = cfg.postprocess.color_code
    normalization = (
        cfg.input.normalization
        if input_normalization is None
        else InputNormalization.model_validate(input_normalization)
    )
    progress = InferenceProgress(
        enabled=resolve_progress_bar(True, progress_bar, stg=paths.settings)
    )

    data_path = Path(data_path)
    if promoted_model_name is not None and saved_run_dir is not None:
        raise ValueError("Choose promoted_model_name or saved_run_dir, not both.")
    if promoted_model_name is not None:
        from lisai.promoted_models import load_promoted_model

        promoted = load_promoted_model(promoted_model_name, paths=paths)
        saved_run = promoted.saved_run
        model_dataset = saved_run.dataset_name
        model_subfolder = "promoted"
        model_name = promoted.manifest.name
        runtime = initialize_runtime(
            saved_run=saved_run,
            paths=paths,
            tiling_size=cfg.inference.tiling_size,
            device=device,
            checkpoint_path=promoted.weights_path,
            noise_model_path=promoted.noise_model_path,
            noise_model_norm_prm_path=promoted.noise_model_norm_prm_path,
        )
    else:
        run_dir = (Path(saved_run_dir) if saved_run_dir is not None else resolve_run_dir(
            dataset_name=model_dataset, subfolder=model_subfolder, exp_name=model_name, paths=paths
        ))
        saved_run = load_saved_run(run_dir, paths=paths)
        runtime = initialize_runtime(
            saved_run=saved_run,
            paths=paths,
            best_or_last=cfg.checkpoint.best_or_last,
            epoch_number=cfg.checkpoint.epoch_number,
            tiling_size=cfg.inference.tiling_size,
            device=device,
        )
    if saved_run.is_lvae:
        assert cfg.inference.lvae_num_samples is not None, (
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
    print(f"Tiling size: {_format_tiling_size_for_display(cfg.inference.tiling_size, tiling_size)}\n")

    context_length = saved_run.context_length
    if context_length is not None:
        print(f"Found context length to be: {context_length}\n")

    data_path, list_files, name_file = resolve_prediction_inputs(
        data_path,
        filters=cfg.input.filters,
        skip_if_contain=cfg.input.skip_if_contain,
    )

    input_dir = data_path if data_path.is_dir() else data_path.parent
    source_name = _source_name(data_path)
    overwrite = bool(overwrite)
    skip_existing = bool(skip_existing)
    reuse_folder = bool(reuse_folder) or skip_existing
    if overwrite and reuse_folder:
        raise ValueError("--overwrite cannot be combined with --reuse-folder or --skip-existing.")

    if output_policy.mode == "in_place":
        if overwrite or reuse_folder:
            raise ValueError(
                "--overwrite, --reuse-folder, and --skip-existing cannot be used with "
                "in-place apply output because it targets the input folder."
            )
        save_folder = input_dir
        progress.write(f"Saving outputs in place: {save_folder}")
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
        save_folder = _create_apply_save_folder(
            input_dir / prediction_folder_name,
            overwrite=overwrite,
            reuse_folder=reuse_folder,
            progress=progress,
        )
    elif output_policy.mode == "folder_outside":
        prediction_folder_name = _prediction_folder_name(
            source_name=source_name,
            model_subfolder=model_subfolder,
            model_name=model_name,
        )
        save_folder = _create_apply_save_folder(
            input_dir.parent / prediction_folder_name,
            overwrite=overwrite,
            reuse_folder=reuse_folder,
            progress=progress,
        )
    elif output_policy.mode == "folder":
        assert output_policy.save_folder is not None
        save_folder = _create_apply_save_folder(
            output_policy.save_folder,
            overwrite=overwrite,
            reuse_folder=reuse_folder,
            progress=progress,
        )
    else:
        save_folder = _create_apply_save_folder(
            paths.inference_output_dir(
                source_name=source_name,
                model_name=model_name,
            ),
            overwrite=overwrite,
            reuse_folder=reuse_folder,
            progress=progress,
        )

    list_files = _resolve_apply_files_for_output(
        list_files,
        save_folder=save_folder,
        name_file=name_file,
        limit_n_imgs=cfg.input.limit_n_imgs,
        reuse_folder=reuse_folder,
        skip_existing=skip_existing,
        progress=progress,
    )
    if not skip_existing:
        print(f"Found #{len(list_files)} files.")
    # A .tif/.tiff pair with the same stem, or any other naming collision,
    # must not silently overwrite the first prediction in a directory batch.
    names = [_apply_img_name(file, name_file) for file in list_files]
    if len(names) != len(set(name.casefold() for name in names)):
        raise ValueError("Multiple input TIFFs would produce the same prediction filename.")
    if not list_files:
        if return_result:
            return PredictionResult(Path(save_folder), (), {})
        return
    saved_predictions: list[Path] = []
    info_by_path: dict[Path, PredictionInfo] = {}

    for idx, file in enumerate(list_files):
        print(f"File {idx+1}/{max(1, len(list_files))}: {file}")

        file_path = data_path / file
        img = imread(file_path)
        pred_stack, samples_stack, normalized_input, info, internal_pred = compute_prediction_array(
            img, saved_run=saved_run, runtime=runtime, cfg=cfg,
            normalization=normalization, progress=progress, verbose=True,
        )
        volumetric = img.ndim == 4
        tosave = {"pred": pred_stack}

        if color_code_cfg.enabled and volumetric:
            try:
                from lisai.evaluation.visualization.z_projection import (
                    add_colorbar,
                    create_color_coded_image,
                    enhance_contrast,
                )

                pred_stack_color_coded = create_color_coded_image(
                    internal_pred,
                    colormap=color_code_cfg.colormap,
                    stack_order="ZTYX",
                )
                pred_stack_color_coded = enhance_contrast(
                    pred_stack_color_coded,
                    color_code_cfg.saturation,
                )
                if color_code_cfg.add_colorbar:
                    zmax = (internal_pred.shape[0] - 1) * color_code_cfg.zstep
                    pred_stack_color_coded = add_colorbar(pred_stack_color_coded, zmax=zmax)
                tosave["pred_colorCoded"] = pred_stack_color_coded

            except Exception as e:
                warnings.warn(f"Failed to apply color coding: {e}")

        if saved_run.is_lvae and cfg.saving.lvae_save_samples and samples_stack is not None:
            tosave["samples"] = samples_stack.astype(np.float32)

        img_name = _apply_img_name(file, name_file)
        if save_input:
            tosave["inp"] = normalized_input

        written = save_outputs(tosave, save_folder, img_name)
        if return_result:
            if written is None or "pred" not in written:
                raise RuntimeError("Inference did not return the saved prediction path.")
            prediction_path = written["pred"]
            saved_predictions.append(prediction_path)
            info_by_path[prediction_path] = info

    if return_result:
        return PredictionResult(Path(save_folder), tuple(saved_predictions), info_by_path)
