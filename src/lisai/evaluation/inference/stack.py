from __future__ import annotations

import numpy as np
import torch

from lisai.evaluation.inference.engine import predict
from lisai.evaluation.inference.progress import ProgressLike, ensure_progress


def infer_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    is_lvae: bool,
    tiling_size: int | None,
    num_samples: int | None,
    upsamp: int,
    ch_out: int | None,
    progress: ProgressLike | None = None,
    progress_level: int = 0,
):
    return predict(
        model,
        x,
        is_lvae=is_lvae,
        tiling_size=tiling_size,
        num_samples=num_samples,
        upsamp=upsamp,
        ch_out=ch_out,
        progress=progress,
        progress_level=progress_level,
    )


def _build_temporal_input(
    img: np.ndarray,
    *,
    z: int,
    t: int,
    context_length: int | None,
    dark_frame_context_length: bool,
    verbose: bool = False,
    progress: ProgressLike | None = None,
) -> np.ndarray | None:
    if context_length is None:
        x = img[z, t, ...]
        return np.expand_dims(x, axis=(0, 1))  # [B=1, C=1, H, W]

    start = t - context_length // 2
    end = t + context_length // 2 + 1
    if start < 0 or end > img.shape[1]:
        if dark_frame_context_length:
            x = np.zeros((1, context_length, img.shape[-2], img.shape[-1]), dtype=img.dtype)
            x[:, context_length // 2] = img[z, t, ...]
            return x
        if verbose:
            message = f"Skipping frame {t} because not enough context_length"
            write_progress = getattr(progress, "write", None)
            if write_progress is not None:
                write_progress(message)
            else:
                print(message)
        return None

    x = img[z, start:end, ...]
    return np.expand_dims(x, axis=0)  # [B=1, C=T, H, W]


def predict_4d_stack(
    model: torch.nn.Module,
    img: np.ndarray,
    *,
    timelapse: bool,
    ch_out: int | None,
    device: torch.device,
    is_lvae: bool,
    tiling_size: int | None,
    lvae_num_samples: int | None,
    lvae_save_samples: bool,
    upsamp: int,
    context_length: int | None,
    dark_frame_context_length: bool,
    verbose: bool = False,
    progress: ProgressLike | None = None,
):
    progress = ensure_progress(progress)
    if timelapse:
        output_shape = (*img.shape[:-2], img.shape[-2] * upsamp, img.shape[-1] * upsamp)
    else:
        # Non-timelapse inputs use axis 1 as channels, not time.
        output_shape = (img.shape[0], 1, img.shape[-2] * upsamp, img.shape[-1] * upsamp)
    pred_stack = np.empty(shape=output_shape)
    samples_stack = None
    if is_lvae and lvae_save_samples:
        samples_stack = np.empty(shape=(lvae_num_samples, *output_shape))

    n_timepoints = img.shape[1] if timelapse else 1
    n_stack_items = img.shape[0] * n_timepoints
    show_stack_progress = verbose and n_stack_items > 1
    stack_iter = ((z, t) for z in range(img.shape[0]) for t in range(n_timepoints))
    if show_stack_progress:
        stack_iter = progress.track(
            stack_iter,
            total=n_stack_items,
            desc="Stack inference",
            level=0,
            leave=True,
        )
    batch_progress_level = 1 if show_stack_progress else 0

    for z, t in stack_iter:
        if timelapse:
            x_np = _build_temporal_input(
                img,
                z=z,
                t=t,
                context_length=context_length,
                dark_frame_context_length=dark_frame_context_length,
                verbose=verbose,
                progress=progress,
            )
        else:
            x_np = np.expand_dims(img[z, ...], axis=0)  # [B=1, C, H, W]
        if x_np is None:
            continue

        x = torch.from_numpy(x_np).to(device)
        resolved_ch_out = ch_out
        if resolved_ch_out is None and context_length is not None:
            resolved_ch_out = 1
        outputs = infer_batch(
            model,
            x,
            is_lvae=is_lvae,
            tiling_size=tiling_size,
            num_samples=lvae_num_samples,
            upsamp=upsamp,
            ch_out=resolved_ch_out,
            progress=progress,
            progress_level=batch_progress_level,
        )

        pred_stack[z, t, ...] = outputs.get("prediction")
        if is_lvae and lvae_save_samples and samples_stack is not None:
            samples_stack[:, z, t, ...] = outputs.get("samples")

    return pred_stack, samples_stack
