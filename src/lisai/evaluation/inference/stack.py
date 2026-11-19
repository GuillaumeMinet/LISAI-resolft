"""Inference over images, time series and volumes using one model execution path."""

from __future__ import annotations

import numpy as np
import torch

from lisai.evaluation.inference.engine import predict
from lisai.evaluation.inference.progress import ProgressLike, ensure_progress


def infer_batch(
    model: torch.nn.Module, x: torch.Tensor, *, is_lvae: bool,
    tiling_size: int | None, num_samples: int | None, upsamp: int,
    ch_out: int | None, progress: ProgressLike | None = None,
    progress_level: int = 0,
):
    return predict(
        model, x, is_lvae=is_lvae, tiling_size=tiling_size,
        num_samples=num_samples, upsamp=upsamp, ch_out=ch_out,
        progress=progress, progress_level=progress_level,
    )


def _build_temporal_input(
    img: np.ndarray, *, z: int, t: int,
    context_length: int | None, temporal_edge_mode: str = "drop",
    normalized_black_value: float = 0.0,
    dark_frame_context_length: bool | None = None,
    verbose: bool = False, progress: ProgressLike | None = None,
) -> np.ndarray | None:
    """Include all real context neighbors; fill only missing ones for black_input."""
    if dark_frame_context_length is not None:
        if temporal_edge_mode != "drop":
            raise ValueError("Specify only one temporal edge control.")
        temporal_edge_mode = "black_input" if dark_frame_context_length else "drop"
    if context_length is None:
        return img[z, t, ...][None, None, ...]
    half = context_length // 2
    start, stop = t - half, t + half + 1
    if start < 0 or stop > img.shape[1]:
        if temporal_edge_mode != "black_input":
            if verbose:
                message = f"Skipping frame {t} because not enough context_length"
                write = getattr(progress, "write", None)
                write(message) if write is not None else print(message)
            return None
        x = np.full(
            (1, context_length, *img.shape[-2:]),
            normalized_black_value, dtype=img.dtype,
        )
        src_start, src_stop = max(0, start), min(img.shape[1], stop)
        x[0, src_start-start:src_stop-start, ...] = img[z, src_start:src_stop, ...]
        return x
    return img[z, start:stop, ...][None, ...]


def predict_4d_stack(
    model: torch.nn.Module, img: np.ndarray, *, timelapse: bool,
    ch_out: int | None, device: torch.device, is_lvae: bool,
    tiling_size: int | None, lvae_num_samples: int | None,
    lvae_save_samples: bool, upsamp: int, context_length: int | None,
    temporal_edge_mode: str = "drop", normalized_black_value: float = 0.0,
    dark_frame_context_length: bool | None = None,
    verbose: bool = False, progress: ProgressLike | None = None,
):
    """Return [Z,T,Y,X] or [Z,1,Y,X]; temporal drop physically removes edges."""
    progress = ensure_progress(progress)
    if dark_frame_context_length is not None:
        if temporal_edge_mode != "drop":
            raise ValueError("Specify temporal_edge_mode or dark_frame_context_length, not both.")
        temporal_edge_mode = "black_input" if dark_frame_context_length else "drop"
    if temporal_edge_mode not in ("drop", "black_output", "black_input"):
        raise ValueError(f"Unknown temporal_edge_mode: {temporal_edge_mode!r}")
    if context_length is not None:
        if context_length < 1 or context_length % 2 != 1:
            raise ValueError("Temporal context_length must be a positive odd integer.")
        if not timelapse:
            raise ValueError(f"Model requires at least {context_length} input frames, but received 1.")
        if img.shape[1] < context_length:
            raise ValueError(
                f"Model requires at least {context_length} input frames, but received {img.shape[1]}."
            )
    n_timepoints = img.shape[1] if timelapse else 1
    half = context_length // 2 if context_length is not None else 0
    do_drop = context_length is not None and temporal_edge_mode == "drop"
    output_time = n_timepoints - 2 * half if do_drop else n_timepoints
    h, w = img.shape[-2] * upsamp, img.shape[-1] * upsamp
    pred_stack = np.zeros((img.shape[0], output_time, h, w), dtype=np.float32)
    samples_stack = None
    if is_lvae and lvae_save_samples:
        samples_stack = np.zeros((lvae_num_samples, *pred_stack.shape), dtype=np.float32)

    n_stack_items = img.shape[0] * n_timepoints
    show_stack_progress = verbose and n_stack_items > 1
    stack_iter = ((z, t) for z in range(img.shape[0]) for t in range(n_timepoints))
    if show_stack_progress:
        stack_iter = progress.track(
            stack_iter, total=n_stack_items, desc="Stack inference", level=0, leave=True,
        )
    for z, t in stack_iter:
        if timelapse:
            x_np = _build_temporal_input(
                img, z=z, t=t, context_length=context_length,
                temporal_edge_mode=temporal_edge_mode,
                normalized_black_value=normalized_black_value,
                verbose=verbose, progress=progress,
            )
        else:
            x_np = img[z, ...][None, ...]
        if x_np is None:
            continue
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
        resolved_ch_out = 1 if ch_out is None and context_length is not None else ch_out
        outputs = infer_batch(
            model, x, is_lvae=is_lvae, tiling_size=tiling_size,
            num_samples=lvae_num_samples, upsamp=upsamp, ch_out=resolved_ch_out,
            progress=progress, progress_level=1 if show_stack_progress else 0,
        )
        output_t = t-half if do_drop else t
        pred_stack[z, output_t, ...] = np.asarray(outputs["prediction"]).squeeze()
        if samples_stack is not None:
            samples_stack[:, z, output_t, ...] = np.asarray(outputs["samples"]).reshape(
                lvae_num_samples, h, w
            )
    return pred_stack, samples_stack
