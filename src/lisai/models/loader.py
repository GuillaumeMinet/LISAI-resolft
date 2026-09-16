from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Protocol

import torch

from lisai.infra.paths import Paths
from lisai.models.params import AnyModelParams, LVAEParams

from .registry import get_model_class

logger = logging.getLogger("lisai.model")


class TrainingModelLoadSpec(Protocol):
    architecture: str
    parameters: AnyModelParams | None
    mode: str
    patch_size: int | None
    downsamp_factor: int | None
    origin_run_dir: Path | None
    checkpoint_method: str | None
    checkpoint_selector: str | None
    checkpoint_epoch: int | None
    checkpoint_filename: str | None



def init_model(
    architecture: str,
    model_prm: AnyModelParams,
    device: torch.device,
    *,
    model_norm_prm: dict | None = None,
    noise_model=None,
    img_shape: int | None = None,
):
    ModelClass = get_model_class(architecture)

    if architecture == "lvae":
        if not isinstance(model_prm, LVAEParams):
            raise TypeError(f"LVAE expects LVAEParams, got {type(model_prm)!r}.")

        missing = []
        if model_norm_prm is None:
            missing.append("model_norm_prm")
        if img_shape is None:
            missing.append("img_shape")
        if noise_model is None:
            missing.append("noise_model")
        if missing:
            raise ValueError(f"LVAE initialization missing required args: {', '.join(missing)}")

        model = ModelClass(
            model_prm,
            device=device,
            norm_prm=model_norm_prm,
            noise_model=noise_model,
            img_shape=(img_shape, img_shape),
        )
    else:
        model = ModelClass(model_prm)

    return model.to(device)



def _compute_img_shape(patch_size: int | None, downsamp_factor: int | None) -> int | None:
    if patch_size is None:
        return None
    ds = downsamp_factor or 1
    try:
        return int(patch_size) // int(ds)
    except Exception:
        return None



def _origin_checkpoint_path(spec: TrainingModelLoadSpec) -> Path:
    if spec.origin_run_dir is None:
        raise ValueError("origin_run_dir is required to load a checkpoint.")

    origin_dir = Path(spec.origin_run_dir)
    paths = Paths()

    if spec.checkpoint_filename:
        return paths.checkpoint_path(
            run_dir=origin_dir,
            model_name=spec.checkpoint_filename,
        )

    method = spec.checkpoint_method or "state_dict"
    selector = spec.checkpoint_selector
    epoch = spec.checkpoint_epoch
    mode = spec.mode

    if selector is None and epoch is None:
        if mode == "continue_training":
            selector = "last"
        elif mode == "retrain":
            selector = "best"
        else:
            selector = "best"

    return paths.checkpoint_path(
        run_dir=origin_dir,
        load_method=method,
        best_or_last=selector,
        epoch_number=epoch,
    )



def prepare_model_for_training(
    *,
    spec: TrainingModelLoadSpec,
    device: torch.device,
    model_norm_prm: dict | None = None,
    noise_model=None,
):
    """
    Build (and optionally load) model based on the provided training load spec.
    """
    arch = spec.architecture
    model_prm = spec.parameters

    if not arch:
        raise ValueError("Model load spec architecture is empty")
    if model_prm is None:
        raise ValueError("Model load spec parameters are missing")

    if arch == "lvae" and noise_model is None:
        raise ValueError("Need noise model to perform lvae training")

    should_load = spec.mode in {"continue_training", "retrain"}

    if should_load:
        ckpt_method = spec.checkpoint_method or "state_dict"
        origin_ckpt = _origin_checkpoint_path(spec)
        logger.info(f"Loading checkpoint from: {origin_ckpt}")

        if not origin_ckpt.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {origin_ckpt}")

        if ckpt_method == "full_model":
            model = torch.load(origin_ckpt, map_location=device)
            if spec.mode == "continue_training":
                warnings.warn(
                    "continue_training with full_model load: optimizer/scheduler state handling depends on your trainer."
                )
            return model, None

    model = init_model(
        architecture=arch,
        model_prm=model_prm,
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
        img_shape=_compute_img_shape(spec.patch_size, spec.downsamp_factor),
    )

    state = None
    if should_load:
        origin_ckpt = _origin_checkpoint_path(spec)
        loaded = torch.load(origin_ckpt, map_location=device, weights_only=True)

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
            state = loaded
        elif isinstance(loaded, dict):
            model.load_state_dict(loaded)
            state = None
        else:
            raise ValueError(f"Unsupported checkpoint type at {origin_ckpt}: {type(loaded)}")

    return model, state
