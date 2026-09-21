"""Live evaluation runtime construction.

This module owns the runtime-side boundary of evaluation. It takes a
`SavedTrainingRun`, resolves the checkpoint to load, materializes the model on a
chosen device, and returns the small `InferenceRuntime` object used by the
entrypoints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.infra.paths.checkpoint_resolution import (
    resolve_checkpoint_path as resolve_existing_checkpoint_path,
)
from lisai.models import load_noise_model, load_noise_model_from_paths
from lisai.models.loader import init_model

from .saved_run import CheckpointMethod, SavedTrainingRun

TilingSizePolicy: TypeAlias = int | Literal["auto", "off"] | None



@dataclass
class InferenceRuntime:
    """Live resources needed to run inference for one evaluation call."""

    model: Any
    device: torch.device
    checkpoint_path: Path
    load_method: CheckpointMethod
    tiling_size: int | None
    resolved_epoch: int | None



def _default_device() -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _compute_img_shape(patch_size: int | None, downsamp_factor: int) -> int | None:
    """Convert saved patch metadata into the image size expected by LVAE models."""
    if patch_size is None:
        return None
    return int(patch_size) // max(1, int(downsamp_factor))



def _resolve_checkpoint_path(
    saved_run: SavedTrainingRun,
    *,
    best_or_last: str,
    epoch_number: int | None,
    paths: Paths,
) -> tuple[CheckpointMethod, Path]:
    """Find the first existing checkpoint matching the requested selector."""
    method, checkpoint_path = resolve_existing_checkpoint_path(
        paths=paths,
        run_dir=saved_run.run_dir,
        load_methods=saved_run.checkpoint_methods,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
        missing_description="a model checkpoint for inference",
    )
    return method, checkpoint_path



def _epoch_from_checkpoint_path(checkpoint_path: Path) -> int | None:
    match = re.search(r"model_epoch_(\d+)", checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))



def _load_state_dict_model(
    saved_run: SavedTrainingRun,
    checkpoint_path: Path,
    device: torch.device,
    paths: Paths,
    *,
    noise_model_path: Path | None = None,
    noise_model_norm_prm_path: Path | None = None,
) -> tuple[Any, int | None]:
    """Instantiate the model structure and load weights from a state-dict checkpoint."""
    model_norm_prm = dict(saved_run.model_norm_prm) if saved_run.model_norm_prm is not None else None
    noise_model = None

    if saved_run.is_lvae:
        if not saved_run.noise_model_name:
            raise ValueError("Saved LVAE run is missing noise_model.name.")
        if noise_model_path is not None:
            noise_model, nm_norm_prm = load_noise_model_from_paths(
                noise_model_path, noise_model_norm_prm_path, device
            )
        else:
            noise_model, nm_norm_prm = load_noise_model(saved_run.noise_model_name, device, paths)
        if model_norm_prm is None and nm_norm_prm is not None:
            model_norm_prm = dict(nm_norm_prm)
        if model_norm_prm is None and saved_run.data_norm_prm is not None:
            model_norm_prm = dict(saved_run.data_norm_prm)

    model = init_model(
        architecture=saved_run.model_architecture,
        model_prm=saved_run.model_parameters,
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
        img_shape=_compute_img_shape(saved_run.patch_size, saved_run.downsamp_factor),
    )

    loaded = torch.load(checkpoint_path, map_location=device)
    resolved_epoch = _epoch_from_checkpoint_path(checkpoint_path)
    if isinstance(loaded, dict):
        epoch = loaded.get("epoch")
        if epoch is not None:
            try:
                resolved_epoch = int(epoch)
            except (TypeError, ValueError):
                pass

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model.load_state_dict(loaded["model_state_dict"])
    elif isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise ValueError(f"Unsupported checkpoint type at {checkpoint_path}: {type(loaded)}")

    model.eval()
    return model, resolved_epoch



def resolve_tiling_size(
    saved_run: SavedTrainingRun,
    tiling_size: TilingSizePolicy = "auto",
) -> int | None:
    """Resolve a public tiling policy into the value consumed by inference."""
    if tiling_size is None:
        return saved_run.default_tiling_size

    if isinstance(tiling_size, str):
        normalized = tiling_size.strip().lower()
        if normalized in {"", "auto"}:
            return saved_run.default_tiling_size
        if normalized in {"off", "none", "disable", "disabled"}:
            return None
        try:
            tiling_size = int(normalized)
        except ValueError:
            raise ValueError(
                "tiling_size must be a positive integer, 'auto', 'off', or null."
            ) from None

    if isinstance(tiling_size, bool):
        raise ValueError("tiling_size must be a positive integer, 'auto', 'off', or null.")

    resolved = int(tiling_size)
    if resolved <= 0:
        raise ValueError("tiling_size must be greater than 0.")
    return resolved



def initialize_runtime(
    *,
    saved_run: SavedTrainingRun,
    device: torch.device | str | None = None,
    best_or_last: str = "best",
    epoch_number: int | None = None,
    tiling_size: TilingSizePolicy = "auto",
    checkpoint_path: str | Path | None = None,
    noise_model_path: str | Path | None = None,
    noise_model_norm_prm_path: str | Path | None = None,
) -> InferenceRuntime:
    """Load the requested checkpoint and build the live inference runtime."""
    paths = Paths(settings)
    resolved_device = _default_device() if device is None else torch.device(device)
    if checkpoint_path is None:
        load_method, resolved_checkpoint_path = _resolve_checkpoint_path(
            saved_run,
            best_or_last=best_or_last,
            epoch_number=epoch_number,
            paths=paths,
        )
    else:
        load_method = "state_dict"
        resolved_checkpoint_path = Path(checkpoint_path)
        if not resolved_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {resolved_checkpoint_path}")

    if load_method == "full_model":
        model = torch.load(resolved_checkpoint_path, map_location=resolved_device)
        model.eval()
        resolved_epoch = _epoch_from_checkpoint_path(resolved_checkpoint_path)
    else:
        if noise_model_path is None and noise_model_norm_prm_path is None:
            model, resolved_epoch = _load_state_dict_model(
                saved_run,
                resolved_checkpoint_path,
                resolved_device,
                paths,
            )
        else:
            model, resolved_epoch = _load_state_dict_model(
                saved_run,
                resolved_checkpoint_path,
                resolved_device,
                paths,
                noise_model_path=Path(noise_model_path) if noise_model_path is not None else None,
                noise_model_norm_prm_path=(
                    Path(noise_model_norm_prm_path)
                    if noise_model_norm_prm_path is not None
                    else None
                ),
            )

    effective_tiling_size = resolve_tiling_size(saved_run, tiling_size)
    return InferenceRuntime(
        model=model,
        device=resolved_device,
        checkpoint_path=resolved_checkpoint_path,
        load_method=load_method,
        tiling_size=effective_tiling_size,
        resolved_epoch=resolved_epoch,
    )



__all__ = ["InferenceRuntime", "TilingSizePolicy", "initialize_runtime", "resolve_tiling_size"]
