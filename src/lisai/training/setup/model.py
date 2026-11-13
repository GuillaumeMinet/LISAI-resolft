"""Training model setup helpers.

This module owns the model-side setup boundary of training. It turns the
resolved experiment config into a small `TrainingModelSpec`, performs any
LVAE-specific noise-model setup, and delegates the final build/load step to the
model loader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lisai.models.loader import prepare_model_for_training
from lisai.models.params import AnyModelParams

from .noise_model import load_noise_model_object, resolve_noise_model_name

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
    from lisai.infra.paths import Paths


@dataclass(frozen=True)
class TrainingModelSpec:
    """Training-local contract for model build/load decisions."""

    architecture: str
    parameters: AnyModelParams | None
    mode: str
    patch_size: int | None = None
    downsamp_factor: int = 1
    origin_run_dir: Path | None = None
    checkpoint_method: str | None = None
    checkpoint_selector: str | None = None
    checkpoint_epoch: int | None = None
    checkpoint_filename: str | None = None
    noise_model_name: str | None = None

    @classmethod
    def from_config(cls, cfg: ResolvedExperiment) -> TrainingModelSpec:
        """Project the resolved experiment config into the model setup contract."""
        ckpt = cfg.load_model.checkpoint if cfg.load_model else None
        origin_run_dir = cfg.experiment.origin_run_dir
        patch_size = cfg.data.model_patch_size
        downsamp_factor = cfg.data.downsampling_factor
        if cfg.model is None:
            raise ValueError("Resolved training config is missing the model section.")

        return cls(
            architecture=cfg.model.architecture,
            parameters=cfg.model.parameters,
            mode=cfg.experiment.mode,
            patch_size=int(patch_size) if patch_size is not None else None,
            downsamp_factor=int(downsamp_factor) if downsamp_factor is not None else 1,
            origin_run_dir=Path(origin_run_dir) if origin_run_dir else None,
            checkpoint_method=getattr(ckpt, "method", None) if ckpt else None,
            checkpoint_selector=getattr(ckpt, "selector", None) if ckpt else None,
            checkpoint_epoch=getattr(ckpt, "epoch", None) if ckpt else None,
            checkpoint_filename=getattr(ckpt, "filename", None) if ckpt else None,
            noise_model_name=resolve_noise_model_name(cfg),
        )



def _load_training_noise_model(spec: TrainingModelSpec, device, lisai_paths: Paths):
    """Load the LVAE noise model required by the training model when needed."""
    if spec.architecture != "lvae":
        return None
    return load_noise_model_object(spec.noise_model_name, device, lisai_paths)



def build_model(cfg: ResolvedExperiment, device, lisai_paths: Paths, model_norm_prm):
    """Build and optionally load the training model for the current run."""
    logger = logging.getLogger("lisai")
    spec = TrainingModelSpec.from_config(cfg)
    noise_model = _load_training_noise_model(spec, device, lisai_paths)

    model, state = prepare_model_for_training(
        spec=spec,
        paths=lisai_paths,
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
    )

    logger.info(f"Model initialized: {type(model).__name__}")
    return model, state
