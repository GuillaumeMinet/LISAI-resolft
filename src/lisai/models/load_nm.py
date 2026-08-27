from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from lisai.infra.paths import Paths

logger = logging.getLogger("lisai.noise_model")


def load_noise_model_from_paths(
    noise_model_path: str | Path,
    norm_prm_path: str | Path | None,
    device: torch.device,
):
    nm_path = Path(noise_model_path)
    if not nm_path.exists():
        raise FileNotFoundError(f"Noise model not found: {nm_path}")

    from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel

    nm_params = np.load(nm_path)
    noise_model = GaussianMixtureNoiseModel(params=nm_params, device=device)
    logger.info(f"Loaded noise GMM: {nm_path}")

    nm_norm_prm = None
    if norm_prm_path is not None:
        norm_path = Path(norm_prm_path)
        if norm_path.exists():
            with norm_path.open() as handle:
                nm_norm_prm = json.load(handle)
    return noise_model, nm_norm_prm


def load_noise_model(noise_model_name: str | None, device: torch.device, lisaiPaths: Paths):
    if not noise_model_name:
        return None, None

    nm_path = lisaiPaths.noise_model_path(noiseModel_name=noise_model_name)
    norm_path = lisaiPaths.noise_model_norm_prm_path(noiseModel_name=noise_model_name)
    return load_noise_model_from_paths(nm_path, norm_path, device)


__all__ = ["load_noise_model", "load_noise_model_from_paths"]
