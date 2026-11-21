"""Public in-memory HDN noise-model calibration.

This is the NumPy input boundary over LISAI's existing histogram/GMM pipeline.
The file/dataset-based builder remains available from ``lisai.data.noise_model``.
"""
from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lisai.data.noise_model.builder import _fit_and_save_prepared
from lisai.data.noise_model.core import clip_crop_pair, prepare_noise_model_data
from lisai.infra.paths import Paths

if TYPE_CHECKING:
    from lisai.config.settings import Settings

__all__ = ["NoiseModelOptions", "NoiseModelResult", "create_noise_model"]


@dataclass(frozen=True)
class NoiseModelOptions:
    """Scientific options for the existing HDN GMM fitter.

    Defaults match the file-based ``NoiseModelBuildConfig``. Array inputs are
    already paired; file selection, reference averaging and multiple-noise-level
    selection remain the responsibility of the existing file-based builder.
    """

    norm_sig_to_obs: bool = True
    normalize_everything: bool = True
    clip: float | None = -3.0
    crop_size: int | tuple[int, int] | None = None
    histogram_bins: int = 256
    gmm_n_gaussian: int = 5
    gmm_n_coeff: int = 4
    gmm_n_epochs: int = 3000
    gmm_learning_rate: float = 0.05
    gmm_batch_size: int = 250_000
    gmm_min_sigma: float = 1.0
    save_gmm_plot: bool = True
    gmm_plot_points: int = 15
    gmm_plot_max_columns: int = 4

    def __post_init__(self) -> None:
        for name in ("norm_sig_to_obs", "normalize_everything", "save_gmm_plot"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        minima = {
            "histogram_bins": 2, "gmm_n_gaussian": 1, "gmm_n_coeff": 2,
            # GMM.train computes an interval using int(n_epochs * 0.5).
            "gmm_n_epochs": 2, "gmm_batch_size": 1, "gmm_plot_points": 1,
            "gmm_plot_max_columns": 1,
        }
        for name, minimum in minima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("gmm_learning_rate", "gmm_min_sigma"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.clip is not None and (
            not isinstance(self.clip, (int, float)) or not math.isfinite(self.clip)
        ):
            raise ValueError("clip must be finite or None")
        crop_size = self.crop_size
        if crop_size is not None:
            sizes = (crop_size,) if isinstance(crop_size, int) else crop_size
            if not isinstance(sizes, tuple) or len(sizes) not in (1, 2) or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in sizes
            ):
                raise ValueError("crop_size must be a positive integer or (height, width)")


@dataclass(frozen=True)
class NoiseModelResult:
    """Locations and basic diagnostics of a saved HDN-compatible calibration."""

    name: str
    save_dir: Path
    histogram_path: Path
    gmm_path: Path
    norm_prm_path: Path
    info_path: Path
    gmm_plot_png_path: Path | None
    gmm_plot_svg_path: Path | None
    n_pairs: int
    signal_shape: tuple[int, ...]
    observation_shape: tuple[int, ...]
    min_signal: float
    max_signal: float


def _paired_arrays(
    signal: np.ndarray, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy YX or matched NYX inputs without guessing axis semantics."""
    for name, data in (("signal", signal), ("observation", observation)):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{name} must be a NumPy ndarray")
        if data.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain real numeric values, got {data.dtype}")
        if data.ndim not in (2, 3):
            raise ValueError(f"{name} must have shape (Y, X) or (N, Y, X), got {data.shape}")
        if any(n == 0 for n in data.shape):
            raise ValueError(f"{name} must not have empty dimensions")
        if not np.isfinite(data).all():
            raise ValueError(f"{name} contains non-finite values")
    if signal.shape != observation.shape:
        raise ValueError(f"Paired signal and observation shapes must match: {signal.shape} != {observation.shape}")
    return (
        signal[None].copy() if signal.ndim == 2 else signal.copy(),
        observation[None].copy() if observation.ndim == 2 else observation.copy(),
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid noise-model device: {device!r}") from exc
    if selected.type not in ("cuda", "cpu"):
        raise ValueError("Noise-model fitting supports 'cpu', 'cuda' or 'auto'")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for noise-model fitting but is not available")
    return selected


def create_noise_model(
    signal: np.ndarray,
    observation: np.ndarray,
    *,
    name: str,
    settings: Settings,
    options: NoiseModelOptions | None = None,
    device: str | torch.device = "auto",
    overwrite: bool = False,
) -> NoiseModelResult:
    """Fit and save an HDN noise model from paired in-memory images.

    ``signal`` is a high-SNR reference and ``observation`` the corresponding
    noisy acquisition. Shapes are ``YX`` or ``NYX``; each pair must be aligned
    and have the same intensity interpretation as the HDN training data.

    Saves ``GMM.npz``, ``norm_prm.json``, ``Histogram.npy``, ``info.txt`` and,
    by default, ``GMM_plot.png/svg`` in the selected workspace's noise-model
    directory. The returned ``name`` can be passed to quick-HDN training.

    Requires the optional ``lisai[full]`` dependencies for plotting when
    ``options.save_gmm_plot`` is true. No registered dataset or TIFF I/O is used.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name.endswith("."):
        raise ValueError("name must be one safe directory component (letters, digits, '_', '-' or '.')")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if options is None:
        options = NoiseModelOptions()
    elif not isinstance(options, NoiseModelOptions):
        raise TypeError("options must be a NoiseModelOptions instance")

    selected_device = _resolve_device(device)
    # Construct paths from the explicitly selected settings; never inspect CLI state.
    base_save_dir = Paths(settings).noise_model_dir(noiseModel_name=name)
    if base_save_dir.exists():
        if not base_save_dir.is_dir():
            raise NotADirectoryError(f"Noise-model location is not a directory: {base_save_dir}")
        if not overwrite:
            raise FileExistsError(f"Noise model '{name}' already exists: {base_save_dir}")

    signal_stack, observation_stack = _paired_arrays(signal, observation)
    signal_stack, observation_stack = clip_crop_pair(
        signal_stack, observation_stack, clip=options.clip, crop_size=options.crop_size
    )
    prepared = prepare_noise_model_data(
        list(signal_stack), list(observation_stack),
        noise_level=0,
        norm_sig_to_obs=options.norm_sig_to_obs,
        normalize_everything=options.normalize_everything,
        clip=options.clip,
    )
    if prepared.max_signal <= prepared.min_signal or prepared.std_obs == 0:
        raise ValueError("Calibration requires non-constant signal and observation data")

    n_pairs = len(signal_stack)
    dt_string = datetime.now().strftime("%d/%m/%Y - %Hh%Mm%Ss")
    info = (
        f"{dt_string}\nNoise Model Info\n\n"
        f"Source: in-memory paired NumPy arrays\n"
        f"Number of pairs: {n_pairs}\n"
        f"Input signal shape: {tuple(signal.shape)}\n"
        f"Input observation shape: {tuple(observation.shape)}\n"
        f"Clip: {options.clip}\n"
        f"Normalized signal to observation: {options.norm_sig_to_obs}\n"
        f"Normalized everything: {options.normalize_everything}\n"
        f"Mean(obs): {prepared.avg_obs}\nStd(obs): {prepared.std_obs}\n"
        f"Mean(sig): {prepared.avg_sig}\nStd(sig): {prepared.std_sig}\n"
        f"Histogram bins: {options.histogram_bins}\n"
        f"Crop Size: {options.crop_size}\n"
        f"GMM parameters: #{options.gmm_n_gaussian} gaussian, #{options.gmm_n_coeff} coeff\n"
        f"GMM training params: #{options.gmm_n_epochs}, learning rate {options.gmm_learning_rate}"
    )

    # Patch C will make the full overwrite/persistence lifecycle atomic.
    created = not base_save_dir.exists()
    if created:
        base_save_dir.mkdir(parents=True, exist_ok=False)
    try:
        built = _fit_and_save_prepared(
            prepared,
            save_dir=base_save_dir,
            device=selected_device,
            n_files_used=n_pairs,
            info_text=info,
            histogram_bins=options.histogram_bins,
            gmm_n_gaussian=options.gmm_n_gaussian,
            gmm_n_coeff=options.gmm_n_coeff,
            gmm_n_epochs=options.gmm_n_epochs,
            gmm_learning_rate=options.gmm_learning_rate,
            gmm_batch_size=options.gmm_batch_size,
            gmm_min_sigma=options.gmm_min_sigma,
            gmm_name="GMM",
            save_gmm_plot=options.save_gmm_plot,
            gmm_plot_points=options.gmm_plot_points,
            gmm_plot_max_columns=options.gmm_plot_max_columns,
        )
    except BaseException:
        # Only remove a directory that this invocation created, never an
        # existing calibration explicitly selected for overwrite.
        if created:
            shutil.rmtree(base_save_dir)
        raise
    return NoiseModelResult(
        name=name,
        save_dir=built.save_dir,
        histogram_path=built.histogram_path,
        gmm_path=built.gmm_path,
        norm_prm_path=built.norm_prm_path,
        info_path=built.info_path,
        gmm_plot_png_path=built.gmm_plot_png_path,
        gmm_plot_svg_path=built.gmm_plot_svg_path,
        n_pairs=n_pairs,
        signal_shape=built.signal_shape,
        observation_shape=built.observation_shape,
        min_signal=built.min_signal,
        max_signal=built.max_signal,
    )
