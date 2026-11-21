"""Numerical core of HDN noise-model calibration (independent of datasets and workspaces).

The file-based builder and later in-memory adapters share these operations. Keep the
order and formulas of the legacy multi-noise-level normalization unchanged: saved
HDN normalization metadata and the trained likelihood depend on that convention.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lisai.data.utils import crop_center
from lisai.lib.hdn import histNoiseModel
from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel


@dataclass(frozen=True)
class PreparedNoiseModelData:
    signal: np.ndarray
    observation: np.ndarray
    avg_obs: float
    std_obs: float
    avg_sig: float
    std_sig: float
    min_signal: float
    max_signal: float
    norm_prm: dict[str, object]


@dataclass(frozen=True)
class NoiseModelFitParameters:
    histogram_bins: int = 256
    gmm_n_gaussian: int = 5
    gmm_n_coeff: int = 4
    gmm_n_epochs: int = 3000
    gmm_learning_rate: float = 0.05
    gmm_batch_size: int = 250_000
    gmm_min_sigma: float = 1.0
    gmm_name: str = "GMM"


@dataclass(frozen=True)
class FittedNoiseModel:
    histogram: np.ndarray
    gmm: GaussianMixtureNoiseModel


def _as_noise_levels(noise_level: int | Sequence[int]) -> tuple[list[int], bool]:
    if isinstance(noise_level, int):
        return [int(noise_level)], False

    levels = [int(v) for v in noise_level]
    if not levels:
        raise ValueError("noise_level sequence cannot be empty.")
    return levels, True


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std == 0:
        raise ValueError("Cannot normalize with zero standard deviation.")
    return (x - float(np.mean(x))) / std


def _prepare_signal_observation_arrays(
    signal_list: list[np.ndarray],
    observation_list: list[np.ndarray],
    *,
    noise_level: int | Sequence[int],
    norm_sig_to_obs: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    levels, mltpl_noise = _as_noise_levels(noise_level)
    per_noise_stats: dict[str, list[float]] = {}

    if not mltpl_noise:
        signal = np.stack(signal_list, axis=0)
        observation = np.stack(observation_list, axis=0)
        if norm_sig_to_obs:
            signal = _zscore(signal)
            signal = signal * float(np.std(observation)) + float(np.mean(observation))
        return signal, observation, per_noise_stats

    n_noise = len(levels)

    if not norm_sig_to_obs:
        signal = np.repeat(np.stack(signal_list, axis=0), n_noise, axis=0)
        observation = np.concatenate(observation_list, axis=0)
        return signal, observation, per_noise_stats

    n_imgs = len(signal_list)
    h, w = observation_list[0].shape[-2:]

    observation = np.reshape(
        np.concatenate(observation_list, axis=0),
        (n_noise, n_imgs, h, w),
        order="F",
    )
    avg_obs_per_noise = np.mean(observation, axis=(1, 2, 3))
    std_obs_per_noise = np.std(observation, axis=(1, 2, 3))

    signal = np.stack(signal_list, axis=0)
    signal = _zscore(signal)
    signal = np.tile(signal, (n_noise, 1, 1, 1))
    signal = (
        signal * std_obs_per_noise[:, np.newaxis, np.newaxis, np.newaxis]
        + avg_obs_per_noise[:, np.newaxis, np.newaxis, np.newaxis]
    )

    observation = np.reshape(observation, (n_noise * n_imgs, h, w))
    signal = np.reshape(signal, (n_noise * n_imgs, h, w))

    per_noise_stats["avgObs_per_noise"] = avg_obs_per_noise.tolist()
    per_noise_stats["stdObs_per_noise"] = std_obs_per_noise.tolist()
    return signal, observation, per_noise_stats


def clip_crop_pair(
    signal: np.ndarray,
    observation: np.ndarray,
    *,
    clip: float | None,
    crop_size: int | tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply legacy per-pair clipping and spatial center-cropping (on final two axes)."""
    if clip is not None:
        signal = np.maximum(signal, clip)
        observation = np.maximum(observation, clip)
    if crop_size is not None:
        signal = crop_center(signal, crop_size)
        observation = crop_center(observation, crop_size)
    return signal, observation


def prepare_noise_model_data(
    signal_list: list[np.ndarray],
    observation_list: list[np.ndarray],
    *,
    noise_level: int | Sequence[int],
    norm_sig_to_obs: bool,
    normalize_everything: bool,
    clip: float | None,
) -> PreparedNoiseModelData:
    """Prepare already clipped/cropped pairs and retain pre-global-normalization stats.

    ``clip`` is recorded in the HDN metadata, but clipping must have happened
    per pair via ``clip_crop_pair`` *before* calling this function. Keeping these
    stages separate preserves the existing file adapter's skip-on-crop semantics.
    """
    signal, observation, per_noise_stats = _prepare_signal_observation_arrays(
        signal_list,
        observation_list,
        noise_level=noise_level,
        norm_sig_to_obs=norm_sig_to_obs,
    )
    avg_obs = float(np.mean(observation))
    std_obs = float(np.std(observation))
    avg_sig = float(np.mean(signal))
    std_sig = float(np.std(signal))

    if normalize_everything:
        signal = _zscore(signal)
        observation = _zscore(observation)

    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    levels, multiple_noise = _as_noise_levels(noise_level)
    norm_prm: dict[str, object] = {
        "clip": clip,
        "normalize_data": normalize_everything,
        "normSig2Obs": norm_sig_to_obs,
        "avgObs": avg_obs,
        "stdObs": std_obs,
        "avgSig": avg_sig,
        "stdSig": std_sig,
    }
    if multiple_noise and norm_sig_to_obs:
        norm_prm["avgObs_per_noise"] = per_noise_stats["avgObs_per_noise"]
        norm_prm["stdObs_per_noise"] = per_noise_stats["stdObs_per_noise"]
    elif multiple_noise:
        norm_prm["avgObs_per_noise"] = [avg_obs] * len(levels)
        norm_prm["stdObs_per_noise"] = [std_obs] * len(levels)

    return PreparedNoiseModelData(
        signal=signal,
        observation=observation,
        avg_obs=avg_obs,
        std_obs=std_obs,
        avg_sig=avg_sig,
        std_sig=std_sig,
        min_signal=min_val,
        max_signal=max_val,
        norm_prm=norm_prm,
    )


def make_noise_histogram(data: PreparedNoiseModelData, *, bins: int) -> np.ndarray:
    """Generate the same conditional histogram used by the original builder."""
    return histNoiseModel.createHistogram(
        bins, data.min_signal, data.max_signal, data.observation, data.signal
    )


def fit_noise_model(
    data: PreparedNoiseModelData,
    *,
    params: NoiseModelFitParameters,
    save_dir: Path,
    device: torch.device,
    histogram: np.ndarray | None = None,
) -> FittedNoiseModel:
    """Fit the existing GMM without changing its training or random-number order.

    The legacy ``GaussianMixtureNoiseModel.train`` writes intermediate NPZ files
    into ``save_dir``. This is intentionally preserved in patch A; moving artifact
    lifecycle handling to the common persistence layer is a later patch.
    """
    if histogram is None:
        histogram = make_noise_histogram(data, bins=params.histogram_bins)

    gmm = GaussianMixtureNoiseModel(
        min_signal=data.min_signal,
        max_signal=data.max_signal,
        path=str(save_dir) + os.sep,
        weight=None,
        n_gaussian=params.gmm_n_gaussian,
        n_coeff=params.gmm_n_coeff,
        min_sigma=params.gmm_min_sigma,
        device=device,
    )
    gmm.train(
        data.signal,
        data.observation,
        batchSize=params.gmm_batch_size,
        n_epochs=params.gmm_n_epochs,
        learning_rate=params.gmm_learning_rate,
        name=params.gmm_name,
    )
    return FittedNoiseModel(histogram=histogram, gmm=gmm)
