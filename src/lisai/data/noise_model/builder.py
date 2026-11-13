from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tifffile import imread

from lisai.data.utils import crop_center
from lisai.infra.paths import Paths
from lisai.lib.hdn import histNoiseModel
from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from lisai.lib.hdn.utils import plotProbabilityDistribution

from .io import (
    ensure_output_dir,
    list_image_files,
    save_gmm_parameters,
    save_histogram,
    save_norm_prm,
    save_text,
)


@dataclass(frozen=True)
class NoiseModelBuildConfig:
    dataset_name: str
    signal_subfolder: str
    observation_subfolder: str | None = None  # None/"same" -> same as signal

    noise_level: int | Sequence[int] = 0
    signal_idx: int | None = None
    create_avg_signal: bool = False
    create_avg_signal_n_frames: int | str = 3  # int or "all"

    norm_sig_to_obs: bool = True
    normalize_everything: bool = True
    clip: float | None = -3.0
    crop_size: int | tuple[int, int] | None = None
    filters: tuple[str, ...] = ("tif", "tiff")

    histogram_bins: int = 256

    gmm_n_gaussian: int = 5
    gmm_n_coeff: int = 4
    gmm_n_epochs: int = 3000
    gmm_learning_rate: float = 0.05
    gmm_batch_size: int = 250_000
    gmm_min_sigma: float = 1.0
    gmm_name: str = "GMM"

    display: bool = False
    display_crop_size: int = 400
    gmm_plot_points: int = 15
    gmm_plot_max_columns: int = 4
    save_gmm_plot: bool = True

    overwrite: bool = False
    signal_info: str = "SnrAvg"
    save_name: str | None = None


@dataclass(frozen=True)
class NoiseModelBuildResult:
    save_dir: Path
    histogram_path: Path
    gmm_path: Path
    norm_prm_path: Path
    info_path: Path
    gmm_plot_png_path: Path | None
    gmm_plot_svg_path: Path | None
    n_files_used: int
    signal_shape: tuple[int, ...]
    observation_shape: tuple[int, ...]
    min_signal: float
    max_signal: float


def _as_noise_levels(noise_level: int | Sequence[int]) -> tuple[list[int], bool]:
    if isinstance(noise_level, int):
        return [int(noise_level)], False

    levels = [int(v) for v in noise_level]
    if not levels:
        raise ValueError("noise_level sequence cannot be empty.")
    return levels, True


def _as_hw(crop_size: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if crop_size is None:
        return None
    if isinstance(crop_size, int):
        return crop_size, crop_size
    return crop_size


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std == 0:
        raise ValueError("Cannot normalize with zero standard deviation.")
    return (x - float(np.mean(x))) / std


def _noise_level_label(noise_level: int | Sequence[int]) -> str:
    if isinstance(noise_level, int):
        return str(noise_level)
    return "".join(str([int(v) for v in noise_level]).split(", "))[1:-1]


def _build_default_save_name(cfg: NoiseModelBuildConfig) -> str:
    norm_str = "norm" if cfg.normalize_everything else "noNorm"
    return (
        f"{cfg.dataset_name}_Noise{_noise_level_label(cfg.noise_level)}_"
        f"Sig{cfg.signal_info}_Clip{cfg.clip}_{norm_str}"
    )


def _resolve_observation_dir(
    dataset_dir: Path, observation_subfolder: str | None, signal_dir: Path
) -> tuple[Path, bool]:
    if observation_subfolder is None or observation_subfolder == "same":
        return signal_dir, True
    return dataset_dir / observation_subfolder, False


def _resolve_avg_n_frames(spec: int | str, total_frames: int) -> int:
    if isinstance(spec, str):
        if spec.lower() == "all":
            return total_frames
        raise ValueError("create_avg_signal_n_frames must be int or 'all'.")
    if int(spec) <= 0:
        raise ValueError("create_avg_signal_n_frames must be > 0.")
    return min(int(spec), total_frames)


def _load_signal_image(path: Path, cfg: NoiseModelBuildConfig) -> np.ndarray:
    img = np.asarray(imread(path))

    if cfg.create_avg_signal:
        if img.ndim != 3:
            raise ValueError(
                f"create_avg_signal=True requires a stack (3D). Got shape {img.shape} for {path}."
            )
        n_frames = _resolve_avg_n_frames(cfg.create_avg_signal_n_frames, img.shape[0])
        return np.mean(img[:n_frames], axis=0)

    if cfg.signal_idx is not None:
        return np.asarray(img[cfg.signal_idx])

    if img.ndim > 2:
        raise ValueError(
            f"Signal image is not 2D for {path} (shape={img.shape}). "
            "Set signal_idx or create_avg_signal=True."
        )
    return img


def _load_observation_image(path: Path, noise_level: int | Sequence[int]) -> np.ndarray:
    img = np.asarray(imread(path))
    levels, mltpl_noise = _as_noise_levels(noise_level)

    if not mltpl_noise:
        level = levels[0]
        if img.ndim == 2:
            if level not in (0, -1):
                raise IndexError(f"Observation is 2D, cannot index noise level {level}: {path}")
            return img
        return np.asarray(img[level])

    if img.ndim < 3:
        raise ValueError(
            f"Multiple noise levels were requested, but observation is not a stack: {path}"
        )
    return np.asarray(img[levels])


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


def _display_examples(signal: np.ndarray, observation: np.ndarray, crop_size: int) -> None:
    import matplotlib.pyplot as plt

    if observation.shape[0] < 2:
        idxs = [0]
    else:
        idxs = np.random.randint(0, observation.shape[0], size=2).tolist()

    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(idxs):
        sig = crop_center(signal[idx], crop_size)
        obs = crop_center(observation[idx], crop_size)
        plt.subplot(1, 2 * len(idxs), 2 * i + 1)
        plt.imshow(obs, cmap="gray")
        plt.title(f"Observation #{idx}")
        plt.subplot(1, 2 * len(idxs), 2 * i + 2)
        plt.imshow(sig, cmap="gray")
        plt.title(f"Signal #{idx}")
    plt.show()


def _build_info_text(
    *,
    cfg: NoiseModelBuildConfig,
    signal_dir: Path,
    observation_dir: Path,
    avg_obs: float,
    std_obs: float,
    avg_sig: float,
    std_sig: float,
) -> str:
    dt_string = datetime.now().strftime("%d/%m/%Y - %Hh%Mm%Ss")
    header = f"{dt_string}\nNoise Model Info\n\n"

    if cfg.create_avg_signal:
        sig_info = f"Average first #{cfg.create_avg_signal_n_frames} frames"
    else:
        sig_info = f"Used frame idx {cfg.signal_idx}"

    params = (
        f"Dataset: {cfg.dataset_name}\n"
        f"\n"
        f"Signal data path: {signal_dir}\n"
        f"Observation data path: {observation_dir}\n"
        f"Noise Level: {cfg.noise_level}\n"
        f"Signal: {sig_info}\n"
        f"\n"
        f"Clip: {cfg.clip}\n"
        f"Normalized signal to observation: {cfg.norm_sig_to_obs}\n"
        f"Normalized everything: {cfg.normalize_everything}\n"
        f"Mean(obs): {avg_obs}\n"
        f"Std(obs): {std_obs}\n"
        f"Mean(sig): {avg_sig}\n"
        f"Std(sig): {std_sig}\n"
        f"\n"
        f"Histogram bins: {cfg.histogram_bins}\n"
        f"Crop Size: {cfg.crop_size}\n"
        f"GMM parameters: #{cfg.gmm_n_gaussian} gaussian, #{cfg.gmm_n_coeff} coeff\n"
        f"GMM training params: #{cfg.gmm_n_epochs}, learning rate {cfg.gmm_learning_rate}"
    )
    return header + params


def build_noise_model(
    cfg: NoiseModelBuildConfig,
    *,
    paths: Paths,
    device: torch.device | None = None,
) -> NoiseModelBuildResult:
    """
    Build and save histogram + GMM noise models from paired signal/observation data.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_dir = paths.dataset_dir(dataset_name=cfg.dataset_name)
    signal_dir = dataset_dir / cfg.signal_subfolder
    observation_dir, obs_is_same_as_signal = _resolve_observation_dir(
        dataset_dir, cfg.observation_subfolder, signal_dir
    )

    save_name = cfg.save_name or _build_default_save_name(cfg)
    base_save_dir = paths.noise_model_dir(noiseModel_name=save_name)
    save_dir = ensure_output_dir(base_save_dir, overwrite=cfg.overwrite)

    files_signal = list_image_files(signal_dir, cfg.filters)
    if not files_signal:
        raise ValueError(f"No valid files found in {signal_dir} for filters {cfg.filters}")

    if obs_is_same_as_signal:
        files_obs = files_signal
    else:
        files_obs = list_image_files(observation_dir, cfg.filters)
        if len(files_obs) != len(files_signal):
            raise ValueError(
                f"Mismatched file counts. signals={len(files_signal)}, observations={len(files_obs)}"
            )

    signal_list: list[np.ndarray] = []
    observation_list: list[np.ndarray] = []

    expected_hw = _as_hw(cfg.crop_size)

    for i, signal_path in enumerate(files_signal):
        obs_path = signal_path if obs_is_same_as_signal else files_obs[i]

        signal_img = _load_signal_image(signal_path, cfg)
        try:
            observation_img = _load_observation_image(obs_path, cfg.noise_level)
        except IndexError:
            print(f"Skipping {obs_path.name}: noise level index out of bounds.")
            continue

        if cfg.clip is not None:
            signal_img = np.maximum(signal_img, cfg.clip)
            observation_img = np.maximum(observation_img, cfg.clip)

        if cfg.crop_size is not None:
            signal_img = crop_center(signal_img, cfg.crop_size)
            observation_img = crop_center(observation_img, cfg.crop_size)

        if expected_hw is not None:
            if signal_img.shape[-2:] != expected_hw or observation_img.shape[-2:] != expected_hw:
                print(f"Skipping {signal_path.name}: crop did not match requested size {expected_hw}.")
                continue

        signal_list.append(signal_img)
        observation_list.append(observation_img)
        print(
            f"Signal {signal_path.name}: Observation {obs_path.name}: shape={observation_img.shape}"
        )

    if not signal_list:
        raise ValueError("No valid signal/observation pairs were loaded.")

    signal, observation, per_noise_stats = _prepare_signal_observation_arrays(
        signal_list,
        observation_list,
        noise_level=cfg.noise_level,
        norm_sig_to_obs=cfg.norm_sig_to_obs,
    )

    avg_obs = float(np.mean(observation))
    std_obs = float(np.std(observation))
    avg_sig = float(np.mean(signal))
    std_sig = float(np.std(signal))

    if cfg.normalize_everything:
        signal = _zscore(signal)
        observation = _zscore(observation)

    min_val = float(np.min(signal))
    max_val = float(np.max(signal))

    if cfg.display:
        _display_examples(signal, observation, crop_size=cfg.display_crop_size)

    histogram = histNoiseModel.createHistogram(
        cfg.histogram_bins, min_val, max_val, observation, signal
    )
    histogram_path = save_histogram(save_dir / "Histogram.npy", histogram)
    print(f"Histogram saved at: {histogram_path}")

    if cfg.display:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 7))
        plt.title(f"Histogram-based model for noise level: {cfg.noise_level}")
        plt.imshow(histogram[0] ** 0.25, cmap="gray")
        plt.show()

    gmm = GaussianMixtureNoiseModel(
        min_signal=min_val,
        max_signal=max_val,
        path=str(save_dir) + os.sep,
        weight=None,
        n_gaussian=cfg.gmm_n_gaussian,
        n_coeff=cfg.gmm_n_coeff,
        min_sigma=cfg.gmm_min_sigma,
        device=device,
    )
    gmm.train(
        signal,
        observation,
        batchSize=cfg.gmm_batch_size,
        n_epochs=cfg.gmm_n_epochs,
        learning_rate=cfg.gmm_learning_rate,
        name=cfg.gmm_name,
    )

    gmm_path = save_gmm_parameters(gmm, save_dir / f"{cfg.gmm_name}.npz")
    print(f"GMM saved at: {gmm_path}")

    levels, mltpl_noise = _as_noise_levels(cfg.noise_level)
    norm_prm: dict[str, object] = {
        "clip": cfg.clip,
        "normalize_data": cfg.normalize_everything,
        "normSig2Obs": cfg.norm_sig_to_obs,
        "avgObs": avg_obs,
        "stdObs": std_obs,
        "avgSig": avg_sig,
        "stdSig": std_sig,
    }
    if mltpl_noise and cfg.norm_sig_to_obs:
        norm_prm["avgObs_per_noise"] = per_noise_stats["avgObs_per_noise"]
        norm_prm["stdObs_per_noise"] = per_noise_stats["stdObs_per_noise"]
    elif mltpl_noise:
        norm_prm["avgObs_per_noise"] = [avg_obs] * len(levels)
        norm_prm["stdObs_per_noise"] = [std_obs] * len(levels)

    norm_prm_path = save_norm_prm(save_dir / "norm_prm.json", norm_prm)

    info_text = _build_info_text(
        cfg=cfg,
        signal_dir=signal_dir,
        observation_dir=observation_dir,
        avg_obs=avg_obs,
        std_obs=std_obs,
        avg_sig=avg_sig,
        std_sig=std_sig,
    )
    info_path = save_text(save_dir / "info.txt", info_text)

    gmm_plot_png_path: Path | None = None
    gmm_plot_svg_path: Path | None = None

    if cfg.save_gmm_plot or cfg.display:
        import matplotlib.pyplot as plt

        signal_bin_index_list = (
            np.linspace(0, cfg.gmm_plot_points - 1, cfg.gmm_plot_points) * cfg.histogram_bins // cfg.gmm_plot_points
        ).astype(int).tolist()

        gmm_params = np.load(gmm_path)
        gmm_plot_model = GaussianMixtureNoiseModel(params=gmm_params, device=device)

        fig = plotProbabilityDistribution(
            signalBinIndex=signal_bin_index_list,
            histogram=histogram[0],
            gaussianMixtureNoiseModel=gmm_plot_model,
            min_signal=min_val,
            max_signal=max_val,
            n_bin=cfg.histogram_bins,
            device=device,
            max_columns=cfg.gmm_plot_max_columns,
        )

        if cfg.save_gmm_plot:
            gmm_plot_png_path = save_dir / "GMM_plot.png"
            gmm_plot_svg_path = save_dir / "GMM_plot.svg"
            fig.savefig(gmm_plot_png_path)
            fig.savefig(gmm_plot_svg_path)

        if cfg.display:
            plt.show()
        plt.close(fig)

    return NoiseModelBuildResult(
        save_dir=save_dir,
        histogram_path=histogram_path,
        gmm_path=gmm_path,
        norm_prm_path=norm_prm_path,
        info_path=info_path,
        gmm_plot_png_path=gmm_plot_png_path,
        gmm_plot_svg_path=gmm_plot_svg_path,
        n_files_used=len(signal_list),
        signal_shape=tuple(signal.shape),
        observation_shape=tuple(observation.shape),
        min_signal=min_val,
        max_signal=max_val,
    )
