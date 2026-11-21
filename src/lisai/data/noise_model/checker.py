from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lisai.infra.paths import Paths
from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel

from .builder import (
    _as_hw,
    _display_examples,
    _load_observation_image,
    _load_signal_image,
    _resolve_observation_dir,
)
from .core import clip_crop_pair, make_noise_histogram, prepare_noise_model_data
from .io import list_image_files


@dataclass(frozen=True)
class NoiseModelCheckConfig:
    dataset_name: str
    signal_subfolder: str
    observation_subfolder: str | None = None  # None/"same" -> same as signal

    noise_model_name: str | None = None
    noise_model_path: str | Path | None = None

    noise_level: int | Sequence[int] = 0
    signal_idx: int | None = None
    create_avg_signal: bool = False
    create_avg_signal_n_frames: int | str = 3  # int or "all"

    norm_sig_to_obs: bool = False
    normalize_everything: bool = True
    clip: float | None = 0.0
    crop_size: int | tuple[int, int] | None = None
    filters: tuple[str, ...] = ("tif", "tiff")

    histogram_bins: int = 256

    display: bool = True
    display_crop_size: int = 400
    gmm_plot_points: int = 15
    gmm_plot_max_columns: int = 4
    save_plot: bool = True
    plot_basename: str = "GMM_check_plot"


@dataclass(frozen=True)
class NoiseModelCheckResult:
    noise_model_path: Path
    plot_png_path: Path | None
    plot_svg_path: Path | None
    n_files_used: int
    signal_shape: tuple[int, ...]
    observation_shape: tuple[int, ...]
    min_signal: float
    max_signal: float
    avg_obs: float
    std_obs: float


def _resolve_noise_model_path(cfg: NoiseModelCheckConfig, paths: Paths) -> Path:
    if cfg.noise_model_path is not None:
        nm_path = Path(cfg.noise_model_path)
    elif cfg.noise_model_name:
        nm_path = paths.noise_model_path(noiseModel_name=cfg.noise_model_name)
    else:
        raise ValueError("Provide either noise_model_name or noise_model_path.")

    if not nm_path.exists():
        raise FileNotFoundError(f"Noise model not found: {nm_path}")
    return nm_path


def check_noise_model(
    cfg: NoiseModelCheckConfig,
    *,
    paths: Paths,
    device: torch.device | None = None,
) -> NoiseModelCheckResult:
    """
    Compare a trained GMM noise model against a histogram estimated from
    selected signal/observation data.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nm_path = _resolve_noise_model_path(cfg, paths)
    nm_params = np.load(nm_path)
    noise_model = GaussianMixtureNoiseModel(params=nm_params, device=device)

    print("Loaded noise model:")
    print(f"  path: {nm_path}")
    print(f"  #gaussians: {noise_model.n_gaussian}")
    print(f"  #coeffs: {noise_model.n_coeff}")
    print(f"  min_signal: {float(noise_model.min_signal.item())}")
    print(f"  max_signal: {float(noise_model.max_signal.item())}")

    dataset_dir = paths.dataset_dir(dataset_name=cfg.dataset_name)
    signal_dir = dataset_dir / cfg.signal_subfolder
    observation_dir, obs_is_same_as_signal = _resolve_observation_dir(
        dataset_dir, cfg.observation_subfolder, signal_dir
    )

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

        signal_img = _load_signal_image(signal_path, cfg)  # same fields as build config
        try:
            observation_img = _load_observation_image(obs_path, cfg.noise_level)
        except IndexError:
            print(f"Skipping {obs_path.name}: noise level index out of bounds.")
            continue

        signal_img, observation_img = clip_crop_pair(
            signal_img, observation_img, clip=cfg.clip, crop_size=cfg.crop_size
        )

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

    prepared = prepare_noise_model_data(
        signal_list,
        observation_list,
        noise_level=cfg.noise_level,
        norm_sig_to_obs=cfg.norm_sig_to_obs,
        normalize_everything=cfg.normalize_everything,
        clip=cfg.clip,
    )
    signal, observation = prepared.signal, prepared.observation
    avg_obs, std_obs = prepared.avg_obs, prepared.std_obs
    min_val, max_val = prepared.min_signal, prepared.max_signal

    if cfg.display:
        _display_examples(signal, observation, crop_size=cfg.display_crop_size)

    histogram = make_noise_histogram(prepared, bins=cfg.histogram_bins)

    if cfg.display:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 7))
        plt.title(f"Histogram-based check for noise level: {cfg.noise_level}")
        plt.imshow(histogram[0] ** 0.25, cmap="gray")
        plt.show()

    import matplotlib.pyplot as plt
    from lisai.lib.hdn.utils import plotProbabilityDistribution

    signal_bin_index_list = (
        np.linspace(0, cfg.gmm_plot_points - 1, cfg.gmm_plot_points)
        * cfg.histogram_bins
        // cfg.gmm_plot_points
    ).astype(int).tolist()

    fig = plotProbabilityDistribution(
        signalBinIndex=signal_bin_index_list,
        histogram=histogram[0],
        gaussianMixtureNoiseModel=noise_model,
        min_signal=min_val,
        max_signal=max_val,
        n_bin=cfg.histogram_bins,
        device=device,
        max_columns=cfg.gmm_plot_max_columns,
    )

    plot_png_path: Path | None = None
    plot_svg_path: Path | None = None
    if cfg.save_plot:
        plot_png_path = nm_path.parent / f"{cfg.plot_basename}.png"
        plot_svg_path = nm_path.parent / f"{cfg.plot_basename}.svg"
        fig.savefig(plot_png_path)
        fig.savefig(plot_svg_path)

    if cfg.display:
        plt.show()
    plt.close(fig)

    return NoiseModelCheckResult(
        noise_model_path=nm_path,
        plot_png_path=plot_png_path,
        plot_svg_path=plot_svg_path,
        n_files_used=len(signal_list),
        signal_shape=tuple(signal.shape),
        observation_shape=tuple(observation.shape),
        min_signal=min_val,
        max_signal=max_val,
        avg_obs=avg_obs,
        std_obs=std_obs,
    )
