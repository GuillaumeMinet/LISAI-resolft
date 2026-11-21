"""Optional noise-model diagnostics, loaded only when plotting is requested."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Noise-model plotting requires Matplotlib; install 'lisai[full]' "
            "or disable save_gmm_plot/display."
        ) from exc
    return plt


def display_examples(signal: np.ndarray, observation: np.ndarray, crop_size: int) -> None:
    from lisai.data.utils import crop_center

    plt = _pyplot()
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


def display_histogram(histogram: np.ndarray, noise_level_label: str) -> None:
    plt = _pyplot()
    plt.figure(figsize=(7, 7))
    plt.title(f"Histogram-based model for noise level: {noise_level_label}")
    plt.imshow(histogram[0] ** 0.25, cmap="gray")
    plt.show()


def render_gmm_diagnostics(
    *,
    gmm_path: Path,
    histogram: np.ndarray,
    min_signal: float,
    max_signal: float,
    histogram_bins: int,
    device: torch.device,
    plot_points: int,
    plot_max_columns: int,
    save: bool,
    display: bool,
) -> tuple[Path | None, Path | None]:
    """Render existing probability plots; close figures even if saving fails."""
    plt = _pyplot()
    from lisai.lib.hdn.utils import plotProbabilityDistribution

    signal_bin_index_list = (
        np.linspace(0, plot_points - 1, plot_points) * histogram_bins // plot_points
    ).astype(int).tolist()
    with np.load(gmm_path) as data:
        params = {key: data[key] for key in data.files}
    gmm_model = GaussianMixtureNoiseModel(params=params, device=device)
    fig = plotProbabilityDistribution(
        signalBinIndex=signal_bin_index_list,
        histogram=histogram[0],
        gaussianMixtureNoiseModel=gmm_model,
        min_signal=min_signal,
        max_signal=max_signal,
        n_bin=histogram_bins,
        device=device,
        max_columns=plot_max_columns,
    )
    png_path: Path | None = None
    svg_path: Path | None = None
    try:
        if save:
            png_path = gmm_path.parent / "GMM_plot.png"
            svg_path = gmm_path.parent / "GMM_plot.svg"
            fig.savefig(png_path)
            fig.savefig(svg_path)
        if display:
            plt.show()
    finally:
        plt.close(fig)
    return png_path, svg_path
