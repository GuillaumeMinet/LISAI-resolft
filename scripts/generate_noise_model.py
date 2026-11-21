"""Example: calibrate an HDN noise model from paired TIFF images.

Save as ``scripts/generate_noise_model.py`` and run from the LISAI environment::

    python scripts/generate_noise_model.py

WHAT IS BEING CALIBRATED?
------------------------
HDN needs a noise model describing P(noisy observation | clean signal).
LISAI estimates this distribution from matched high-SNR reference images
(``signal``) and noisy images (``observation``), then fits a Gaussian mixture
model (GMM). The result can later be selected for unsupervised HDN training.

WHAT DATA SHOULD I PROVIDE?
---------------------------
* Each signal image and observation image must show the SAME stationary sample
  region, aligned pixel by pixel. A reference can be created by averaging
  repeated high-SNR acquisitions *before* running this script.
* Use the same processing/intensity domain as HDN training. For example, if
  HDN will train on reconstructed MoNaLISA images, calibrate on reconstructions,
  not raw camera frames.
* Either provide two TIFF files (YX images or NYX stacks) or two folders of
  TIFF files. Folder contents are paired by the SAME relative filename; a TIFF
  stack provides one pair per corresponding frame. Different shapes or filenames
  are rejected rather than paired by guesswork.
* A series of changing live-cell structures is NOT a set of aligned clean/noisy
  pairs. Do not use an unrelated time-lapse as the noisy calibration reference.

Edit only the PARAMETERS section for an ordinary calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from lisai.config.settings import Settings
from lisai.noise_models import NoiseModelOptions, create_noise_model


# =============================================================================
# PARAMETERS — edit this section
# =============================================================================

# LISAI workspace that will own the saved model (must already be initialized).
WORKSPACE = Path(r"CHANGE_ME")

# Two matching TIFF files, or two directories with matching relative filenames.
# SIGNAL: high-SNR (ideally nearly noise-free) reference of the stationary sample.
# OBSERVATION: noisy image acquired at the conditions HDN will encounter.
SIGNAL_SOURCE = Path(r"CHANGE_ME")
OBSERVATION_SOURCE = Path(r"CHANGE_ME")

# Model identifier used for saving and, later, for quick-HDN training.
MODEL_NAME = "CHANGE_ME"

# Use the GPU if available, otherwise CPU. Also accepts "cpu" or "cuda".
DEVICE = "auto"

# Avoid accidentally overwriting an existing calibration.
OVERWRITE = False

# --- Data preparation / normalization ---

# Reference and observation can have different overall intensity statistics.
# True: rescale the reference's mean and std to match the noisy observations.
# This follows LISAI's existing noise-calibration convention.
NORM_SIGNAL_TO_OBSERVATION = True

# True: standardize signal and observation for histogram/GMM fitting.
# LISAI saves the normalization parameters so HDN can interpret the model.
NORMALIZE_EVERYTHING = True

# Lower clipping floor applied BEFORE normalization; values below it are set
# to this level. Use None to disable. Default follows the original LISAI fitter.
CLIP = -3.0

# Optional center crop of EACH pair (in pixels), useful for restricting the
# calibration region. None keeps whole images; e.g. (512, 512) crops to 512x512.
CROP_SIZE = None

# --- Noise histogram and GMM fitting ---

# Number of bins per axis in the empirical conditional noise histogram.
# 256 is the established default. More bins need more representative data.
HISTOGRAM_BINS = 256

# Number of Gaussian mixture components used to describe the noise at each
# signal value. Default: 5. More components allow more complex distributions,
# but may overfit if calibration data are sparse.
GMM_N_GAUSSIAN = 5

# Number of optimization epochs. Default: 3000 for full fitting; a smaller
# number (e.g. 100) is useful only for a quick functional test.
GMM_N_EPOCHS = 3000

# Optimizer learning rate (existing LISAI default: 0.05).
GMM_LEARNING_RATE = 0.05

# Save the diagnostic GMM-versus-histogram plots in the model directory.
# Requires the optional plotting dependencies from lisai[full].
SAVE_GMM_PLOTS = True

# Less commonly changed defaults are intentionally left to NoiseModelOptions:
# gmm_n_coeff=4 (signal-dependent polynomial parameterization),
# gmm_batch_size=250_000, gmm_min_sigma=1.0, gmm_plot_points=15.

# =============================================================================
# INPUT LOADING — generally no changes needed below
# =============================================================================


def _tiff_files(source: Path) -> dict[str, Path]:
    """Return one file or all TIFFs in a directory, indexed by relative name."""
    if source.is_file():
        if source.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Expected TIFF file: {source}")
        return {"__single__": source}
    if not source.is_dir():
        raise FileNotFoundError(f"Input not found: {source}")

    files = {
        p.relative_to(source).as_posix(): p
        for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    }
    if not files:
        raise ValueError(f"No TIFF files found in {source}")
    return files


def load_paired_tiffs(signal_source: Path, observation_source: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read matched YX images / NYX stacks into two arrays shaped (N, Y, X)."""
    if signal_source.is_file() != observation_source.is_file():
        raise ValueError("Provide two TIFF files or two TIFF directories, not one of each")

    signals = _tiff_files(signal_source)
    observations = _tiff_files(observation_source)
    if signals.keys() != observations.keys():
        missing_observations = sorted(signals.keys() - observations.keys())
        missing_signals = sorted(observations.keys() - signals.keys())
        raise ValueError(
            "Signal and observation filenames must match. "
            f"Missing observations: {missing_observations[:5]}; "
            f"missing signals: {missing_signals[:5]}"
        )

    signal_parts, observation_parts = [], []
    expected_hw = None
    for name in sorted(signals):
        signal = np.asarray(tifffile.imread(signals[name]), dtype=np.float32)
        observation = np.asarray(tifffile.imread(observations[name]), dtype=np.float32)
        if signal.shape != observation.shape or signal.ndim not in (2, 3):
            raise ValueError(
                f"Invalid pair {name!r}: signal {signal.shape}, "
                f"observation {observation.shape}. Expected matching YX or NYX data."
            )
        if expected_hw is None:
            expected_hw = signal.shape[-2:]
        if signal.shape[-2:] != expected_hw:
            raise ValueError(f"Spatial image size differs in {name!r}: {signal.shape[-2:]}")

        signal_parts.append(signal[None, ...] if signal.ndim == 2 else signal)
        observation_parts.append(observation[None, ...] if observation.ndim == 2 else observation)

    return np.concatenate(signal_parts), np.concatenate(observation_parts)


# =============================================================================
# CALIBRATION
# =============================================================================


def main() -> None:
    signal, observation = load_paired_tiffs(SIGNAL_SOURCE, OBSERVATION_SOURCE)
    print(f"Loaded {signal.shape[0]} paired images; each is {signal.shape[-2:]} pixels")
    print(f"Saving calibration '{MODEL_NAME}' in workspace: {WORKSPACE}")

    # Passing Settings explicitly avoids relying on whichever workspace happens
    # to be selected in a separate CLI or ImProcess session.
    settings = Settings.from_workspace(WORKSPACE)

    options = NoiseModelOptions(
        norm_sig_to_obs=NORM_SIGNAL_TO_OBSERVATION,
        normalize_everything=NORMALIZE_EVERYTHING,
        clip=CLIP,
        crop_size=CROP_SIZE,
        histogram_bins=HISTOGRAM_BINS,
        gmm_n_gaussian=GMM_N_GAUSSIAN,
        gmm_n_epochs=GMM_N_EPOCHS,
        gmm_learning_rate=GMM_LEARNING_RATE,
        save_gmm_plot=SAVE_GMM_PLOTS,
    )

    result = create_noise_model(
        signal=signal,
        observation=observation,
        name=MODEL_NAME,
        settings=settings,
        options=options,
        device=DEVICE,
        overwrite=OVERWRITE,
    )

    print("\nNoise model generated successfully")
    print(f"  Directory:     {result.save_dir}")
    print(f"  GMM:           {result.gmm_path}")
    print(f"  Normalization: {result.norm_prm_path}")
    print(f"  Histogram:     {result.histogram_path}")
    if result.gmm_plot_png_path is not None:
        print(f"  Plot:          {result.gmm_plot_png_path}")
    print(f"\nFor quick-HDN training, use noise_model_name={result.name!r}.")


if __name__ == "__main__":
    main()
