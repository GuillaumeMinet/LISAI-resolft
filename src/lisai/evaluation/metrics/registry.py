from __future__ import annotations

import numpy as np

from lisai.evaluation.metrics.ra_psnr import RangeInvariantPsnr as ra_psnr


def _ra_psnr_metric(ref: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    del data_range  # unused by range-invariant PSNR
    ref_b = np.expand_dims(ref, axis=0)
    pred_b = np.expand_dims(pred, axis=0)
    return float(ra_psnr(ref_b, pred_b))


def _psnr_metric(ref: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    from skimage.metrics import peak_signal_noise_ratio

    return float(peak_signal_noise_ratio(ref, pred, data_range=data_range))


def _ssim_metric(ref: np.ndarray, pred: np.ndarray, data_range: float) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(ref, pred, data_range=data_range))


METRIC_REGISTRY = {
    "psnr": _psnr_metric,
    "ra_psnr": _ra_psnr_metric,
    "ssim": _ssim_metric,
}

HIGHER_BETTER = frozenset({"psnr", "ra_psnr", "ssim"})
LOWER_BETTER = frozenset()
