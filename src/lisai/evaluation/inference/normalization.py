"""Input and output normalization for saved LISAI models."""

import numpy as np

from lisai.config.models.inference import InputNormalization


def normalize_inp(inp, clip, data_norm, model_norm):
    """
    Normalize input image before prediction.
    """
    if not isinstance(clip, bool):
        inp[inp < clip] = clip

    data_norm = data_norm or {}
    if data_norm.get("normalize_data"):
        inp = (inp - data_norm.get("avgObs")) / data_norm.get("stdObs")
    if model_norm is not None:
        inp = (inp - model_norm.get("data_mean")) / model_norm.get("data_std")
    return inp


def _training_input_reference(data_norm, model_norm) -> tuple[float, float]:
    """Express the saved input z-score reference in raw-image intensity units.

    HDN may normalize first using noise-model observation statistics, then with
    model statistics; ordinary U-Nets generally use only the latter stage.
    """
    data_norm = data_norm or {}
    if model_norm is None and not data_norm.get("normalize_data"):
        raise ValueError("Image normalization requires saved training input statistics.")

    mean, std = 0.0, 1.0
    if model_norm is not None:
        mean = float(model_norm["data_mean"])
        std = float(model_norm["data_std"])
    if data_norm.get("normalize_data"):
        mean = float(data_norm["avgObs"]) + float(data_norm["stdObs"]) * mean
        std *= float(data_norm["stdObs"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Saved training input normalization has invalid mean/std.")
    return mean, std


def normalize_apply_input(inp, clip, data_norm, model_norm, config: InputNormalization, *, reference_image=None):
    """Adapt raw intensities, then apply the model's *unchanged* saved normalization.

    Image statistics are computed once on each selected image/stack (including
    all selected frames), not independently per tile or frame. Manual and
    image adaptations happen before the saved raw-intensity clipping threshold.
    """
    if config.mode == "training":
        return normalize_inp(np.array(inp, dtype=np.float32, copy=True), clip, data_norm, model_norm)

    # A float copy prevents clipping from modifying the original input and avoids
    # integer overflow/truncation when applying a gain or offset.
    raw = np.array(inp, dtype=np.float32, copy=True)
    if config.mode == "manual":
        adapted = raw * config.gain + config.offset
    elif config.mode == "image":
        if not np.isfinite(raw).all():
            raise ValueError("Image normalization requires finite input pixels.")
        reference = raw if reference_image is None else np.asarray(reference_image, dtype=np.float32)
        if not np.isfinite(reference).all():
            raise ValueError("Image normalization requires finite input pixels.")
        image_mean = float(np.mean(reference, dtype=np.float64))
        image_std = float(np.std(reference, dtype=np.float64))
        if not np.isfinite(image_std) or image_std <= 0:
            raise ValueError("Image normalization requires nonzero input variance.")
        training_mean, training_std = _training_input_reference(data_norm, model_norm)
        adapted = (raw - image_mean) * (training_std / image_std) + training_mean
    else:
        raise ValueError(f"Unknown input normalization mode: {config.mode!r}")
    if not np.isfinite(adapted).all():
        raise ValueError("Input normalization produced non-finite pixel values.")
    return normalize_inp(adapted, clip, data_norm, model_norm)


def denormalize_pred(pred, data_norm, model_norm):
    """
    Denormalize model output.
    """
    data_norm = data_norm or {}
    if model_norm is not None:
        pred = pred * model_norm.get("data_std") + model_norm.get("data_mean")

    if data_norm.get("normalize_data"):
        pred = pred * data_norm.get("avgObs") + data_norm.get("stdObs")
    return pred

