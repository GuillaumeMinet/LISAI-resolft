from __future__ import annotations

from collections.abc import Callable

import numpy as np


PredictionFormat = Callable[[np.ndarray], np.ndarray]


def _image(array: np.ndarray) -> np.ndarray:
    return np.asarray(array)


def _stack_inp_pred(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim < 3 or array.shape[0] != 2:
        raise ValueError(
            "prediction_format='stack_inp_pred' expects a two-entry stack on axis 0 "
            f"(input, prediction); got shape {array.shape}."
        )
    return np.asarray(array[-1])


def _singleton_4d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 1:
        raise ValueError(
            "prediction_format='4d' expects shape (1, 1, H, W); "
            f"got shape {array.shape}."
        )
    return np.asarray(array[0, 0])


_PREDICTION_FORMATS: dict[str, PredictionFormat] = {
    "image": _image,
    "stack_inp_pred": _stack_inp_pred,
    "4d": _singleton_4d,
    "4d_singleton": _singleton_4d,
}


def available_prediction_formats() -> tuple[str, ...]:
    return tuple(sorted(_PREDICTION_FORMATS))


def convert_prediction(array: np.ndarray, prediction_format: str) -> np.ndarray:
    try:
        converter = _PREDICTION_FORMATS[prediction_format]
    except KeyError as exc:
        known = ", ".join(available_prediction_formats())
        raise ValueError(
            f"Unknown prediction_format {prediction_format!r}. Available formats: {known}."
        ) from exc
    return converter(np.asarray(array))


__all__ = ["available_prediction_formats", "convert_prediction"]
