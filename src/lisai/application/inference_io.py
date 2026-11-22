"""Single-file TIFF I/O for the future standalone inference viewer.

The helper saves an *existing* NumPy prediction; it never invokes a model or
changes the standard LISAI file-based prediction pipeline.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from lisai.config.models.inference import ApplyOverrides
from lisai.evaluation.inference.results import PredictionInfo
from lisai.model_sources import RunRef


@dataclass(frozen=True)
class TiffInputInfo:
    path: Path
    axes: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class InferenceProvenance:
    """Minimal provenance captured at inference time by the host application."""

    source_path: str | Path
    model: str | RunRef
    overrides: ApplyOverrides | None = None


def _check_image(data: np.ndarray, *, axes: str, label: str) -> None:
    if axes not in {"YX", "TYX"} or data.ndim != len(axes):
        raise ValueError(
            f"{label} must have explicit YX or TYX axes matching its dimensions; "
            f"received {axes!r} and shape {data.shape}."
        )
    if not all(s > 0 for s in data.shape):
        raise ValueError(f"{label} has an empty dimension: {data.shape}.")
    if data.dtype.kind not in "uif":
        raise ValueError(f"{label} requires a real numeric grayscale array, not {data.dtype}.")


def load_inference_tiff(path: str | Path) -> tuple[np.ndarray, TiffInputInfo]:
    """Read exactly one explicit YX image or TYX sequence; never guess T vs Z.

    Many ordinary multi-page TIFFs report ``QYX`` rather than an explicit time
    axis. Those are intentionally rejected pending a future confirm-axes UI.
    """
    from tifffile import TiffFile

    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Select a .tif or .tiff image.")
    if not source.is_file():
        raise FileNotFoundError(f"TIFF file not found: {source}")
    with TiffFile(source) as tiff:
        if len(tiff.series) != 1:
            raise ValueError(
                f"Expected a single TIFF image/series, found {len(tiff.series)}; "
                "select one unambiguous YX or TYX TIFF."
            )
        series = tiff.series[0]
        axes = str(series.axes).upper()
        if axes not in {"YX", "TYX"}:
            raise ValueError(
                f"Unsupported TIFF axes {axes!r}. Only YX or explicitly labelled "
                "TYX timelapses are supported; Z, channels and ambiguous stacks "
                "are not automatically treated as time."
            )
        data = np.asarray(series.asarray())
    _check_image(data, axes=axes, label="Input TIFF")
    return data, TiffInputInfo(source, axes, tuple(data.shape), str(data.dtype))


def _validated_metadata(
    data: np.ndarray, axes: str, info: PredictionInfo,
    provenance: InferenceProvenance,
) -> dict:
    if not isinstance(info, PredictionInfo):
        raise TypeError("info must be the PredictionInfo returned by LISAI predict().")
    if not isinstance(provenance, InferenceProvenance):
        raise TypeError("provenance must be an InferenceProvenance snapshot.")
    if not isinstance(provenance.model, (str, RunRef)) or not str(provenance.model).strip():
        raise ValueError("Prediction provenance must identify a model or training run.")
    if provenance.overrides is not None and not isinstance(provenance.overrides, ApplyOverrides):
        raise TypeError("provenance.overrides must be an ApplyOverrides instance or None.")

    length = data.shape[0] if axes == "TYX" else 1
    if len(info.source_frame_indices) != length:
        raise ValueError("Frame-alignment metadata does not match the prediction time axis.")
    if any(not isinstance(i, int) or i < 0 for i in info.source_frame_indices):
        raise ValueError("Source frame indices must be nonnegative integers.")
    if any(not isinstance(i, int) or i < 0 or i >= length
           for i in info.black_output_indices):
        raise ValueError("Black output indices must index actual output frames.")
    if len(set(info.black_output_indices)) != len(info.black_output_indices):
        raise ValueError("Black output indices cannot contain duplicates.")
    if axes == "YX" and (tuple(info.source_frame_indices) != (0,) or
                         info.black_output_indices):
        raise ValueError("Single-image predictions require source-frame alignment (0,).")
    try:
        lisai_version = version("lisai")
    except PackageNotFoundError:
        lisai_version = "unknown"
    model = provenance.model
    return {
        "lisai_version": lisai_version,
        "source_path": str(provenance.source_path),
        "model_source": "run" if isinstance(model, RunRef) else "installed",
        "model": model.run_id if isinstance(model, RunRef) else model,
        "overrides": (
            provenance.overrides.model_dump(mode="json", exclude_unset=True)
            if provenance.overrides is not None else {}
        ),
        "source_frame_indices": list(info.source_frame_indices),
        "black_output_indices": list(info.black_output_indices),
    }


def save_prediction_tiff(
    path: str | Path,
    result: np.ndarray,
    *,
    axes: str,
    info: PredictionInfo,
    provenance: InferenceProvenance,
    overwrite: bool = False,
) -> Path:
    """Atomically save a computed prediction and its metadata in one TIFF.

    Both axes and small LISAI provenance are embedded in the TIFF's metadata;
    there is no separately committed JSON sidecar to leave inconsistent.
    Without overwrite, installation uses an exclusive hard link to prevent a
    race from replacing an output file that appeared during writing.
    """
    from tifffile import imwrite

    target = Path(path).expanduser().absolute()
    if target.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Prediction output must be a .tif or .tiff file.")
    prediction = np.asarray(result)
    _check_image(prediction, axes=axes, label="Prediction")
    metadata = _validated_metadata(prediction, axes, info, provenance)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target}")
    # Model outputs are normally float32. Keep them in the same intensity units.
    prediction = prediction.astype(np.float32, copy=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.stem}.", suffix=".tif",
                                         dir=target.parent, delete=False) as handle:
            temp_name = handle.name
        imwrite(temp_name, prediction, metadata={"axes": axes, "lisai": metadata},
                photometric="minisblack")
        if overwrite:
            os.replace(temp_name, target)
        else:
            try:
                os.link(temp_name, target)
            except FileExistsError as exc:
                raise FileExistsError(f"Output already exists: {target}") from exc
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return target


__all__ = ["TiffInputInfo", "InferenceProvenance", "load_inference_tiff",
           "save_prediction_tiff"]
