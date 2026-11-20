"""Import independent recordings as a reproducible, unprepared training dataset.

The operation owns only the raw TIFF snapshot and the existing flat dataset
registry. Splitting and patch extraction remain with LISAI's training loader.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import tifffile

from lisai.data.dataset_registry import load_dataset_registry, save_dataset_registry
from lisai.infra.paths import Paths

if TYPE_CHECKING:
    from lisai.config.settings import Settings


RecordingInput = np.ndarray | str | Path | tuple[str, np.ndarray]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10))}


def _safe_filename(value: str) -> str:
    # Recording filenames can naturally contain spaces and Unicode; only reject
    # unsafe path components and names that fail on Windows.
    if (not isinstance(value, str) or not value or value != Path(value).name
            or "\\" in value or any(c in value for c in '<>:"|?*')
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"Unsafe recording filename: {value!r}")
    filename = Path(value)
    if (filename.suffix.lower() not in {".tif", ".tiff"}
            or filename.stem.upper().split(".")[0] in _WINDOWS_RESERVED
            or filename.stem.endswith((".", " "))):
        raise ValueError(f"Unsupported TIFF recording filename: {value!r}")
    # The existing prep_before=false file scanner uses lowercase glob filters.
    return filename.stem + filename.suffix.lower()


@dataclass(frozen=True)
class DatasetImportResult:
    dataset_name: str
    dataset_dir: Path
    input_dir: Path
    recording_paths: tuple[Path, ...]
    data_format: str
    recording_count: int
    total_frames: int


def _safe_name(value: str, *, kind: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {kind} {value!r}; use letters, digits, '_', '-' or '.'.")
    return value


def _recording(source: RecordingInput, index: int, expected_ndim: int):
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Recording must be an existing TIFF file: {path}")
        name = _safe_filename(path.name)
        array = tifffile.imread(path)
        source_path = path
    else:
        if isinstance(source, tuple):
            if len(source) != 2 or not isinstance(source[0], str):
                raise TypeError("Named input must be a (filename, ndarray) pair.")
            name, array = source
            if not name.lower().endswith((".tif", ".tiff")):
                name += ".tif"
            name = _safe_filename(name)
        elif isinstance(source, np.ndarray):
            array = source
            name = f"recording_{index:04d}.tif"
        else:
            raise TypeError(f"Unsupported recording type: {type(source).__name__}")
        if not isinstance(array, np.ndarray):
            raise TypeError("Recording data must be a NumPy ndarray.")
        source_path = None
    if array.ndim != expected_ndim:
        axes = "YX" if expected_ndim == 2 else "TYX"
        raise ValueError(f"{name}: expected {axes} ({expected_ndim} dimensions), got {array.shape}.")
    if 0 in array.shape or array.dtype.kind not in "uif":
        raise ValueError(f"{name}: expected a nonempty real numeric image, got {array.dtype}, {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: input contains NaN or infinite pixel values.")
    return name, array, source_path


def import_training_dataset(
    inputs: Sequence[RecordingInput],
    *,
    settings: Settings,
    dataset_name: str,
    data_format: str,
    description: str | None = None,
) -> DatasetImportResult:
    """Snapshot TIFFs/NumPy recordings under ``training/<name>/raw`` and register.

    Each input is *one indivisible recording*, including the whole time axis.
    Requires an initialized workspace. Neither this helper nor its cleanup ever
    overwrites an existing dataset, and failed imports do not register it.
    """
    name = _safe_name(dataset_name, kind="dataset name")
    if data_format not in {"single", "timelapse"}:
        raise ValueError("data_format must be 'single' (YX) or 'timelapse' (TYX).")
    if not inputs:
        raise ValueError("At least one recording is required.")
    paths = Paths(settings)
    dataset_dir = paths.dataset_dir(dataset_name=name, usage="training")
    registry_path = paths.dataset_registry_path()
    registry = load_dataset_registry(registry_path)
    if name in registry or dataset_dir.exists():
        raise FileExistsError(f"Dataset {name!r} is already registered or exists at {dataset_dir}.")

    records = [_recording(source, index, 2 if data_format == "single" else 3)
               for index, source in enumerate(inputs, start=1)]
    folded = [item[0].casefold() for item in records]
    if len(set(folded)) != len(folded):
        raise ValueError("Recording filenames collide; give each input a distinct filename.")
    total_frames = sum(1 if data_format == "single" else int(array.shape[0])
                       for _, array, _ in records)

    # Only the staging directory is created before all source images validate.
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{name}.import-", dir=dataset_dir.parent))
    committed = False
    try:
        raw = stage / "raw"
        raw.mkdir()
        for filename, array, source_path in records:
            destination = raw / filename
            if source_path is not None:
                shutil.copy2(source_path, destination)
            else:
                tifffile.imwrite(destination, array, photometric="minisblack")

        # Re-check immediately before publication, without overwriting data.
        registry = load_dataset_registry(registry_path)
        if name in registry or dataset_dir.exists():
            raise FileExistsError(f"Dataset {name!r} appeared during import; refusing overwrite.")
        metadata = {
            "data_format": data_format,
            "usage": "training",
            "for_training": True,
            "description": description or "Raw recordings imported for LISAI training",
            "defaults": {"raw": {"input": "raw", "target": None, "eval_gt": None}},
            "structure": {"raw": ["raw"]},
            "outputs": {"raw": [{"key": "raw", "path": "raw", "role": "inp",
                                  "axes": "YX" if data_format == "single" else "TYX"}]},
            "size": {"raw": {"n_files": len(records), "n_frames": total_frames}},
        }
        registry[name] = metadata
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".dataset-registry-", suffix=".yml", dir=registry_path.parent)
        os.close(fd)
        temp_registry = Path(temp_name)
        try:
            save_dataset_registry(registry, temp_registry)
            # Publish data before registry, so a failed registry write can roll it back.
            stage.rename(dataset_dir)
            committed = True
            os.replace(temp_registry, registry_path)
        finally:
            temp_registry.unlink(missing_ok=True)
    except BaseException:
        if committed:
            shutil.rmtree(dataset_dir)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return DatasetImportResult(
        dataset_name=name,
        dataset_dir=dataset_dir,
        input_dir=dataset_dir / "raw",
        recording_paths=tuple(dataset_dir / "raw" / filename for filename, _, _ in records),
        data_format=data_format,
        recording_count=len(records),
        total_frames=total_frames,
    )


__all__ = ["DatasetImportResult", "RecordingInput", "import_training_dataset"]
