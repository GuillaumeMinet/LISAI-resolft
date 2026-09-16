from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import imwrite

from lisai.data.utils import get_saving_shape


def resolve_prediction_inputs(
    data_path: Path,
    *,
    filters: list[str] | str,
    skip_if_contain: list[str] | None = None,
) -> tuple[Path, list[str], str | None]:
    """Resolve apply-mode input files from a file or directory path."""
    data_path = Path(data_path)
    if isinstance(filters, str):
        filters = [filters]
    filters = [f.lower() for f in filters]

    if data_path.is_dir():
        list_files = []
        for file_name in sorted(data_path.iterdir()):
            if not file_name.is_file():
                continue
            suffix = file_name.suffix.lower().replace(".", "")
            if suffix not in filters:
                continue
            if skip_if_contain is not None and any(skip in file_name.name for skip in skip_if_contain):
                continue
            list_files.append(file_name.name)

        if not list_files:
            raise FileNotFoundError(f"No file found in {data_path} with filters={filters}.")
        return data_path, list_files, None

    if data_path.is_file():
        return data_path, [""], data_path.name

    if data_path.suffix.lower() in {".tif", ".tiff"}:
        suffix_to_try = ".tif" if data_path.suffix.lower() == ".tiff" else ".tiff"
        path_to_try = data_path.with_suffix(suffix_to_try)
        if path_to_try.is_file():
            return path_to_try, [""], path_to_try.name

    raise FileNotFoundError(f"Input path not found: {data_path}")


def save_metrics_json(save_folder: Path, results: dict) -> None:
    """Write evaluation metrics to `metrics.json` inside the save folder."""
    save_folder = Path(save_folder)
    with open(save_folder / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)


def save_outputs(tosave: dict, save_folder: Path, img_name: str, no_suffix: bool = False) -> None:
    """
    Save inference outputs as TIFF files.
    """
    for key, item in tosave.items():
        if item is None:
            continue
        if len(tosave) == 1 and no_suffix:
            path = Path(save_folder) / f"{img_name}.tif"
        else:
            path = Path(save_folder) / f"{img_name}_{key}.tif"

        if key == "pred_colorCoded":
            imwrite(path, item, photometric="rgb")
        else:
            shape = get_saving_shape(item)
            imwrite(path, item, imagej=True, metadata={"axes": shape})


class EvalItemOutputWriter:
    """Collect sample-level outputs and save them at the evaluation-item level."""

    def __init__(self, *, item: Any, save_folder: Path):
        """Create a writer scoped to one evaluation item."""
        self.item = item
        self.save_folder = Path(save_folder)
        self._entries: list[tuple[int, dict]] = []

    def add(self, *, sample_index: int, tosave: dict) -> None:
        """Buffer outputs for one selected sample from the item."""
        self._entries.append((sample_index, tosave))

    def flush(self) -> None:
        """Persist buffered outputs, regrouping timelapses into one stack."""
        if not self._entries:
            return

        if self.item.data_format == "timelapse":
            save_outputs(
                _stack_timelapse_entries(self.item, self._entries),
                self.save_folder,
                img_name=self.item.name,
            )
            return

        for sample_index, tosave in self._entries:
            save_outputs(tosave, self.save_folder, img_name=self.item.sample_name(sample_index))


def _stack_timelapse_entries(item: Any, entries: list[tuple[int, dict]]) -> dict:
    """Stack buffered sample outputs in the item's timelapse order."""

    # Sorts frames back into timelapse order:
    entries = sorted(
        entries,
        key=lambda entry: item.sample_sort_key(entry[0]),
    )

    # Finds which output types exist (e.g. {"inp", "pred", "samples"})
    keys = set()
    for _, tosave in entries:
        for key, value in tosave.items():
            if value is not None:
                keys.add(key)

    # Stack each output across the timelapse.
    stacked = {}
    for key in keys:
        arrays = []

        for _, tosave in entries:
            value = tosave.get(key)
            if value is not None:
                arrays.append(value)

        if not arrays:
            continue

        if key == "samples":
            stacked[key] = np.stack(arrays, axis=1)
        else:
            arrays = _normalize_ndim(arrays)
            stacked[key] = np.stack(arrays, axis=0)

    return stacked

def _normalize_ndim(arrays):
    normalized = []

    for arr in arrays:
        arr = np.asarray(arr)

        if arr.ndim >= 4:
            if arr.shape[0] != 1:
                raise ValueError(
                    f"Expected singleton batch axis for timelapse output, got shape {arr.shape}."
                )
            arr = arr[0]

        normalized.append(arr)
    shapes = {arr.shape for arr in normalized}
    if len(shapes) != 1:
        raise ValueError(f"Cannot stack timelapse outputs with mismatched shapes: {sorted(shapes)}")

    return normalized
