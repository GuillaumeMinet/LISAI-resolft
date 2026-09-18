from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import imwrite

from lisai.config.io import load_yaml, save_yaml
from lisai.data.utils import get_saving_shape


OUTPUTS_MANIFEST_NAME = "outputs_manifest.yaml"


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


def save_outputs(tosave: dict, save_folder: Path, img_name: str, no_suffix: bool = False) -> dict[str, Path]:
    """
    Save inference outputs as TIFF files.
    """
    saved: dict[str, Path] = {}
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
        saved[key] = path
    return saved


def save_outputs_manifest(save_folder: Path, items: list[dict[str, Any]]) -> Path:
    """Atomically write provenance for the outputs actually persisted by an evaluation."""
    save_folder = Path(save_folder)
    manifest_path = save_folder / OUTPUTS_MANIFEST_NAME
    tmp_path = save_folder / f".{OUTPUTS_MANIFEST_NAME}.tmp"
    manifest_items = [_serialize_manifest_item(item) for item in items]
    save_yaml(
        {
            "version": 1,
            "items": manifest_items,
        },
        tmp_path,
        default_flow_style=None,
    )
    tmp_path.replace(manifest_path)
    return manifest_path


def clear_outputs_manifest(save_folder: Path) -> None:
    """Remove stale finalized/temporary manifests before a new evaluation starts."""
    save_folder = Path(save_folder)
    for path in (
        save_folder / OUTPUTS_MANIFEST_NAME,
        save_folder / f".{OUTPUTS_MANIFEST_NAME}.tmp",
    ):
        path.unlink(missing_ok=True)


def load_outputs_manifest(path: Path) -> dict[str, Any]:
    """Load an outputs manifest, expanding compact source-index ranges."""
    path = Path(path)
    manifest_path = path / OUTPUTS_MANIFEST_NAME if path.is_dir() else path
    manifest = load_yaml(manifest_path)
    if manifest.get("version") != 1:
        raise ValueError(
            f"Unsupported outputs manifest version in {manifest_path}: {manifest.get('version')!r}."
        )
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Outputs manifest {manifest_path} must contain an `items` list.")
    manifest["items"] = [_deserialize_manifest_item(item, manifest_path) for item in items]
    return manifest


def _serialize_manifest_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return a manifest item with long contiguous source indices compacted on disk."""
    serialized = dict(item)
    if "source_indices" in serialized:
        serialized["source_indices"] = _serialize_source_indices(serialized["source_indices"])
    return serialized


def _serialize_source_indices(indices: Any) -> Any:
    """Compact long contiguous integer sequences while preserving irregular selections."""
    if not isinstance(indices, (list, tuple)):
        return indices

    values = list(indices)
    if len(values) < 5 or not all(isinstance(value, int) for value in values):
        return values

    if all(current == previous + 1 for previous, current in zip(values, values[1:])):
        return {"start": values[0], "end": values[-1]}

    return values


def _deserialize_manifest_item(item: Any, manifest_path: Path) -> dict[str, Any]:
    """Normalize one on-disk manifest item to the public in-memory representation."""
    if not isinstance(item, dict):
        raise ValueError(f"Outputs manifest {manifest_path} contains a non-mapping item.")

    normalized = dict(item)
    if "source_indices" in normalized:
        normalized["source_indices"] = _deserialize_source_indices(
            normalized["source_indices"],
            manifest_path=manifest_path,
        )
    return normalized


def _deserialize_source_indices(indices: Any, *, manifest_path: Path) -> Any:
    """Expand the compact ``{start, end}`` form used only for YAML readability."""
    if not isinstance(indices, dict):
        return indices

    if set(indices) != {"start", "end"}:
        raise ValueError(
            f"Invalid compact source_indices in {manifest_path}: expected only `start` and `end`."
        )

    start = indices["start"]
    end = indices["end"]
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError(
            f"Invalid compact source_indices in {manifest_path}: start/end must be integers with end >= start."
        )

    return list(range(start, end + 1))


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

    def flush(self) -> dict[str, Any] | None:
        """Persist buffered outputs and return manifest metadata for what was written."""
        if not self._entries:
            return None

        if self.item.data_format == "timelapse":
            entries = _sorted_timelapse_entries(self.item, self._entries)
            saved = save_outputs(
                _stack_timelapse_entries(self.item, entries),
                self.save_folder,
                img_name=self.item.name,
            )
            return self._manifest_record(
                entries=entries,
                saved_by_key={key: [path] for key, path in saved.items()},
                sample_axes={key: 1 if key == "samples" else 0 for key in saved},
            )

        saved_by_key: dict[str, list[Path]] = {}
        for sample_index, tosave in self._entries:
            saved = save_outputs(tosave, self.save_folder, img_name=self.item.sample_name(sample_index))
            for key, path in saved.items():
                saved_by_key.setdefault(key, []).append(path)

        return self._manifest_record(
            entries=self._entries,
            saved_by_key=saved_by_key,
            sample_axes={key: None for key in saved_by_key},
        )

    def _manifest_record(
        self,
        *,
        entries: list[tuple[int, dict]],
        saved_by_key: dict[str, list[Path]],
        sample_axes: dict[str, int | None],
    ) -> dict[str, Any]:
        """Build one item record whose ordering matches the persisted outputs."""
        sample_indices = [sample_index for sample_index, _ in entries]
        record: dict[str, Any] = {
            "name": self.item.name,
            "input_id": self.item.input_id or self.item.inp_path.as_posix(),
            "gt_id": self.item.gt_id,
        }

        if self.item.source_axis is not None:
            record["source_axis"] = self.item.source_axis
            record["source_indices"] = self.item.persisted_source_indices(sample_indices)

        outputs = {}
        for key, paths in saved_by_key.items():
            outputs[key] = {
                "files": [path.relative_to(self.save_folder).as_posix() for path in paths],
                "sample_axis": sample_axes[key],
            }
        record["outputs"] = outputs
        return record


def _stack_timelapse_entries(item: Any, entries: list[tuple[int, dict]]) -> dict:
    """Stack buffered sample outputs in the item's timelapse order."""

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


def _sorted_timelapse_entries(item: Any, entries: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """Return buffered timelapse samples in the same order used for persisted stacks."""
    return sorted(
        entries,
        key=lambda entry: item.sample_sort_key(entry[0]),
    )

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
