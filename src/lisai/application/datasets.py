"""Training dataset discovery, explicit selection and quick-import inputs.

This layer never creates a dataset while browsing and contains no Qt or host
image types. Scientific data loading and import remain in existing LISAI APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TrainingRecording:
    """One independent host-provided recording, not an individual time frame."""

    name: str
    data: np.ndarray
    axes: tuple[str, ...]
    selected: bool = False


@dataclass(frozen=True)
class TrainingDatasetEntry:
    name: str
    data_format: str | None
    usage: str
    for_training: bool
    data_types: tuple[str, ...]
    available_roles: tuple[str, ...]
    description: str | None
    quick_hdn_eligible: bool
    quick_recording_count: int


@dataclass(frozen=True)
class DatasetChangePreview:
    """Fields updated *only* after an explicit user confirmation."""

    dataset_name: str
    data_type: str | None
    changes: tuple[tuple[tuple[str, ...], Any, Any], ...]
    warnings: tuple[str, ...] = ()


def list_training_datasets(settings) -> list[TrainingDatasetEntry]:
    """List ALL training registry entries, including prepared and ineligible ones."""
    from lisai.data.dataset_registry import (load_dataset_registry, registry_data_types,
                                             registry_value_for_data_type)
    from lisai.infra.paths import Paths

    paths = Paths(settings)
    entries = []
    for name, info in sorted(load_dataset_registry(paths.dataset_registry_path()).items()):
        usage = str(info.get("usage", "training"))
        if usage != "training":
            continue
        data_types = tuple(sorted(registry_data_types(info)))
        roles: set[str] = set()
        for data_type in (data_types or (None,)):
            outputs = registry_value_for_data_type(info, "outputs", data_type)
            if isinstance(outputs, list):
                for item in outputs:
                    if isinstance(item, dict) and item.get("role"):
                        roles.add(str(item["role"]))
        raw_dir = paths.dataset_dir(dataset_name=name, usage="training") / "raw"
        count = sum(1 for path in raw_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}) if raw_dir.is_dir() else 0
        fmt = info.get("data_format")
        eligible = bool(info.get("for_training", True)) and fmt in {"single", "timelapse"} and count >= 2
        entries.append(TrainingDatasetEntry(
            name=name, data_format=str(fmt) if fmt is not None else None,
            usage=usage, for_training=bool(info.get("for_training", True)),
            data_types=data_types, available_roles=tuple(sorted(roles)),
            description=str(info["description"]) if info.get("description") is not None else None,
            quick_hdn_eligible=eligible, quick_recording_count=count,
        ))
    return entries


def training_recording_sources(
    recordings: Sequence[TrainingRecording], files: Sequence[str | Path], data_format: str,
    *, minimum_recordings: int = 1,
) -> list[tuple[str, np.ndarray] | Path]:
    """Normalize selected host arrays and TIFFs for import_training_dataset()."""
    if data_format not in {"single", "timelapse"}:
        raise ValueError("Data format must be 'single' (YX) or 'timelapse' (TYX).")
    expected = ("Y", "X") if data_format == "single" else ("T", "Y", "X")
    sources: list[tuple[str, np.ndarray] | Path] = []
    for i, recording in enumerate(recordings, 1):
        array = np.asarray(recording.data)
        if tuple(recording.axes) != expected or array.ndim != len(expected):
            raise ValueError(f"Recording {recording.name!r} does not have {''.join(expected)} axes.")
        name = str(recording.name or f"recording_{i:04d}")
        name = name.replace("/", "_").replace("\\", "_")
        for char in '<>:"|?*':
            name = name.replace(char, "_")
        name = name.strip().rstrip(". ") or f"recording_{i:04d}"
        if not name.lower().endswith((".tif", ".tiff")):
            name += ".tif"
        sources.append((name, array))
    unique_paths: set[Path] = set()
    for file in files:
        path = Path(file).expanduser().resolve()
        if path in unique_paths:
            continue
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Not a TIFF recording: {path}")
        unique_paths.add(path)
        sources.append(path)
    if len(sources) < minimum_recordings:
        raise ValueError(f"At least {minimum_recordings} independent recording(s) required.")
    names = [source[0] if isinstance(source, tuple) else source.name for source in sources]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Two recordings share a TIFF filename; rename one before importing.")
    return sources


def preview_dataset_change(document, settings, dataset_name: str,
                           *, data_type: str | None = None) -> DatasetChangePreview:
    """Inspect an explicit dataset change without mutating the config document."""
    from lisai.config.catalog import _registry_scaffold_defaults
    from lisai.data.dataset_registry import load_dataset_registry, registry_data_types
    from lisai.infra.paths import Paths
    import yaml

    paths = Paths(settings)
    info = load_dataset_registry(paths.dataset_registry_path()).get(dataset_name)
    if info is None or info.get("usage", "training") != "training" or info.get("for_training", True) is False:
        raise ValueError(f"Dataset {dataset_name!r} is not registered for training.")
    cfg = yaml.safe_load(document.yaml_text)
    if not isinstance(cfg, dict):
        raise ValueError("Training YAML must be a mapping before applying a dataset.")
    old_data = cfg.get("data", {})
    if not isinstance(old_data, dict):
        raise ValueError("Training YAML data section must be a mapping.")
    data_types = sorted(registry_data_types(info))
    if data_type is not None and data_type not in data_types:
        raise ValueError(f"Dataset {dataset_name!r} has no {data_type!r} data type.")
    if data_type is None and len(data_types) > 1:
        raise ValueError(f"Select a data type for {dataset_name!r}: {', '.join(data_types)}")
    selected_type = data_type or (data_types[0] if data_types else None)
    # Only select an output when the registry has an unambiguous default/role.
    proposed_cfg = {"data": {"data_type": selected_type}, "routing": {"data_subfolder": selected_type}}
    defaults = _registry_scaffold_defaults(paths=paths, dataset_name=dataset_name,
                                           cfg=proposed_cfg)
    updates: list[tuple[tuple[str, ...], Any, Any]] = []
    for key, value in (("dataset_name", dataset_name), ("input", defaults.input_name),
                       ("data_format", defaults.data_format)):
        if value is not None and old_data.get(key) != value:
            updates.append((("data", key), old_data.get(key), value))
    warnings: list[str] = []
    if old_data.get("paired"):
        if defaults.target_name:
            if old_data.get("target") != defaults.target_name:
                updates.append((("data", "target"), old_data.get("target"), defaults.target_name))
        else:
            warnings.append("No unambiguous paired target in the registry; review data.target manually.")
    if defaults.input_name is None:
        warnings.append("No unambiguous input in the registry; review data.input manually.")
    # The quick-import API has one known physical layout: dataset/raw/*.tif.
    # Offer those changes in the explicit preview, never during selection.
    if selected_type == "raw" and defaults.input_name == "raw" and (
            paths.dataset_dir(dataset_name=dataset_name, usage="training") / "raw").is_dir():
        routing = cfg.get("routing", {})
        if isinstance(routing, dict) and routing.get("data_subfolder") != "":
            updates.append((("routing", "data_subfolder"), routing.get("data_subfolder"), ""))
        for key in ("prep_before", "already_split"):
            if old_data.get(key) is not False:
                updates.append((("data", key), old_data.get(key), False))
    else:
        warnings.append("Review routing.data_subfolder and data preparation/splitting for this dataset.")
    return DatasetChangePreview(dataset_name, selected_type, tuple(updates), tuple(warnings))


def apply_dataset_change(document, preview: DatasetChangePreview) -> None:
    """Commit one already-reviewed preview atomically to the shared YAML document."""
    from .schema_editor import update_yaml_field
    text = document.yaml_text
    # Check that the preview is still based on the current text.
    for path, old, _new in preview.changes:
        if document.read_field(path) != old:
            raise RuntimeError("Configuration changed after dataset preview. Review the selection again.")
    for path, _old, new in preview.changes:
        text = update_yaml_field(text, path, new)
    document.replace_text(text)


__all__ = ["TrainingRecording", "TrainingDatasetEntry", "DatasetChangePreview",
           "list_training_datasets", "training_recording_sources",
           "preview_dataset_change", "apply_dataset_change"]
