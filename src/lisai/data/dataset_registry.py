from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lisai.config.io.yaml import load_yaml, save_yaml

DATA_FORMAT_KEY = "data_format"
DATA_FORMAT_OVERRIDE_KEY = "data_format_override"
LEGACY_FORMAT_KEY = "format"
NESTED_DATASETS_KEY = "datasets"
RANGE_SUMMARY_KEYS = {"snr_levels", "timepoints"}


class DatasetRegistryError(ValueError):
    """Raised when the dataset registry does not match LISAI's flat contract."""


def normalize_dataset_registry(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return canonical flat registry metadata keyed by dataset name.

    LISAI's on-disk registry contract is a flat mapping:

    ``dataset_name -> dataset metadata``.

    Legacy entries that used ``format`` are read as ``data_format`` so old local
    registries still resolve, but callers only receive the canonical key.
    """
    if NESTED_DATASETS_KEY in raw:
        raise DatasetRegistryError(
            "Dataset registry must use the flat layout. "
            "Remove the top-level `datasets:` block and put dataset names at the file root."
        )

    registry: dict[str, dict[str, Any]] = {}
    for dataset_name, value in raw.items():
        if not isinstance(value, Mapping):
            raise DatasetRegistryError(
                f"Dataset registry entry {dataset_name!r} must be a mapping, got {type(value).__name__}."
            )

        entry = dict(value)
        if entry.get(DATA_FORMAT_KEY) is None and entry.get(LEGACY_FORMAT_KEY) is not None:
            entry[DATA_FORMAT_KEY] = entry[LEGACY_FORMAT_KEY]
        entry.pop(LEGACY_FORMAT_KEY, None)
        registry[str(dataset_name)] = entry

    return registry


def load_dataset_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a flat dataset registry, returning an empty mapping if it is absent."""
    registry_path = Path(path)
    if not registry_path.exists():
        return {}
    return normalize_dataset_registry(load_yaml(registry_path))


def load_dataset_info(path: str | Path, dataset_name: str | None) -> dict[str, Any] | None:
    """Return canonical metadata for one dataset, when present."""
    if not dataset_name:
        return None

    info = load_dataset_registry(path).get(dataset_name)
    return dict(info) if isinstance(info, Mapping) else None


def registry_data_types(info: Mapping[str, Any]) -> set[str]:
    """Return data type keys described by a registry entry."""
    keys: set[str] = set()
    for section_name in ("defaults", "structure", "outputs", "size", "split"):
        section = info.get(section_name)
        if isinstance(section, Mapping):
            keys.update(str(key) for key in section)
    return keys


def registry_value_for_data_type(
    info: Mapping[str, Any],
    section_name: str,
    data_type: str | None,
) -> Any:
    """Return a registry subsection value for a data type when unambiguous."""
    section = info.get(section_name)
    if not isinstance(section, Mapping):
        return None

    if data_type is not None:
        return section.get(data_type)

    if len(section) == 1:
        return next(iter(section.values()))
    return None


def registry_mapping_for_data_type(
    info: Mapping[str, Any],
    section_name: str,
    data_type: str | None,
) -> Mapping[str, Any] | None:
    """Return a registry subsection mapping for a data type when unambiguous."""
    value = registry_value_for_data_type(info, section_name, data_type)
    return value if isinstance(value, Mapping) else None


def registry_output_axes(info: Mapping[str, Any], data_type: str | None, value: Any) -> str | None:
    """Return axes for a registered output key/path, when known."""
    if value is None:
        return None

    outputs = registry_value_for_data_type(info, "outputs", data_type)
    if not isinstance(outputs, list):
        return None

    text = str(value)
    for output in outputs:
        if not isinstance(output, Mapping):
            continue
        if text not in {str(output.get("key")), str(output.get("path"))}:
            continue
        axes = output.get("axes")
        return str(axes) if axes is not None else None
    return None


def registry_data_format_for_output(
    info: Mapping[str, Any],
    data_type: str | None,
    value: Any,
) -> str | None:
    """Return the loader format for an output, honoring explicit overrides."""
    fallback = info.get(DATA_FORMAT_KEY)
    fallback_format = str(fallback) if fallback is not None else None
    if value is None:
        return fallback_format

    outputs = registry_value_for_data_type(info, "outputs", data_type)
    if not isinstance(outputs, list):
        return fallback_format

    text = str(value)
    for output in outputs:
        if not isinstance(output, Mapping):
            continue
        if text not in {str(output.get("key")), str(output.get("path"))}:
            continue
        override = output.get(DATA_FORMAT_OVERRIDE_KEY)
        return str(override) if override is not None else fallback_format
    return fallback_format


def registry_output_data_format_override(info: Mapping[str, Any], data_type: str | None, value: Any) -> str | None:
    """Return an output-specific data format override, when present."""
    if value is None:
        return None

    outputs = registry_value_for_data_type(info, "outputs", data_type)
    if not isinstance(outputs, list):
        return None

    text = str(value)
    for output in outputs:
        if not isinstance(output, Mapping):
            continue
        if text not in {str(output.get("key")), str(output.get("path"))}:
            continue
        override = output.get(DATA_FORMAT_OVERRIDE_KEY)
        return str(override) if override is not None else None
    return None


def registry_paths_for_data_type(info: Mapping[str, Any], data_type: str | None) -> set[str]:
    """Return registered output keys/paths for a data type."""
    paths: set[str] = set()

    structure = registry_value_for_data_type(info, "structure", data_type)
    if isinstance(structure, list):
        paths.update(str(item) for item in structure)

    outputs = registry_value_for_data_type(info, "outputs", data_type)
    if isinstance(outputs, list):
        for output in outputs:
            if isinstance(output, Mapping):
                if output.get("path") is not None:
                    paths.add(str(output["path"]))
                elif output.get("key") is not None:
                    # legacy fallback
                    paths.add(str(output["key"]))
            elif output is not None:
                paths.add(str(output))

    return paths


def summarize_numeric_range(value: Any) -> Any:
    """Return a compact min/max mapping for numeric registry ranges."""
    if not isinstance(value, list):
        return value
    if not value:
        return value
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return value
    return {"min": min(value), "max": max(value)}


def compact_dataset_registry_ranges(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Compact verbose numeric size lists into structured min/max ranges."""
    compacted: dict[str, Any] = {}
    for dataset_name, entry in registry.items():
        if not isinstance(entry, Mapping):
            compacted[str(dataset_name)] = entry
            continue

        compact_entry = dict(entry)
        size = compact_entry.get("size")
        if isinstance(size, Mapping):
            compact_size: dict[str, Any] = {}
            for data_type, size_entry in size.items():
                if isinstance(size_entry, Mapping):
                    compact_size[str(data_type)] = {
                        key: summarize_numeric_range(value) if key in RANGE_SUMMARY_KEYS else value
                        for key, value in size_entry.items()
                    }
                else:
                    compact_size[str(data_type)] = size_entry
            compact_entry["size"] = compact_size

        compacted[str(dataset_name)] = compact_entry
    return compacted


def save_dataset_registry(registry: Mapping[str, Any], path: str | Path) -> None:
    """Persist registry metadata using the canonical flat on-disk contract."""
    save_yaml(compact_dataset_registry_ranges(normalize_dataset_registry(registry)), path)


__all__ = [
    "DATA_FORMAT_KEY",
    "DATA_FORMAT_OVERRIDE_KEY",
    "DatasetRegistryError",
    "compact_dataset_registry_ranges",
    "load_dataset_info",
    "load_dataset_registry",
    "normalize_dataset_registry",
    "registry_data_format_for_output",
    "registry_data_types",
    "registry_mapping_for_data_type",
    "registry_output_axes",
    "registry_output_data_format_override",
    "registry_paths_for_data_type",
    "registry_value_for_data_type",
    "save_dataset_registry",
    "summarize_numeric_range",
]
