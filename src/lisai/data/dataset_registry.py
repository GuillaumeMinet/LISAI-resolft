from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lisai.config.io.yaml import load_yaml, save_yaml

DATA_FORMAT_KEY = "data_format"
LEGACY_FORMAT_KEY = "format"
NESTED_DATASETS_KEY = "datasets"


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


def save_dataset_registry(registry: Mapping[str, Any], path: str | Path) -> None:
    """Persist registry metadata using the canonical flat on-disk contract."""
    save_yaml(normalize_dataset_registry(registry), path)


__all__ = [
    "DATA_FORMAT_KEY",
    "DatasetRegistryError",
    "load_dataset_info",
    "load_dataset_registry",
    "normalize_dataset_registry",
    "save_dataset_registry",
]
