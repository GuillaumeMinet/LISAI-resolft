from __future__ import annotations

from pathlib import Path

import pytest

from lisai.config.io.yaml import save_yaml
from lisai.data.dataset_registry import (
    DatasetRegistryError,
    load_dataset_info,
    load_dataset_registry,
)


def test_load_dataset_registry_reads_flat_data_format(tmp_path: Path):
    path = tmp_path / "dataset_registry.yml"
    save_yaml(
        {
            "demo": {
                "data_format": "single",
                "structure": {"recon": ["inp", "gt"]},
            }
        },
        path,
    )

    registry = load_dataset_registry(path)

    assert registry == {
        "demo": {
            "data_format": "single",
            "structure": {"recon": ["inp", "gt"]},
        }
    }


def test_load_dataset_registry_maps_legacy_format_to_data_format(tmp_path: Path):
    path = tmp_path / "dataset_registry.yml"
    save_yaml(
        {"demo": {"format": "timelapse", "size": {"recon": {"n_files": 2}}}},
        path,
    )

    info = load_dataset_info(path, "demo")

    assert info == {
        "data_format": "timelapse",
        "size": {"recon": {"n_files": 2}},
    }


def test_load_dataset_registry_rejects_nested_datasets_layout(tmp_path: Path):
    path = tmp_path / "dataset_registry.yml"
    save_yaml({"datasets": {"demo": {"data_format": "single"}}}, path)

    with pytest.raises(DatasetRegistryError, match="flat layout"):
        load_dataset_registry(path)


def test_load_dataset_registry_returns_empty_for_missing_file(tmp_path: Path):
    assert load_dataset_registry(tmp_path / "missing.yml") == {}
