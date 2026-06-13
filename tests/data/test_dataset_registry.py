from __future__ import annotations

from pathlib import Path

import pytest

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.data.dataset_registry import (
    DatasetRegistryError,
    load_dataset_info,
    load_dataset_registry,
    registry_data_format_for_output,
    save_dataset_registry,
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


def test_registry_data_format_for_output_uses_override_or_dataset_format():
    info = {
        "data_format": "mltpl_snr",
        "outputs": {
            "recon": [
                {"key": "inp_mltpl_snr", "path": "inp_mltpl_snr", "role": "inp", "axes": "TYX"},
                {
                    "key": "inp_single",
                    "path": "inp_single",
                    "role": "inp",
                    "axes": "YX",
                    "data_format_override": "single",
                },
            ]
        },
    }

    assert registry_data_format_for_output(info, "recon", "inp_mltpl_snr") == "mltpl_snr"
    assert registry_data_format_for_output(info, "recon", "inp_single") == "single"


def test_registry_data_format_for_output_does_not_infer_from_axes_without_override():
    info = {
        "data_format": "mltpl_snr",
        "outputs": {
            "recon": [
                {"key": "inp_single", "path": "inp_single", "role": "inp", "axes": "YX"},
            ]
        },
    }

    assert registry_data_format_for_output(info, "recon", "inp_single") == "mltpl_snr"


def test_save_dataset_registry_compacts_numeric_size_ranges(tmp_path: Path):
    path = tmp_path / "dataset_registry.yml"

    save_dataset_registry(
        {
            "demo": {
                "data_format": "timelapse",
                "size": {
                    "recon": {
                        "n_files": 2,
                        "timepoints": [10, 26],
                        "snr_levels": [5, 6],
                    }
                },
            }
        },
        path,
    )

    registry = load_yaml(path)
    assert registry["demo"]["size"]["recon"]["timepoints"] == {"min": 10, "max": 26}
    assert registry["demo"]["size"]["recon"]["snr_levels"] == {"min": 5, "max": 6}
