"""Patch D: preprocessing and dataset services retain the selected configuration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.settings import Settings
from lisai.data.catalog import DatasetCatalog
from lisai.data.cli import list_datasets
from lisai.data.dataset_registry import load_dataset_registry
from lisai.data.install import resolve_dataset_plan
from lisai.data.noise_model import (
    NoiseModelBuildConfig,
    NoiseModelCheckConfig,
    build_noise_model,
    check_noise_model,
)
from lisai.infra.paths import Paths
from lisai.preprocess import PreprocessRun


def _settings_with_format(settings: Settings, *, label: str, suffix: str) -> Settings:
    # Deliberately different *on-disk* legacy config, not just monkeypatched globals.
    root = settings.CONFIGS_ROOT
    project = load_yaml(root / "project_config.yml")
    project["naming"]["sample_id"] = f"{label}{{id:02d}}"
    save_yaml(project, root / "project_config.yml")
    data = load_yaml(root / "data_config.yml")
    data["data_types"]["recon"] = [suffix]
    data["format"]["single"]["recon"] = f"{{sample_id}}{suffix}"
    data["logs"]["recon_preprocess"] = f"preprocess_{label}.yaml"
    save_yaml(data, root / "data_config.yml")
    return Settings.from_config_dir(root)


def _cfg():
    return {
        "dataset_name": "common",
        "pipeline": "single_recon",
        "data_type": "recon",
        "fmt": "single",
        "pipeline_cfg": {},
        "split": {"enabled": False},
    }


def test_real_preprocess_a_b_a_has_independent_extensions_names_and_registry(settings_factory, capsys):
    a = _settings_with_format(settings_factory("a"), label="a", suffix=".tif")
    b = _settings_with_format(settings_factory("b"), label="b", suffix=".tiff")
    results = {}
    for stg, label, suffix in ((a, "a", ".tif"), (b, "b", ".tiff"), (a, "a", ".tif")):
        paths = Paths(stg)
        dump = paths.dataset_dump_dir(dataset_name="common", data_type="recon")
        dump.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(dump / f"original{suffix}", np.full((8, 8), 3, dtype=np.uint16))
        # Other extension is deliberately present but excluded by each configuration.
        alternate = ".tiff" if suffix == ".tif" else ".tif"
        tifffile.imwrite(dump / f"ignored{alternate}", np.full((8, 8), 9, dtype=np.uint16))

        run = PreprocessRun.from_cfg(_cfg(), paths=paths)
        assert run.paths is paths
        result = run.execute(overwrite=True)
        assert result.n_files == 1
        target = paths.dataset_preprocess_dir(dataset_name="common", data_type="recon")
        saved = target / f"{label}00{suffix}"
        assert saved.is_file()
        np.testing.assert_array_equal(tifffile.imread(saved), np.full((8, 8), 3, dtype=np.uint16))
        assert not (target / f"{'b' if label == 'a' else 'a'}00{suffix}").exists()
        assert (target / f"preprocess_{label}.yaml").is_file()
        assert load_dataset_registry(paths.dataset_registry_path())["common"]["size"]["recon"]["n_files"] == 1
        list_datasets(paths=paths)
        assert "common" in capsys.readouterr().out
        results[label] = target

    assert results["a"] != results["b"]
    assert (results["a"] / "a00.tif").exists()
    assert (results["b"] / "b00.tiff").exists()


def test_preprocess_internal_factory_requires_paths():
    with pytest.raises(TypeError, match="paths"):
        PreprocessRun.from_cfg(_cfg())


def test_dataset_catalog_resolution_a_b_a_stays_in_registry(settings_factory):
    a, b = settings_factory("catalog_a"), settings_factory("catalog_b")
    registry_entry = {"usage": "training", "data_format": "single"}
    catalog = DatasetCatalog.model_validate({
        "schema_version": 1,
        "datasets": {"common": {
            "usage": "training", "install_path": "datasets/training",
            "archives": [{"filename": "common.zip", "size_bytes": 1, "sha256": "a" * 64}],
            "registry_entry": registry_entry,
        }},
    })
    paths_a, paths_b = Paths(a), Paths(b)
    root_a = paths_a.dataset_dir(dataset_name="common")
    root_a.mkdir(parents=True)
    registry_path = paths_a.dataset_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml({"common": registry_entry}, registry_path)
    assert [resolve_dataset_plan(catalog, ["common"], paths=p).entries[0].status
            for p in (paths_a, paths_b, paths_a)] == ["installed", "install", "installed"]
    assert not paths_b.dataset_registry_path().exists()


def test_noise_model_builder_and_checker_respect_selected_root(settings_factory):
    a, b = settings_factory("noise_a"), settings_factory("noise_b")
    cfg = NoiseModelBuildConfig(
        dataset_name="common", signal_subfolder="preprocess/recon", save_name="test_noise",
        display=False, save_gmm_plot=False, overwrite=True,
    )
    check_cfg = NoiseModelCheckConfig(
        dataset_name="common", signal_subfolder="preprocess/recon", noise_model_name="test_noise",
    )
    for stg in (a, b, a):
        paths = Paths(stg)
        (paths.dataset_dir(dataset_name="common") / "preprocess" / "recon").mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="No valid files found") as exc:
            build_noise_model(cfg, paths=paths, device=torch.device("cpu"))
        assert str(paths.dataset_dir(dataset_name="common")) in str(exc.value)
        with pytest.raises(FileNotFoundError) as exc:
            check_noise_model(check_cfg, paths=paths, device=torch.device("cpu"))
        assert str(paths.noise_model_path(noiseModel_name="test_noise")) in str(exc.value)
