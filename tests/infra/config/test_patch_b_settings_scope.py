"""Patch B: resolution stays within the settings chosen for an operation."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.config import save_yaml
from lisai.config.catalog import create_training_config_from_preset, discover_training_configs
from lisai.config.cli import main as configs_main
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.resources import resource_directory
from lisai.config.io.resolver import resolve_config_dict
from lisai.config.json_schema.experiment import experiment_json_schema
from lisai.config.json_schema.preprocess import preprocess_json_schema
from lisai.config.settings import Settings
from lisai.evaluation.defaults import resolve_apply_config, resolve_inference_config_path
from lisai.infra.paths import Paths
from lisai.preprocess import cli as preprocess_cli
from lisai.training import cli as training_cli


def _build_tree(stg: Settings, label: str) -> None:
    root = stg.TRAINING_CONFIG_DIR
    (root / "local").mkdir(parents=True)
    (stg.PREPROCESS_CONFIG_DIR).mkdir(parents=True)
    stg.INFERENCE_CONFIG_DIR.mkdir(parents=True)

    save_yaml({"metadata": {"kind": "example", "name": label},
               "experiment": {"mode": "train", "exp_name": label},
               "data": {"dataset_name": label},
               "model": {"architecture": "unet", "parameters": {}}},
              root / "local" / "common.yml")
    save_yaml({"dataset_name": label, "pipeline": "single_recon", "data_type": "recon",
               "fmt": "single", "pipeline_cfg": {}},
              stg.PREPROCESS_CONFIG_DIR / "common.yml")
    save_yaml({"apply": {"inference": {"tiling_size": 256 if label == "A" else 512}}},
              stg.INFERENCE_CONFIG_DIR / "defaults.yml")
    save_yaml({"apply": {"inference": {"crop_size": 64 if label == "A" else 96}}},
              stg.INFERENCE_CONFIG_DIR / "common.yml")

    path = Paths(stg).dataset_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml({label: {"data_format": "single", "defaults": {"recon": {"input": "inp"}},
                       "outputs": {"recon": [{"key": "inp", "path": "inp", "role": "inp", "axes": "YX"}]}}},
              path)


def test_operation_scoped_config_paths_and_registries(settings_factory, tmp_path, capsys):
    a, b = settings_factory("A"), settings_factory("B")
    for stg, label in ((a, "A"), (b, "B")):
        _build_tree(stg, label)

    assert "local/common.yml" in [x.relative_path for x in discover_training_configs(stg=a)]
    assert "presets/denoising_hdn_unsup.yml" in [x.relative_path for x in discover_training_configs(stg=a)]
    for stg, label in ((a, "A"), (b, "B"), (a, "A")):
        train_resolver = ConfigPathResolver("training", stg=stg)
        prep_resolver = ConfigPathResolver("preprocess", stg=stg)
        inf_resolver = ConfigPathResolver("inference", stg=stg)
        assert train_resolver.resolve("local/common").is_relative_to(stg.CONFIGS_ROOT)
        assert prep_resolver.resolve("common").is_relative_to(stg.CONFIGS_ROOT)
        assert inf_resolver.resolve("common") == (stg.INFERENCE_CONFIG_DIR / "common.yml").resolve()
        with pytest.raises(ValueError, match="must be instantiated"):
            training_cli.resolve_config_path("presets/denoising_hdn_unsup", resolver=train_resolver)

        cfg = resolve_config_dict({"experiment": {"mode": "train", "exp_name": label},
                                   "data": {"dataset_name": label, "input": "inp"},
                                   "model": {"architecture": "unet", "parameters": {}}}, stg=stg)
        assert cfg.data.dataset_name == label
        assert cfg.data.dataset_info is not None
        assert cfg.data.input == "inp"

        schema = experiment_json_schema(stg=stg)
        data_def = schema["properties"]["data"]["$ref"].rsplit("/", 1)[-1]
        assert schema["$defs"][data_def]["properties"]["dataset_name"]["examples"] == [label]
        assert "pipeline" in preprocess_json_schema(stg=stg)["properties"]
        assert resolve_inference_config_path("common", stg=stg).is_relative_to(stg.CONFIGS_ROOT)
        assert resolve_apply_config(stg=stg).inference.tiling_size == (256 if label == "A" else 512)
        assert resolve_apply_config(config="common", stg=stg).inference.crop_size == (64 if label == "A" else 96)

        result = configs_main(["list", "--kind", "example"], settings=stg)
        assert result == 0
        assert "examples/continue_training.yml" in capsys.readouterr().out

        out, _ = create_training_config_from_preset(
            "denoising_hdn_unsup", exp_name=f"made_{label}", dataset_name=label,
            stg=stg, overwrite=True,
        )
        assert out == (stg.TRAINING_CONFIG_DIR / f"made_{label}.yml").resolve()
        assert out.is_file()

    assert not (a.TRAINING_CONFIG_DIR / "made_B.yml").exists()
    assert not (b.TRAINING_CONFIG_DIR / "made_A.yml").exists()


def test_training_and_preprocess_command_resolve_one_selected_context(settings_factory, monkeypatch):
    a, b = settings_factory("command_a"), settings_factory("command_b")
    for stg, label in ((a, "A"), (b, "B")):
        _build_tree(stg, label)

    recorded: list[tuple[str, Path, object]] = []
    monkeypatch.setattr(training_cli, "run_training", lambda path, *, settings, **kw:
                        recorded.append(("train", path, settings)))
    monkeypatch.setattr(preprocess_cli, "run_preprocess_config", lambda path, *, paths:
                        recorded.append(("preprocess", path, paths.settings)))

    for stg in (a, b, a):
        training_cli.run_from_args(
            argparse.Namespace(config="local/common", config_option=None, progress_bar=None),
            argparse.ArgumentParser(), settings=stg,
        )
        preprocess_cli.run_from_args(
            argparse.Namespace(config="common", config_option=None),
            argparse.ArgumentParser(), settings=stg,
        )

    assert [item[2] for item in recorded] == [a, a, b, b, a, a]
    assert all(path.is_relative_to(stg.CONFIGS_ROOT) for _, path, stg in recorded)


def test_config_path_resolver_requires_settings():
    with pytest.raises(TypeError, match="stg"):
        ConfigPathResolver("training")
