"""M2 C: immutable package YAMLs and independently writable user configs."""
from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai.config import load_yaml, save_yaml
from lisai.config import cli as configs_cli
from lisai.config.catalog import (
    create_training_config_from_preset,
    discover_training_configs,
    resolve_training_preset_path,
)
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.resources import find_resource, iter_resource_files, resource_directory
from lisai.config.settings import Settings
from lisai.evaluation.defaults import load_inference_config, resolve_apply_config
from lisai.promoted_models.dependencies import PromotionFile
from lisai.promoted_models.inference_config import resolve_promoted_inference_config_source
from lisai.promoted_models.package import _copy_plan_files


def test_training_package_discovery_never_reads_shadow_presets(settings_factory):
    stg = settings_factory("resource-shadow")
    shadow = stg.TRAINING_CONFIG_DIR / "presets" / "denoising_hdn_unsup.yml"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("experiment: not-our-preset\n", encoding="utf-8")

    packaged = resolve_training_preset_path("denoising_hdn_unsup", stg=stg)
    assert packaged != shadow.resolve()
    assert load_yaml(packaged)["experiment"]["task"]["name"] == "denoising_hdn"
    entries = discover_training_configs(kind="preset", stg=stg)
    assert len(entries) == 6
    assert all(entry.ownership == "built-in" for entry in entries)
    assert "presets/denoising_hdn_unsup.yml" in {entry.relative_path for entry in entries}


def test_workspaces_share_package_but_not_editable_configs(settings_factory, tmp_path):
    a = Settings.from_workspace(settings_factory("A").WORKSPACE_ROOT)
    b = Settings.from_workspace(settings_factory("B").WORKSPACE_ROOT)
    for stg, name in ((a, "A"), (b, "B")):
        folder = stg.TRAINING_CONFIG_DIR / "local"
        folder.mkdir(parents=True)
        save_yaml({"metadata": {"kind": "local", "name": name},
                   "experiment": {"mode": "train", "exp_name": name},
                   "data": {"dataset_name": name}}, folder / "same.yml")
    for stg, name in ((a, "A"), (b, "B"), (a, "A")):
        resolver = ConfigPathResolver("training", stg=stg)
        assert load_yaml(resolver.resolve("local/same"))["data"]["dataset_name"] == name
        assert resolver.resolve_reference("local/same").ownership == "local"
        assert resolver.resolve_reference("presets/denoising_hdn_sup").ownership == "built-in"
        assert not (stg.TRAINING_CONFIG_DIR / "presets").exists()
        output, _ = create_training_config_from_preset(
            "denoising_hdn_unsup", exp_name=f"new_{name}", dataset_name="CHANGEME", stg=stg, overwrite=True,
        )
        assert output == stg.TRAINING_CONFIG_DIR / "local" / f"new_{name}.yml"
        assert output.is_file()
    assert not (a.TRAINING_CONFIG_DIR / "local" / "new_B.yml").exists()


def test_namespace_and_ambiguous_shorthand_are_explicit(settings_factory):
    stg = settings_factory("collisions")
    infer = stg.INFERENCE_CONFIG_DIR / "local"
    infer.mkdir(parents=True)
    local = infer / "hdn_sup.yaml"
    local.write_text("apply:\n  inference:\n    lvae_num_samples: 4\n", encoding="utf-8")
    resolver = ConfigPathResolver("inference", stg=stg)
    with pytest.raises(ValueError, match="Ambiguous"):
        resolver.resolve("hdn_sup")
    assert resolver.resolve("local/hdn_sup") == local.resolve()
    built = resolver.resolve("presets/hdn_sup")
    assert load_yaml(built)["apply"]["inference"]["lvae_num_samples"] == 10
    assert resolver.resolve(local.resolve()) == local.resolve()
    with pytest.raises(ValueError, match="Local config"):
        resolver.resolve("local/../presets/hdn_sup")


def test_inference_package_preset_and_model_bundle_source(settings_factory):
    stg = settings_factory("model-preset")
    src = resolve_promoted_inference_config_source("presets/hdn_sup", stg=stg)
    assert load_yaml(src)["apply"]["inference"]["lvae_num_samples"] == 10
    assert load_inference_config("presets/hdn_sup", stg=stg)[0].apply is not None
    assert resolve_apply_config(config="presets/hdn_sup", stg=stg).inference.lvae_num_samples == 10
    with pytest.raises(ValueError, match="does not define an 'apply'"):
        absent_apply = stg.INFERENCE_CONFIG_DIR / "local" / "not_apply.yml"
        absent_apply.parent.mkdir(parents=True, exist_ok=True)
        save_yaml({"evaluate": {"metrics": ["psnr"]}}, absent_apply)
        resolve_promoted_inference_config_source(absent_apply, stg=stg)


def test_zip_traversable_read_and_package_copy(tmp_path):
    zname = tmp_path / "resources.zip"
    with zipfile.ZipFile(zname, "w") as archive:
        archive.writestr("presets/hdn_sup.yaml", "apply:\n  inference:\n    lvae_num_samples: 10\n")
    with zipfile.ZipFile(zname) as archive:
        source = zipfile.Path(archive, at="presets/hdn_sup.yaml")
        assert load_yaml(source)["apply"]["inference"]["lvae_num_samples"] == 10
        root = tmp_path / "model"
        root.mkdir()
        _copy_plan_files(SimpleNamespace(copy_files=(PromotionFile(source, "config_inference.yaml"),)), root)
        assert (root / "config_inference.yaml").read_bytes() == source.read_bytes()


def test_config_cli_labels_resource_ownership(settings_factory, capsys):
    stg = settings_factory("cli")
    assert configs_cli.main(["list", "--kind", "preset"], settings=stg) == 0
    output = capsys.readouterr().out
    assert "built-in" in output
    assert "presets/denoising_hdn_unsup.yml" in output
    assert configs_cli.main(["info", "presets/denoising_hdn_sup"], settings=stg) == 0
    assert "owner: built-in" in capsys.readouterr().out
    assert configs_cli.main(["validate", "presets/denoising_hdn_sup"], settings=stg) == 0
    assert "(built-in)" in capsys.readouterr().out
    assert configs_cli.main(["new", "denoising_hdn_unsup", "--name", "demo", "--dataset", "CHANGEME"], settings=stg) == 0
    assert (stg.TRAINING_CONFIG_DIR / "local" / "demo.yml").is_file()
    assert "Created training config" in capsys.readouterr().out


def test_resources_are_read_only_and_have_no_checkout_requirement():
    assert len(list(iter_resource_files("training", "examples"))) >= 3
    assert find_resource("inference", "presets", "hdn") is not None
    with pytest.raises(ValueError, match="without '..'"):
        find_resource("inference", "presets", "../local/secret")
    assert resource_directory("configs", "training", "presets").is_dir()
