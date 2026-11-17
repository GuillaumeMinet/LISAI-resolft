"""Patch C: writable configs are flat, names deterministic, built-ins independent."""
from __future__ import annotations

from pathlib import Path

import pytest

from lisai.cli import main as root_cli
from lisai.config import cli as configs_cli
from lisai.config.catalog import (
    create_training_config_from_preset, discover_training_configs,
)
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.resources import find_resource
from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.settings import Settings
from lisai.preprocess.cli import resolve_config_path as resolve_preprocess_config
from lisai.preprocess.templates import create_preprocess_config_from_template


def test_flat_user_names_explicit_paths_and_package_are_disjoint(settings_factory, tmp_path, monkeypatch):
    stg = Settings.from_workspace(settings_factory("flat").WORKSPACE_ROOT)
    writable = stg.TRAINING_CONFIG_DIR
    (writable / "local").mkdir(parents=True)
    (writable / "vimentin").mkdir(parents=True)
    save_yaml({"experiment": {"exp_name": "managed"}}, writable / "foo.yml")
    save_yaml({"experiment": {"exp_name": "nested"}}, writable / "vimentin" / "upsamp.yml")
    save_yaml({"experiment": {"exp_name": "local-is-valid"}}, writable / "local" / "foo.yml")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    save_yaml({"experiment": {"exp_name": "cwd"}}, cwd / "foo.yml")
    monkeypatch.chdir(cwd)
    resolver = ConfigPathResolver("training", stg=stg)
    assert resolver.resolve("foo") == (writable / "foo.yml").resolve()
    assert resolver.resolve("vimentin/upsamp") == (writable / "vimentin/upsamp.yml").resolve()
    assert resolver.resolve("local/foo") == (writable / "local/foo.yml").resolve()
    assert resolver.resolve("./foo.yml") == (cwd / "foo.yml").resolve()
    assert resolver.resolve((cwd / "foo.yml").resolve()) == (cwd / "foo.yml").resolve()
    assert resolver.resolve_reference("foo").ownership == "local"
    assert resolver.resolve_reference("presets/denoising_hdn_unsup").ownership == "built-in"
    assert {"foo.yml", "local/foo.yml", "vimentin/upsamp.yml"} <= set(resolver.available())
    with pytest.raises(FileNotFoundError, match="not found"):
        resolver.resolve("denoising_hdn_unsup")  # no implicit package alias


def test_shorthand_rejects_traversal_reserved_prefix_and_symlink_escape(settings_factory, tmp_path):
    stg = Settings.from_workspace(settings_factory("safety").WORKSPACE_ROOT)
    root = stg.TRAINING_CONFIG_DIR
    root.mkdir()
    resolver = ConfigPathResolver("training", stg=stg)
    with pytest.raises(ValueError, match="Invalid managed"):
        resolver.resolve("nested/../missing")
    with pytest.raises(ValueError, match="Reserved packaged"):
        create_training_config_from_preset(
            "denoising_hdn_unsup", exp_name="x", output="presets/replaced", stg=stg)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "bad.yml").write_text("test: bad\n")
    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks unavailable")
    with pytest.raises(ValueError, match="escapes writable root"):
        resolver.resolve("linked/bad")
    with pytest.raises(ValueError, match="escapes writable root"):
        create_training_config_from_preset(
            "denoising_hdn_unsup", exp_name="x", output="linked/new.yml", stg=stg)


def test_training_and_preprocess_creation_nested_names_and_relative_schema_links(settings_factory, tmp_path, monkeypatch):
    stg = Settings.from_workspace(settings_factory("creation").WORKSPACE_ROOT)
    train, _ = create_training_config_from_preset(
        "denoising_hdn_unsup", exp_name="demo", dataset_name="example",
        output="vimentin/example", stg=stg)
    assert train == (stg.TRAINING_CONFIG_DIR / "vimentin/example.yml").resolve()
    schema_ref = train.read_text().splitlines()[0].split("$schema=", 1)[1]
    assert (train.parent / schema_ref).resolve() == stg.CONFIGS_ROOT / "schema/experiment.schema.json"
    preprocess, _ = create_preprocess_config_from_template(
        "single_recon", dataset_name="demo", output="vimentin/recon", stg=stg)
    assert preprocess == (stg.PREPROCESS_CONFIG_DIR / "vimentin/recon.yml").resolve()
    assert load_yaml(preprocess)["pipeline"] == "single_recon"
    hint = preprocess.read_text().splitlines()[0].split("$schema=", 1)[1]
    assert (preprocess.parent / hint).resolve() == stg.CONFIGS_ROOT / "schema/preprocess.schema.json"
    assert resolve_preprocess_config("vimentin/recon", resolver=ConfigPathResolver("preprocess", stg=stg)) == preprocess
    assert find_resource("preprocess", "templates", "single_recon") is not None
    with pytest.raises(ValueError, match="cannot be run directly"):
        resolve_preprocess_config("templates/single_recon", resolver=ConfigPathResolver("preprocess", stg=stg))
    with pytest.raises(FileExistsError):
        create_preprocess_config_from_template("single_recon", dataset_name="demo", output="vimentin/recon", stg=stg)

    # An explicit output path is relative to the terminal, not the config root.
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    external, _ = create_preprocess_config_from_template(
        "single_recon", dataset_name="out", output="./my_recon.yml", stg=stg)
    assert external == cwd / "my_recon.yml"
    header = external.read_text().splitlines()[0].split("$schema=", 1)[1]
    assert (external.parent / header).resolve() == stg.CONFIGS_ROOT / "schema/preprocess.schema.json"


def test_flat_catalog_does_not_confuse_folder_named_local_with_owner(settings_factory):
    stg = Settings.from_workspace(settings_factory("catalog").WORKSPACE_ROOT)
    (stg.TRAINING_CONFIG_DIR / "local").mkdir(parents=True)
    save_yaml({"metadata": {"kind": "example", "name": "organization"},
               "experiment": {"task": {"name": "denoising_hdn"}}},
              stg.TRAINING_CONFIG_DIR / "local" / "group.yml")
    entries = discover_training_configs(kind="local", stg=stg)
    assert len(entries) == 1
    assert entries[0].relative_path == "local/group.yml"
    assert entries[0].kind == "local"  # metadata cannot misclassify user-owned file
    assert entries[0].ownership == "local"
    assert len(discover_training_configs(kind="preset")) >= 1
    with pytest.raises(ValueError, match="selected workspace"):
        discover_training_configs()


def test_only_explicit_package_catalog_commands_work_without_selection(monkeypatch, capsys):
    monkeypatch.delenv("LISAI_HOME", raising=False)
    monkeypatch.setattr("lisai.config.selection._source_checkout_for_import", lambda: None)
    assert root_cli(["configs", "list", "--kind", "preset"]) == 0
    assert "presets/denoising_hdn_unsup.yml" in capsys.readouterr().out
    assert root_cli(["configs", "list", "--domain", "preprocess", "--kind", "template"]) == 0
    assert "templates/single_recon.yml" in capsys.readouterr().out
    assert root_cli(["configs", "info", "presets/denoising_hdn_unsup"]) == 0
    assert "owner: built-in" in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing:
        root_cli(["configs", "info", "presets/not_a_real_preset"])
    assert missing.value.code == 1
    assert "Packaged training config not found" in capsys.readouterr().err
    assert root_cli(["configs", "list"]) == 2
    assert "LISAI_HOME" in capsys.readouterr().err
    assert root_cli(["configs", "list", "--kind", "local"]) == 2
    assert "LISAI_HOME" in capsys.readouterr().err


def test_flat_seeded_inference_default_not_implicitly_created(settings_factory):
    from lisai.config.user_configs import seed_inference_local_configs
    from lisai.evaluation.defaults import resolve_apply_config, resolve_inference_config_path

    stg = Settings.from_workspace(settings_factory("core").WORKSPACE_ROOT)
    assert resolve_inference_config_path(None, stg=stg) is None
    assert resolve_apply_config(stg=stg).inference.tiling_size is not None
    assert not stg.INFERENCE_CONFIG_DIR.exists()
    created = seed_inference_local_configs(stg.CONFIGS_ROOT)
    assert {p.name for p in created} == {"defaults.yml", "post_training.yml"}
    assert all(p.parent == stg.INFERENCE_CONFIG_DIR for p in created)
    assert resolve_inference_config_path(None, stg=stg) == (stg.INFERENCE_CONFIG_DIR / "defaults.yml").resolve()
    assert seed_inference_local_configs(stg.CONFIGS_ROOT) == []


def test_sweep_base_config_does_not_search_sweep_folder_before_managed_root(settings_factory, tmp_path):
    from lisai.infra.paths import Paths
    from lisai_runner.queue.cli import _resolve_sweep_base_config

    stg = Settings.from_workspace(settings_factory("sweeps").WORKSPACE_ROOT)
    train_dir = stg.TRAINING_CONFIG_DIR
    (train_dir / "sweeps").mkdir(parents=True)
    base = train_dir / "hdn.yml"
    shadow = train_dir / "sweeps/hdn.yml"
    base.write_text("experiment: root\n")
    shadow.write_text("experiment: shadow\n")
    sweep = train_dir / "sweeps/multiple.yml"
    assert _resolve_sweep_base_config("hdn.yml", sweep_path=sweep, paths=Paths(stg)) == base.resolve()
    assert _resolve_sweep_base_config("sweeps/hdn.yml", sweep_path=sweep, paths=Paths(stg)) == shadow.resolve()
    assert _resolve_sweep_base_config(str(shadow), sweep_path=sweep, paths=Paths(stg)) == shadow.resolve()
