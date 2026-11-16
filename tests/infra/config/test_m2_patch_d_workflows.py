"""M2 D: preprocess templates, inference local defaults, and explicit editor schema refresh."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lisai.config import cli as config_cli
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.io.resources import find_resource, iter_resource_files
from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.models.inference import ApplyOverrides, ApplyInferenceOverrides
from lisai.config.settings import Settings
from lisai.config.user_configs import refresh_user_schemas, seed_inference_local_configs
from lisai.evaluation.defaults import resolve_apply_config, resolve_evaluate_config
from lisai.infra.paths import Paths
from lisai.preprocess.cli import resolve_config_path
from lisai.preprocess.core.config import PreprocessConfig
from lisai.preprocess.pipelines import PIPELINES_REGISTRY
from lisai.preprocess.run_preprocess import PreprocessRun
from lisai.preprocess.templates import (
    create_preprocess_config_from_template, copy_legacy_preprocess_configs,
)


def test_packaged_templates_are_generic_and_inert(settings_factory):
    stg = settings_factory("generic-templates")
    names = {name for name, _ in iter_resource_files("preprocess", "templates")}
    assert names == {
        "single_recon.yml", "paired_single_recon.yml",
        "recon_timelapse_simple.yml", "recon_mltpl_snr.yml",
    }
    for name in names:
        template = load_yaml(find_resource("preprocess", "templates", name))
        assert template["dataset_name"] == "CHANGEME"
        assert "c21" not in repr(template)
        assert "gag_live" not in repr(template)
        cfg = PreprocessConfig.model_validate(template)
        PIPELINES_REGISTRY[cfg.pipeline].parse_cfg(cfg.pipeline_cfg)
        with pytest.raises(ValueError, match="CHANGEME"):
            PreprocessRun.from_cfg(template, paths=Paths(stg))
        with pytest.raises(ValueError, match="cannot be run directly"):
            resolve_config_path(f"templates/{name}", resolver=ConfigPathResolver("preprocess", stg=stg))


def test_preprocess_create_and_ownership_in_isolated_roots(settings_factory):
    a = Settings.from_workspace(settings_factory("A").WORKSPACE_ROOT)
    b = Settings.from_workspace(settings_factory("B").WORKSPACE_ROOT)
    original = (a.CONFIGS_ROOT / "preprocess" / "single.yml")
    old_bytes = original.read_bytes() if original.exists() else None
    for stg, dataset in ((a, "first"), (b, "second"), (a, "third")):
        path, warnings = create_preprocess_config_from_template(
            "single_recon", dataset_name=dataset, stg=stg)
        assert path == stg.PREPROCESS_CONFIG_DIR / "local" / f"{dataset}.yml"
        assert load_yaml(path)["dataset_name"] == dataset
        assert path.read_text().splitlines()[0].endswith("../../schema/preprocess.schema.json")
        assert ConfigPathResolver("preprocess", stg=stg).resolve(f"local/{dataset}") == path.resolve()
        assert all("schemas missing" in w for w in warnings)
        with pytest.raises(FileExistsError):
            create_preprocess_config_from_template("single_recon", dataset_name=dataset, stg=stg)
    assert not (a.PREPROCESS_CONFIG_DIR / "local" / "second.yml").exists()
    assert not (b.PREPROCESS_CONFIG_DIR / "local" / "first.yml").exists()
    if old_bytes is not None:
        assert original.read_bytes() == old_bytes


def test_preprocess_new_cli_and_template_guard(settings_factory, capsys):
    stg = settings_factory("new_cli")
    assert config_cli.main(["list", "--domain", "preprocess"], settings=stg) == 0
    assert "built-in" in capsys.readouterr().out
    assert config_cli.main(["new", "preprocess", "paired_single_recon",
                            "--dataset", "pairs"], settings=stg) == 0
    created = stg.PREPROCESS_CONFIG_DIR / "local" / "pairs.yml"
    assert load_yaml(created)["pipeline"] == "paired_single_recon"
    assert "Created preprocess config" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        config_cli.main(["new", "preprocess", "single_recon", "--loss", "MSE"], settings=stg)
    with pytest.raises(SystemExit):
        config_cli.main(["new", "preprocess"], settings=stg)
    untouched, warnings = create_preprocess_config_from_template("recon_mltpl_snr", stg=stg)
    assert load_yaml(untouched)["dataset_name"] == "CHANGEME"
    assert warnings


def test_explicit_inference_seeding_is_non_destructive_and_resolves_layering(settings_factory):
    stg = Settings.from_workspace(settings_factory("inference_d").WORKSPACE_ROOT)
    local = stg.INFERENCE_CONFIG_DIR / "local"
    assert not (local / "defaults.yml").exists()
    assert not (local / "post_training.yml").exists()
    assert resolve_apply_config(stg=stg).inference.lvae_num_samples >= 1
    created = seed_inference_local_configs(stg.CONFIGS_ROOT)
    assert {p.name for p in created} == {"defaults.yml", "post_training.yml"}
    assert seed_inference_local_configs(stg.CONFIGS_ROOT) == []
    assert (local / "defaults.yml").read_text().splitlines()[0].endswith("../../schema/inference-defaults.schema.json")
    assert (local / "post_training.yml").read_text().splitlines()[0].endswith("../../schema/inference.schema.json")
    assert resolve_evaluate_config(config="local/post_training", stg=stg).checkpoint.best_or_last == "both"

    # Settings loader never modifies or creates editable inference files.
    defaults = local / "defaults.yml"
    contents = "apply:\n  inference:\n    lvae_num_samples: 23\n"
    defaults.write_text(contents, encoding="utf-8")
    stg2 = Settings.from_workspace(stg.WORKSPACE_ROOT)
    assert defaults.read_text(encoding="utf-8") == contents
    assert seed_inference_local_configs(stg.CONFIGS_ROOT) == []
    user = local / "chosen.yml"
    save_yaml({"apply": {"inference": {"lvae_num_samples": 41}}}, user)
    model = local / "bundled.yml"
    save_yaml({"apply": {"inference": {"lvae_num_samples": 31}}}, model)
    assert resolve_apply_config(stg=stg2).inference.lvae_num_samples == 23
    assert resolve_apply_config(model_config=model, stg=stg2).inference.lvae_num_samples == 31
    assert resolve_apply_config(model_config=model, config="local/chosen", stg=stg2).inference.lvae_num_samples == 41
    assert resolve_apply_config(model_config=model, config="local/chosen",
                                overrides=ApplyOverrides(inference=ApplyInferenceOverrides(lvae_num_samples=51)),
                                stg=stg2).inference.lvae_num_samples == 51
    assert resolve_apply_config(config="presets/hdn_sup", stg=stg2).inference.lvae_num_samples == 10
    assert resolve_apply_config(config=user.resolve(), stg=stg2).inference.lvae_num_samples == 41


def test_explicit_schema_refresh_uses_selected_config_root_and_registry(settings_factory, capsys):
    source = settings_factory("schemas_D")
    stg = Settings.from_workspace(source.WORKSPACE_ROOT)
    repo_configs = stg.WORKSPACE_ROOT.parent / "external_editable" / "configs"
    repo_configs.mkdir(parents=True)
    stg_external = Settings.from_workspace(stg.WORKSPACE_ROOT, config_root=repo_configs)
    registry_path = Paths(stg).dataset_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml({"distinct_D": {
        "data_format": "single", "usage": "training", "for_training": True,
        "structure": {"recon": ["inp"]},
        "outputs": {"recon": [{"key": "inp", "path": "inp", "role": "inp", "axes": "YX"}]},
        "defaults": {"recon": {"input": "inp", "target": None}},
    }}, registry_path)
    assert not (repo_configs / "schema").exists()
    assert config_cli.main(["schema", "refresh"], settings=stg_external) == 0
    assert "Refreshed 8 editor schemas" in capsys.readouterr().out
    result = list((repo_configs / "schema").glob("*.schema.json"))
    assert len(result) == 8
    assert not (stg.CONFIGS_ROOT / "schema").exists()
    exp_schema = json.loads((repo_configs / "schema" / "experiment.schema.json").read_text())
    assert "distinct_D" in json.dumps(exp_schema)
    assert "recon_mltpl_snr" in (repo_configs / "schema" / "preprocess.schema.json").read_text()
    # Explicit refresh can safely be rerun; it never writes local experiment YAMLs.
    local = repo_configs / "preprocess" / "local" / "foo.yml"
    local.parent.mkdir(parents=True)
    local.write_text("some: user config\n")
    assert len(refresh_user_schemas(stg=stg_external)) == 8
    assert local.read_text() == "some: user config\n"


def test_explicit_legacy_preprocess_import_preserves_sources(settings_factory, capsys):
    stg = settings_factory("preprocess_legacy")
    root = stg.PREPROCESS_CONFIG_DIR
    source = root / "single.yml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes((Path(__file__).resolve().parents[3] / "configs" / "preprocess" / "single.yml").read_bytes())
    original = source.read_bytes()
    copied, skipped = copy_legacy_preprocess_configs(stg=stg)
    assert (root / "local" / "single.yml") in copied
    assert not skipped
    assert source.read_bytes() == original
    target = root / "local" / "single.yml"
    assert target.read_text().startswith("# yaml-language-server: $schema=../../schema/preprocess.schema.json")
    assert load_yaml(source) == load_yaml(target)
    target.write_text("user: kept\n")
    assert config_cli.main(["import-preprocess"], settings=stg) == 0
    assert "kept existing: single.yml" in capsys.readouterr().out
    assert target.read_text() == "user: kept\n"
    assert source.read_bytes() == original


def test_preprocess_generated_config_executes_with_selected_workspace_and_config_root(
    settings_factory, tmp_path,
):
    import numpy as np
    import tifffile
    from lisai.preprocess.cli import run_preprocess_config
    from lisai.preprocess.reporting import NoOpPreprocessReporter

    for label, external in (("normal", False), ("developer", True)):
        legacy = settings_factory("execution_" + label)
        config_root = (tmp_path / "editor_files" / "configs") if external else None
        if config_root:
            config_root.mkdir(parents=True, exist_ok=True)
        stg = Settings.from_workspace(legacy.WORKSPACE_ROOT, config_root=config_root)
        paths = Paths(stg)
        dataset_name = "sample_" + label
        dump = paths.dataset_dump_dir(dataset_name=dataset_name, data_type="recon")
        dump.mkdir(parents=True)
        tifffile.imwrite(dump / "img01.tif", np.arange(64, dtype=np.uint16).reshape(8, 8))
        local_path, _ = create_preprocess_config_from_template(
            "single_recon", dataset_name=dataset_name, stg=stg)
        assert local_path.parent == stg.CONFIGS_ROOT / "preprocess" / "local"
        result = run_preprocess_config(local_path, paths=paths, reporter=NoOpPreprocessReporter())
        assert result.n_files == 1
        assert paths.dataset_registry_path().exists()
        assert paths.dataset_preprocess_dir(dataset_name=dataset_name, data_type="recon").exists()
