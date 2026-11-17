"""Patch D: ordinary post-training YAML and truthful editor schema links."""
from __future__ import annotations

from pathlib import Path

import pytest

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.models.inference.presets import POST_TRAINING_OVERRIDES
from lisai.config.settings import Settings
from lisai.config.user_configs import (
    inference_seed_files, refresh_user_schemas, seed_inference_local_configs,
)
from lisai.config.workspace import update_workspace
from lisai.evaluation.defaults import resolve_apply_config, resolve_evaluate_config


def _schema_target(yaml_path: Path) -> Path:
    line = yaml_path.read_text(encoding="utf-8").splitlines()[0]
    assert line.startswith("# yaml-language-server: $schema=")
    return (yaml_path.parent / line.split("$schema=", 1)[1]).resolve()


def test_external_root_inference_seed_links_and_post_training_semantics(settings_factory, tmp_path):
    work = settings_factory("d_seed").WORKSPACE_ROOT
    external = tmp_path / "experiment-configs"
    stg = Settings.from_workspace(work, config_root=external)
    assert not external.exists()
    # Reading a workspace never provisions the external config root.
    assert resolve_apply_config(stg=stg).inference.lvae_num_samples >= 1
    assert not external.exists()
    with pytest.raises(FileNotFoundError, match="post_training"):
        resolve_evaluate_config(config="post_training", stg=stg)

    previews = inference_seed_files(external)
    assert {p.name for p in previews} == {"defaults.yml", "post_training.yml"}
    assert not external.exists()
    created = seed_inference_local_configs(external)
    assert set(created) == set(previews)
    for path in created:
        assert path.read_text(encoding="utf-8") == previews[path]
        schema = "inference-defaults.schema.json" if path.name == "defaults.yml" else "inference.schema.json"
        assert _schema_target(path) == external / "schema" / schema
    # Schema refresh is explicit; its source is the selected workspace.
    assert not (external / "schema").exists()
    refresh_user_schemas(stg=stg)
    assert all(_schema_target(path).is_file() for path in created)
    assert load_yaml(external / "inference" / "post_training.yml") == POST_TRAINING_OVERRIDES
    seeded = resolve_evaluate_config(config="post_training", stg=stg)
    assert seeded.checkpoint.best_or_last == "both"
    assert seeded.metrics == ["psnr", "ssim"]
    assert seeded.saving.overwrite is True

    post = external / "inference" / "post_training.yml"
    save_yaml({"evaluate": {"metrics": ["new_metric"]}}, post)
    assert seed_inference_local_configs(external) == []
    custom = resolve_evaluate_config(config="post_training", stg=stg)
    assert custom.metrics == ["new_metric"]
    assert custom.checkpoint.best_or_last == "best"
    assert custom.saving.overwrite is False
    assert load_yaml(post) == {"evaluate": {"metrics": ["new_metric"]}}


def test_update_inference_generation_matches_initial_seed(settings_factory, tmp_path):
    work = settings_factory("d_update").WORKSPACE_ROOT
    external = tmp_path / "external-editable"
    expected = inference_seed_files(external)
    # Patch E will add explicit preview/confirmation; here test only that
    # the current update helper uses the same schema-aware seed source.
    update_workspace(work, config_root=external, replace_editable_defaults=True)
    assert {p: p.read_text(encoding="utf-8") for p in expected} == expected
    for path in expected:
        assert _schema_target(path) == external / "schema" / (
            "inference-defaults.schema.json" if path.name == "defaults.yml"
            else "inference.schema.json"
        )


def test_automatic_post_training_uses_selected_editable_config(settings_factory, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lisai.infra.paths import Paths
    from lisai.training.orchestration import post_training as orchestration

    work = settings_factory("auto_post_D").WORKSPACE_ROOT
    stg = Settings.from_workspace(work)
    seed_inference_local_configs(stg.CONFIGS_ROOT)
    post = stg.INFERENCE_CONFIG_DIR / "post_training.yml"
    save_yaml({"evaluate": {"checkpoint": {"best_or_last": "last"},
                            "metrics": ["special_metric"]}}, post)
    captured = {}
    monkeypatch.setattr(orchestration, "run_evaluate", lambda **kw: captured.update(kw))
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(post_training_inference=True),
        data=SimpleNamespace(dataset_name="vimentin"),
        routing=SimpleNamespace(models_subfolder="denoise"),
        training=SimpleNamespace(progress_bar=False),
    )
    run_dir = tmp_path / "runs" / "run_01"
    runtime = SimpleNamespace(run_dir=run_dir, paths=Paths(stg))
    outcome = SimpleNamespace(reason="completed", last_completed_epoch=1)
    orchestration.run_post_training_evaluation(cfg, runtime, outcome)
    assert captured["cfg"].checkpoint.best_or_last == "last"
    assert captured["cfg"].metrics == ["special_metric"]
    assert captured["cfg"].saving.overwrite is False
    assert captured["dataset_name"] == "vimentin"
    assert captured["model_name"] == "run_01"
    post.unlink()
    with pytest.raises(FileNotFoundError, match="post_training"):
        orchestration.run_post_training_evaluation(cfg, runtime, outcome)


def test_developer_local_schema_hint_survives_settings_edit(settings_factory, tmp_path, monkeypatch):
    from argparse import Namespace
    from lisai.config import settings_cli
    from lisai.config.workspace import initialize_developer

    work = settings_factory("developer_schema_D").WORKSPACE_ROOT
    checkout = tmp_path / "checkout"
    (checkout / "src/lisai").mkdir(parents=True)
    (checkout / "src/lisai/__init__.py").touch()
    (checkout / "pyproject.toml").write_text('[project]\nname="lisai"\n')
    initialize_developer(checkout, work)
    local = checkout / "configs/local.yml"
    assert _schema_target(local) == checkout / "configs/schema/local-config.schema.json"
    assert _schema_target(local).is_file()
    from lisai.config import selection
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: checkout)
    args = Namespace(workspace=None, config_mode="workspace", config_root=None, yes=True)
    assert settings_cli.set_developer_settings(args) == 0
    assert _schema_target(local) == checkout / "configs/schema/local-config.schema.json"
    assert load_yaml(local)["config"]["mode"] == "workspace"
