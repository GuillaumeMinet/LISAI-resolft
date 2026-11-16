"""M2 Patch B: authoritative workspace YAMLs, independent writable root, isolation."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.models.local_config import LocalConfig
from lisai.config.selection import load_settings_for_selected_locations, resolve_cli_locations
from lisai.config.settings import Settings, load_default_settings
from lisai.infra.paths import Paths
from lisai.runs.lifecycle import stored_run_path


REPO = Path(__file__).resolve().parents[3]


def make_workspace(tmp_path: Path, name: str, *, inference_mode: str = "default") -> Path:
    root = tmp_path / name
    cfg_dir = root / "configs"
    cfg_dir.mkdir(parents=True)
    old = load_yaml(REPO / "configs" / "project_config.yml")
    for section in ("roots", "templates"):
        old["paths"][section] = {
            key: (value.replace("{data_root}", "{workspace_root}")
                        .replace("{paths.roots.code_dir}/tensorboard", "{workspace_root}/tensorboard"))
            if isinstance(value, str) else value
            for key, value in old["paths"][section].items()
        }
    old["inference"] = {"output": {"mode": inference_mode, "save_input_mode": "never"}}
    old["console"] = {"progress_bar": False}
    save_yaml(old, cfg_dir / "workspace.yml")
    save_yaml(load_yaml(REPO / "configs" / "data_config.yml"), cfg_dir / "data.yml")
    return root


def file_bytes(root: Path):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_workspace_paths_and_settings_are_side_effect_free(tmp_path, monkeypatch):
    root = make_workspace(tmp_path, "ws")
    before = file_bytes(root)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: pytest.fail("unexpected input()"))
    stg = Settings.from_workspace(root)
    assert file_bytes(root) == before
    assert stg.WORKSPACE_ROOT == root
    assert stg.CONFIGS_ROOT == root / "configs"
    assert stg.PROJECT_CONFIG_PATH == root / "configs" / "workspace.yml"
    assert stg.DATA_CONFIG_PATH == root / "configs" / "data.yml"
    assert stg.INFERENCE_OUTPUT_MODE == "default"
    assert stg.INFERENCE_SAVE_INPUT_MODE == "never"
    assert stg.LOCAL_PROGRESS_BAR is False
    paths = Paths(stg)
    assert paths.workspace_root() == root
    assert paths.datasets_root() == root / "datasets"
    assert paths.promoted_models_root() == root / "models"
    assert paths.dataset_registry_path() == root / "datasets" / "dataset_registry.yml"
    assert paths.promoted_model_registry_path() == root / "models" / "model_registry.yml"
    assert paths.inference_root() == root / "inference"
    assert paths.tensorboard_dir(dataset_name="vim").is_relative_to(root / "tensorboard")
    assert paths.run_dir(dataset_name="vim", models_subfolder="m", exp_name="experiment").is_relative_to(
        root / "datasets" / "training" / "vim" / "runs"
    )
    relative = stored_run_path(
        paths.run_dir(dataset_name="vim", models_subfolder="m", exp_name="experiment"),
        paths=paths,
    )
    assert relative == "datasets/training/vim/runs/m/experiment"
    assert "data_root" not in stg._ctx
    assert "code_dir" not in stg._ctx.paths.roots
    assert not hasattr(stg, "DATA_ROOT")
    assert not hasattr(paths, "data_root")
    assert "infrastructure" not in LocalConfig.model_fields


def test_two_workspaces_are_isolated_even_with_shared_writable_configs(tmp_path):
    a = make_workspace(tmp_path, "A", inference_mode="in_place")
    b = make_workspace(tmp_path, "B")
    shared = tmp_path / "repo" / "configs"
    shared.mkdir(parents=True)
    save_yaml({"workspace": {"root": str(a)}, "dev": {"enabled": True},
               "console": {"progress_bar": True}}, shared / "local.yml")
    for root, expected_mode in ((a, "in_place"), (b, "default"), (a, "in_place")):
        stg = Settings.from_workspace(root, config_root=shared)
        paths = Paths(stg)
        assert stg.CONFIGS_ROOT == shared
        assert stg.WORKSPACE_ROOT == root
        assert stg.INFERENCE_OUTPUT_MODE == expected_mode
        assert stg.LOCAL_PROGRESS_BAR is True
        assert paths.dataset_registry_path().is_relative_to(root)
        assert paths.promoted_model_registry_path().is_relative_to(root)
        assert paths.inference_output_dir(source_name="d", model_name="m").is_relative_to(root)
        assert paths.run_dir(dataset_name="d", models_subfolder="x", exp_name="e").is_relative_to(root)


def test_authoritative_data_yaml_not_taken_from_editable_root(tmp_path):
    root = make_workspace(tmp_path, "space")
    external = tmp_path / "editable" / "configs"
    external.mkdir(parents=True)
    save_yaml({"format": {"broken": {"raw": "bad"}}, "data_types": {}}, external / "data.yml")
    settings = Settings.from_workspace(root, config_root=external)
    assert "single" in settings.data_cfg.format
    assert settings.get_data_filename("single", "recon", sample_id="c01") == Path("c01.tif")


def test_developer_local_prefs_only_authored_fields_override(tmp_path):
    root = make_workspace(tmp_path, "space", inference_mode="folder_inside")
    external = tmp_path / "repo" / "configs"
    external.mkdir(parents=True)
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(root)},
               "inference": {"output": {"mode": "in_place"}},
               "console": {"progress_bar": True}}, external / "local.yml")
    stg = Settings.from_workspace(root, config_root=external)
    assert stg.INFERENCE_OUTPUT_MODE == "in_place"
    assert stg.INFERENCE_SAVE_INPUT_MODE == "never"  # inherited, NOT Pydantic default
    assert stg.LOCAL_PROGRESS_BAR is True
    custom = tmp_path / "custom_predictions"
    save_yaml({"inference": {"inference_dir": str(custom)}}, external / "local.yml")
    stg = Settings.from_workspace(root, config_root=external)
    assert stg.INFERENCE_OUTPUT_MODE == "folder_inside"
    assert stg.INFERENCE_SAVE_INPUT_MODE == "never"
    assert Paths(stg).inference_root() == custom
    assert stg.LOCAL_PROGRESS_BAR is False  # restored workspace baseline


@pytest.mark.parametrize("bad", [
    {"dev": {"enabled": "true"}},
    {"queue": {"paused": True}},
    {"data": {"format": {}}},
    {"paths": {"roots": {"data_dir": "/wrong"}}},
    {"inference": {"output": {"unsupported": "oops"}}},
    {"console": {"unsupported": False}},
    {"workspace": {"root": "relative/path"}},
    {"infrastructure": {"data_root": "/legacy"}},
])
def test_local_unknown_keys_and_invalid_roots_rejected(tmp_path, bad):
    root = make_workspace(tmp_path, "space")
    ext = tmp_path / "repo" / "configs"
    ext.mkdir(parents=True)
    save_yaml(bad, ext / "local.yml")
    with pytest.raises(ValidationError):
        Settings.from_workspace(root, config_root=ext)


def test_default_python_api_does_not_activate_external_configs(tmp_path, monkeypatch):
    root = make_workspace(tmp_path, "workspace")
    external = tmp_path / "repo" / "configs"
    external.mkdir(parents=True)
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(tmp_path / "wrong")},
               "inference": {"output": {"mode": "in_place"}}}, external / "local.yml")
    monkeypatch.setenv("LISAI_HOME", str(root))
    monkeypatch.setenv("LISAI_CONFIG_ROOT", str(external))
    stg = load_default_settings()
    assert stg.WORKSPACE_ROOT == root
    assert stg.CONFIGS_ROOT == root / "configs"
    assert stg.INFERENCE_OUTPUT_MODE == "default"


def test_selected_cli_location_two_roots_load_and_errors(tmp_path):
    root = make_workspace(tmp_path, "workspace")
    external = tmp_path / "repo" / "configs"
    external.mkdir(parents=True)
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(tmp_path / "wrong")},
               "inference": {"output": {"mode": "in_place"}}}, external / "local.yml")
    locations = resolve_cli_locations(
        workspace=root, config_root=external, environ={}, editable_source_root=None)
    stg = load_settings_for_selected_locations(locations)
    assert stg.WORKSPACE_ROOT == root
    assert stg.CONFIGS_ROOT == external
    assert stg.INFERENCE_OUTPUT_MODE == "in_place"
    absent = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="workspace.yml.*lisai workspace init"):
        Settings.from_workspace(absent)
    assert not absent.exists()


def test_legacy_loader_translates_templates_read_only(settings_factory):
    stg = settings_factory("legacy")
    before = file_bytes(stg.CONFIGS_ROOT)
    assert stg.WORKSPACE_ROOT.is_absolute()
    assert Paths(stg).datasets_root() == stg.WORKSPACE_ROOT / "datasets"
    assert "{workspace_root}" in stg.project_cfg.paths.roots["data_dir"]
    assert "{data_root}" not in stg.project_cfg.paths.roots["data_dir"]
    assert stg._ctx.get("data_root") is None
    assert file_bytes(stg.CONFIGS_ROOT) == before


def test_workspace_paths_match_explicit_legacy_at_same_root(settings_factory):
    legacy = settings_factory("same_paths")
    workspace = Settings.from_workspace(legacy.WORKSPACE_ROOT)
    left, right = Paths(legacy), Paths(workspace)
    assert left.dataset_registry_path() == right.dataset_registry_path()
    assert left.promoted_model_registry_path() == right.promoted_model_registry_path()
    assert left.run_dir(dataset_name="d", models_subfolder="m", exp_name="e") == (
        right.run_dir(dataset_name="d", models_subfolder="m", exp_name="e")
    )
    assert left.tensorboard_dir(dataset_name="d") == right.tensorboard_dir(dataset_name="d")
    assert left.noise_model_path(noiseModel_name="n") == right.noise_model_path(noiseModel_name="n")


def test_release_builder_workspace_parameter_and_script_flag():
    import inspect
    from lisai.data.release_builder import build_dataset_catalog
    from scripts.release.build_dataset_catalog import _parser
    assert "workspace_root" in inspect.signature(build_dataset_catalog).parameters
    assert "data_root" not in inspect.signature(build_dataset_catalog).parameters
    assert _parser().parse_args(["/tmp/ws", "/tmp/staged"]).workspace_root == "/tmp/ws"
