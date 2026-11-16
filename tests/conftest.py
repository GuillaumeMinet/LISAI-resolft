"""Temporary, explicit settings for tests; never mutate checkout configuration."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lisai.config.settings import Settings
from lisai.config.io.yaml import save_yaml, load_yaml


@pytest.fixture
def settings_factory(tmp_path):
    """Construct independent Settings trees from copied legacy YAML files."""
    def create(name: str, *, data_root: Path | None = None) -> Settings:
        configs_root = tmp_path / name / "configs"
        configs_root.mkdir(parents=True)
        for filename in ("project_config.yml", "data_config.yml"):
            shutil.copy2(ROOT / "configs" / filename, configs_root / filename)
        workspace = (data_root or tmp_path / name / "data").resolve()
        save_yaml({"infrastructure": {"data_root": str(workspace)}},
                  configs_root / "local_config.yml")
        # M2 workspace global YAMLs exist independently of the user's configs.
        workspace_configs = workspace / "configs"
        workspace_configs.mkdir(parents=True, exist_ok=True)
        project = load_yaml(ROOT / "configs" / "project_config.yml")
        for section in ("roots", "templates"):
            project["paths"][section] = {
                key: value.replace("{data_root}", "{workspace_root}").replace(
                    "{paths.roots.code_dir}/tensorboard", "{workspace_root}/tensorboard")
                if isinstance(value, str) else value
                for key, value in project["paths"][section].items()
            }
        save_yaml(project, workspace_configs / "workspace.yml")
        shutil.copy2(ROOT / "configs" / "data_config.yml", workspace_configs / "data.yml")
        return Settings.from_config_dir(configs_root)
    return create


@pytest.fixture
def settings(settings_factory):
    selected = settings_factory("default")
    # Legacy-default CLI tests need the shipped preset/config directory tree.
    for name in ("training", "preprocess", "inference", "schema"):
        source = ROOT / "configs" / name
        if source.exists():
            shutil.copytree(source, selected.CONFIGS_ROOT / name)
            shutil.copytree(source, selected.WORKSPACE_ROOT / "configs" / name)
    return selected


@pytest.fixture
def legacy_settings(settings):
    return settings


@pytest.fixture
def global_settings(settings):
    return settings

@pytest.fixture(autouse=True)
def _legacy_default_cli_config(settings, monkeypatch):
    """Give legacy boundary-only defaults a temporary config tree per test.

    Unlike the former session fixture, this never writes local_config.yml into
    the checkout. Import-safety is checked separately in a clean subprocess.
    """
    import importlib

    settings_module = importlib.import_module("lisai.config.settings")
    monkeypatch.setattr(settings_module, "_legacy_checkout_configs_root", lambda: settings.CONFIGS_ROOT)
    monkeypatch.setenv("LISAI_CONFIG_ROOT", str(settings.CONFIGS_ROOT))
    monkeypatch.setenv("LISAI_HOME", str(Path(settings.WORKSPACE_ROOT).resolve()))
