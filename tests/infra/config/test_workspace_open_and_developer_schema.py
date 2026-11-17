"""Small M2 follow-up: open selected workspace and refresh developer editor schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai import cli
from lisai.config.cli import run_schema_refresh_from_args
from lisai.config.settings import Settings
from lisai.config.workspace import open_workspace


def test_workspace_open_is_a_workspace_subcommand():
    args = cli.build_parser().parse_args(["workspace", "open"])
    assert args.workspace_action == "open"


def test_workspace_open_uses_selected_root_without_loading_settings(tmp_path, monkeypatch, capsys):
    root = tmp_path / "selected"
    root.mkdir()
    monkeypatch.setattr(cli, "resolve_cli_locations", lambda **kwargs: SimpleNamespace(workspace_root=root))
    from lisai.config import workspace
    opened = []
    monkeypatch.setattr(workspace, "open_workspace", lambda path: (opened.append(path), path)[1])
    monkeypatch.setattr(cli, "load_settings_for_selected_locations", lambda *a, **kw: pytest.fail("Settings loaded"))
    assert cli.main(["workspace", "open"]) == 0
    assert opened == [root]
    assert f"Opened workspace: {root}" in capsys.readouterr().out


def test_workspace_open_requires_selection(monkeypatch, capsys):
    monkeypatch.setattr(cli, "resolve_cli_locations", lambda **kwargs: SimpleNamespace(workspace_root=None))
    assert cli.main(["workspace", "open"]) == 2
    assert "No workspace selected" in capsys.readouterr().err


def test_workspace_open_windows_uses_native_explorer(tmp_path, monkeypatch):
    from lisai.config import workspace
    root = tmp_path / "exists"
    root.mkdir()
    started = []
    monkeypatch.setattr(workspace.sys, "platform", "win32")
    monkeypatch.setattr(workspace.os, "startfile", lambda value: started.append(value), raising=False)
    assert open_workspace(root) == root
    assert started == [str(root)]


def test_workspace_open_requires_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_workspace(tmp_path / "missing")


def test_schema_refresh_updates_checkout_selector_in_custom_mode(settings_factory, tmp_path):
    workspace = settings_factory("schema_open").WORKSPACE_ROOT
    external = tmp_path / "custom"
    settings = Settings.from_workspace(workspace, config_root=external)
    repo_config = tmp_path / "repo" / "configs"
    repo_config.mkdir(parents=True)
    old_schema = repo_config / "schema/local-config.schema.json"
    old_schema.parent.mkdir(parents=True)
    old_schema.write_text('{"properties": {"dev": {}}}', encoding="utf-8")
    local_file = repo_config / "local.yml"
    local_file.write_text('dev:\n  enabled: true\n', encoding="utf-8")
    args = argparse.Namespace(_lisai_locations=SimpleNamespace(
        developer_mode=True, local_config_path=local_file,
    ))
    assert run_schema_refresh_from_args(args, argparse.ArgumentParser(), settings=settings) == 0
    schema = json.loads(old_schema.read_text(encoding="utf-8"))
    assert "config" in schema["properties"]
    assert "workspace" in schema["properties"]
    assert (external / "schema/experiment.schema.json").exists()


def test_schema_refresh_standalone_does_not_write_repo_selector(settings_factory, tmp_path):
    settings = settings_factory("schema_standalone")
    args = argparse.Namespace(_lisai_locations=SimpleNamespace(
        developer_mode=False, local_config_path=None,
    ))
    assert run_schema_refresh_from_args(args, argparse.ArgumentParser(), settings=settings) == 0
    assert (settings.CONFIGS_ROOT / "schema/experiment.schema.json").exists()
