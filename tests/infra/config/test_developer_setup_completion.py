"""Developer initialization is independent of selection; readiness is shared."""
from __future__ import annotations

from pathlib import Path

import pytest

from lisai import cli
from lisai.config import selection
from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.workspace import initialize_developer, initialize_workspace


def _checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    (checkout / "src/lisai").mkdir(parents=True)
    (checkout / "src/lisai/__init__.py").touch()
    (checkout / "pyproject.toml").write_text('[project]\nname="lisai"\n')
    return checkout


def test_readiness_lifecycle_and_no_implicit_home_fallback(tmp_path, monkeypatch, capsys):
    checkout = _checkout(tmp_path)
    workspace = tmp_path / "standalone"
    initialize_workspace(workspace, core_only=True)
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: checkout)
    monkeypatch.setenv("LISAI_HOME", str(workspace))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("Unexpected setup prompt"))

    before = selection.developer_setup_status(checkout)
    assert (before.enabled, before.complete) == (False, False)
    assert cli.main(["dev", "init"]) == 0
    after = selection.developer_setup_status(checkout)
    assert after.enabled is True
    assert after.complete is False
    assert after.reason == "Missing workspace.root"
    assert load_yaml(checkout / "configs/local.yml") == {
        "dev": {"enabled": True}, "config": {"mode": "repo"},
    }
    assert "lisai settings set --workspace PATH" in capsys.readouterr().out

    with pytest.raises(ValueError, match="Developer setup is incomplete"):
        selection.resolve_cli_locations(environ={"LISAI_HOME": str(workspace)},
                                        editable_source_root=checkout)
    assert cli.main(["datasets", "list"]) == 2
    assert "Developer setup is incomplete" in capsys.readouterr().err
    # Diagnostics and operations that need no workspace remain callable.
    assert cli.main(["settings", "show"]) == 0
    assert "Developer setup is incomplete" in capsys.readouterr().out
    assert cli.main(["workspace", "init", str(tmp_path / "other"), "--core-only"]) == 0

    assert cli.main(["settings", "set", "--workspace", str(workspace), "--yes"]) == 0
    assert selection.developer_setup_status(checkout).complete
    loc = selection.resolve_cli_locations(environ={"LISAI_HOME": str(tmp_path / "ignored")},
                                          editable_source_root=checkout)
    assert loc.workspace_root == workspace
    assert loc.experiment_config_root == checkout / "configs"
    # A subsequent dev init must never alter the selected workspace or its mode.
    local_bytes = (checkout / "configs/local.yml").read_bytes()
    assert initialize_developer(checkout) == []
    assert (checkout / "configs/local.yml").read_bytes() == local_bytes


def test_readiness_requires_both_selection_fields(tmp_path):
    checkout = _checkout(tmp_path)
    path = checkout / "configs/local.yml"
    path.parent.mkdir()
    save_yaml({"dev": {"enabled": True}}, path)
    state = selection.developer_setup_status(checkout)
    assert state.enabled and not state.complete
    assert state.reason == "Missing workspace.root, config.mode"
    with pytest.raises(ValueError, match="settings set"):
        selection.resolve_cli_locations(environ={}, editable_source_root=checkout)
    workspace = tmp_path / "workspace"
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(workspace)}}, path)
    assert selection.developer_setup_status(checkout).reason == "Missing config.mode"
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(workspace)},
               "config": {"mode": "repo"}}, path)
    assert selection.developer_setup_status(checkout).complete


def test_dev_init_keeps_completed_custom_selection(tmp_path):
    checkout = _checkout(tmp_path)
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace, core_only=True)
    local = checkout / "configs/local.yml"
    local.parent.mkdir()
    custom = tmp_path / "experiments"
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(workspace)},
               "config": {"mode": "custom", "root": str(custom)},
               "console": {"progress_bar": True}}, local)
    before = local.read_bytes()
    initialize_developer(checkout)
    assert local.read_bytes() == before
    assert selection.developer_setup_status(checkout).complete
    assert (checkout / "configs/inference/defaults.yml").is_file()
    # An initializer never creates or updates the independently selected root.
    assert not custom.exists()
