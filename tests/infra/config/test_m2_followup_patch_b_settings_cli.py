"""Patch B: developer selection changes must be explicit, previewed, and persistent."""
from __future__ import annotations

from pathlib import Path

import pytest

from lisai import cli
from lisai.config import selection
from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.workspace import initialize_workspace


def _activated(tmp_path: Path, monkeypatch, *, complete: bool = True):
    repo = tmp_path / "checkout"
    local = repo / "configs/local.yml"
    local.parent.mkdir(parents=True)
    w1, w2 = tmp_path / "W1", tmp_path / "W2"
    initialize_workspace(w1, core_only=True)
    initialize_workspace(w2, core_only=True)
    raw = {
        "dev": {"enabled": True},
        "workspace": {"root": str(w1)},
        "console": {"progress_bar": True},
    }
    if complete:
        raw["config"] = {"mode": "repo"}
    save_yaml(raw, local)
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: repo)
    return repo, local, w1, w2


def test_settings_set_persists_modes_and_switching_workspace(tmp_path, monkeypatch, capsys):
    repo, local, w1, w2 = _activated(tmp_path, monkeypatch)
    custom = tmp_path / "my-configs"
    assert cli.main(["settings", "set", "--config-mode", "workspace",
                     "--workspace", str(w2), "--yes"]) == 0
    assert load_yaml(local)["config"] == {"mode": "workspace"}
    loc = selection.resolve_cli_locations(environ={})
    assert loc.workspace_root == w2
    assert loc.experiment_config_root == w2 / "configs"
    assert "Proposed file changes:" in capsys.readouterr().out

    assert cli.main(["settings", "set", "--config-mode", "custom",
                     "--config-root", str(custom), "--yes"]) == 0
    loc = selection.resolve_cli_locations(environ={})
    assert loc.workspace_root == w2
    assert loc.experiment_config_root == custom
    assert load_yaml(local)["console"] == {"progress_bar": True}

    # Workspace switching does not switch the independent custom root.
    assert cli.main(["settings", "set", "--workspace", str(w1), "--yes"]) == 0
    loc = selection.resolve_cli_locations(environ={})
    assert (loc.workspace_root, loc.experiment_config_root) == (w1, custom)

    # Switching back to repository mode removes stale custom root metadata.
    assert cli.main(["settings", "set", "--config-mode", "repo", "--yes"]) == 0
    assert load_yaml(local)["config"] == {"mode": "repo"}
    assert selection.resolve_cli_locations(environ={}).experiment_config_root == repo / "configs"


def test_repair_missing_mode_without_valid_current_selection(tmp_path, monkeypatch, capsys):
    _, local, w1, _ = _activated(tmp_path, monkeypatch, complete=False)
    assert cli.main(["settings", "set", "--config-mode", "repo", "--yes"]) == 0
    assert load_yaml(local)["config"] == {"mode": "repo"}
    output = capsys.readouterr().out
    assert "Invalid or incomplete" in output
    assert str(w1) in output
    assert cli.main(["settings", "show"]) == 0
    assert "Workspace status:   valid" in capsys.readouterr().out


def test_cancel_and_invalid_proposals_do_not_modify_file(tmp_path, monkeypatch, capsys):
    _, local, _, w2 = _activated(tmp_path, monkeypatch)
    initial = local.read_bytes()
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert cli.main(["settings", "set", "--workspace", str(w2)]) == 0
    assert local.read_bytes() == initial
    assert "Cancelled" in capsys.readouterr().out

    invalid_cases = [
        ["--workspace", str(tmp_path / "missing"), "--yes"],
        ["--config-mode", "custom", "--yes"],
        ["--config-root", str(tmp_path / "outside"), "--yes"],
        ["--yes"],
    ]
    for flags in invalid_cases:
        assert cli.main(["settings", "set", *flags]) == 2
        assert local.read_bytes() == initial
    assert not list(local.parent.glob(".local-*.tmp"))


def test_set_is_developer_only_even_if_standalone_home_set(tmp_path, monkeypatch, capsys):
    _, local, w1, _ = _activated(tmp_path, monkeypatch)
    initial = local.read_bytes()
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.setenv("LISAI_HOME", str(w1))
    assert cli.main(["settings", "set", "--config-mode", "repo", "--yes"]) == 2
    assert "verified editable" in capsys.readouterr().err
    assert local.read_bytes() == initial


def test_set_preserves_custom_root_on_workspace_only_and_checks_validation(tmp_path, monkeypatch):
    _, local, w1, w2 = _activated(tmp_path, monkeypatch)
    root = tmp_path / "custom"
    assert cli.main(["settings", "set", "--config-mode", "custom",
                     "--config-root", str(root), "--yes"]) == 0
    old = local.read_bytes()
    assert cli.main(["settings", "set", "--workspace", str(w2), "--yes"]) == 0
    assert load_yaml(local)["config"]["root"] == str(root)
    invalid_workspace_file = w1 / "configs/workspace.yml"
    invalid_workspace_file.write_text("not: a valid workspace\n")
    after = local.read_bytes()
    assert cli.main(["settings", "set", "--workspace", str(w1), "--yes"]) == 2
    assert local.read_bytes() == after
    assert old != after


def test_explicit_workspace_targets_do_not_require_selected_workspace(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.delenv("LISAI_HOME", raising=False)
    target = tmp_path / "independent"
    assert cli.main(["workspace", "init", str(target)]) == 0
    assert (target / "configs/workspace.yml").is_file()
    # Explicit update target does not rely on an active selection.
    assert cli.main(["workspace", "update", str(target), "--yes"]) == 0
    assert (target / "configs/data.yml").is_file()
    # The same command without a target requires a selected workspace.
    assert cli.main(["workspace", "update", "--yes"]) == 2
    assert "No workspace selected" in capsys.readouterr().err


def test_settings_set_detects_edit_during_confirmation(tmp_path, monkeypatch, capsys):
    _, local, _, w2 = _activated(tmp_path, monkeypatch)

    def interposed(_prompt):
        with local.open("a", encoding="utf-8") as handle:
            handle.write("# concurrent manual edit\n")
        return "yes"

    monkeypatch.setattr("builtins.input", interposed)
    assert cli.main(["settings", "set", "--workspace", str(w2)]) == 2
    assert "changed during confirmation" in capsys.readouterr().err
    assert local.read_text().endswith("# concurrent manual edit\n")


def test_noop_set_preserves_comments_and_exact_bytes(tmp_path, monkeypatch, capsys):
    _, local, _, _ = _activated(tmp_path, monkeypatch)
    with local.open("a", encoding="utf-8") as stream:
        stream.write("# user comment\n")
    original = local.read_bytes()
    assert cli.main(["settings", "set", "--config-mode", "repo", "--yes"]) == 0
    assert "No changes" in capsys.readouterr().out
    assert local.read_bytes() == original


def test_relative_cli_paths_are_persisted_as_absolute(tmp_path, monkeypatch):
    _, local, _, _ = _activated(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["settings", "set", "--workspace", "W2",
                     "--config-mode", "custom", "--config-root", "config-area", "--yes"]) == 0
    assert load_yaml(local)["workspace"]["root"] == str(tmp_path / "W2")
    assert load_yaml(local)["config"]["root"] == str(tmp_path / "config-area")
