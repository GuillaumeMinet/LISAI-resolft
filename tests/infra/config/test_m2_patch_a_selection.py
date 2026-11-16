"""M2 A: selection is independent, deterministic and side-effect-free."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lisai import cli as root_cli
from lisai.config import selection
from lisai_runner import cli as runner_cli


def _local(folder: Path, *, dev: bool | None = None, workspace: Path | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    if dev is not None:
        lines.extend(["dev:", f"  enabled: {str(dev).lower()}"])
    if workspace is not None:
        lines.extend(["workspace:", f"  root: '{workspace}'"])
    target = folder / "local.yml"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def test_standalone_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(selection.Path, "home", lambda: tmp_path / "home")
    chosen = selection.resolve_cli_locations(environ={}, editable_source_root=None)
    assert chosen.workspace_root == tmp_path / "home" / "LISAI"
    assert chosen.experiment_config_root == chosen.workspace_root / "configs"
    assert chosen.local_config_path is None
    assert chosen.workspace_origin == "~/LISAI"
    assert chosen.config_root_origin == "workspace default"
    assert not chosen.workspace_root.exists()


def test_workspace_env_and_cli_precedence(tmp_path):
    home, override = tmp_path / "env", tmp_path / "override"
    by_env = selection.resolve_cli_locations(environ={"LISAI_HOME": str(home)}, editable_source_root=None)
    assert by_env.workspace_root == home
    assert by_env.experiment_config_root == home / "configs"
    by_cli = selection.resolve_cli_locations(
        environ={"LISAI_HOME": str(home)}, workspace=override, editable_source_root=None,
    )
    assert by_cli.workspace_root == override
    assert by_cli.experiment_config_root == override / "configs"
    assert by_cli.workspace_origin == "--workspace"


def test_developer_auto_activation_and_workspace_override(tmp_path):
    repo = tmp_path / "repo"
    workspace_a, workspace_b = tmp_path / "A", tmp_path / "B"
    _local(repo / "configs", dev=True, workspace=workspace_a)
    baseline = selection.resolve_cli_locations(environ={}, editable_source_root=repo)
    assert baseline.developer_mode
    assert baseline.workspace_root == workspace_a
    assert baseline.experiment_config_root == repo / "configs"
    assert baseline.workspace_origin.endswith("workspace.root")
    assert baseline.config_root_origin == "developer activation"

    selected = selection.resolve_cli_locations(
        workspace=workspace_b, environ={"LISAI_HOME": str(tmp_path / "env")}, editable_source_root=repo,
    )
    assert selected.workspace_root == workspace_b
    assert selected.experiment_config_root == repo / "configs"
    assert selected.local_config_path == repo / "configs" / "local.yml"

    from_workspace = selection.resolve_cli_locations(
        workspace=workspace_b, config_root=workspace_b / "configs",
        editable_source_root=repo, environ={},
    )
    assert from_workspace.experiment_config_root == workspace_b / "configs"
    assert not from_workspace.developer_mode


def test_dev_disabled_and_explicit_root_still_allowed(tmp_path):
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    _local(repo / "configs", dev=False, workspace=workspace)
    inactive = selection.resolve_cli_locations(environ={}, editable_source_root=repo)
    assert inactive.workspace_root == Path.home() / "LISAI"
    assert inactive.experiment_config_root == inactive.workspace_root / "configs"
    assert not inactive.developer_mode
    explicitly = selection.resolve_cli_locations(config_root=repo / "configs", environ={}, editable_source_root=repo)
    assert explicitly.workspace_root == workspace
    assert explicitly.experiment_config_root == repo / "configs"
    assert not explicitly.developer_mode


def test_config_root_env_survives_workspace_override(tmp_path):
    external = tmp_path / "external"
    _local(external, workspace=tmp_path / "from-local")
    chosen = selection.resolve_cli_locations(
        workspace=tmp_path / "new-workspace",
        environ={"LISAI_CONFIG_ROOT": str(external), "LISAI_HOME": str(tmp_path / "env-workspace")},
        editable_source_root=None,
    )
    assert chosen.workspace_root == tmp_path / "new-workspace"
    assert chosen.experiment_config_root == external
    assert chosen.config_root_origin == "LISAI_CONFIG_ROOT"


def test_explicit_config_root_beats_env_and_developer(tmp_path):
    repo, external, overridden = tmp_path / "repo", tmp_path / "envroot", tmp_path / "cli-root"
    _local(repo / "configs", dev=True, workspace=tmp_path / "repo-space")
    _local(external, workspace=tmp_path / "env-space")
    _local(overridden, workspace=tmp_path / "chosen-space")
    loc = selection.resolve_cli_locations(
        config_root=overridden, environ={"LISAI_CONFIG_ROOT": str(external)}, editable_source_root=repo,
    )
    assert loc.experiment_config_root == overridden
    assert loc.workspace_root == tmp_path / "chosen-space"
    assert loc.config_root_origin == "--config-root"


def test_relative_flags_use_cwd_but_local_root_must_be_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected = selection.resolve_cli_locations(
        workspace="relative", config_root="relative/configs", environ={}, editable_source_root=None,
    )
    assert selected.workspace_root == tmp_path / "relative"
    assert selected.experiment_config_root == tmp_path / "relative" / "configs"
    p = tmp_path / "bad" / "configs"
    p.mkdir(parents=True)
    (p / "local.yml").write_text("workspace:\n  root: relative/path\n")
    with pytest.raises(ValueError, match="absolute"):
        selection.resolve_cli_locations(config_root=p, environ={}, editable_source_root=None)
    with pytest.raises(ValueError, match="LISAI_HOME must be an absolute"):
        selection.resolve_cli_locations(environ={"LISAI_HOME": "relative"}, editable_source_root=None)


@pytest.mark.parametrize("contents,match", [
    ("dev: invalid\n", "mappings"),
    ("dev:\n  enabled: 'true'\n", "boolean"),
    ("[]\n", "Expected mapping"),
    ("workspace:\n  root: ''\n", "nonempty"),
])
def test_invalid_local_selector_rejected(tmp_path, contents, match):
    root = tmp_path / "configs"
    root.mkdir()
    (root / "local.yml").write_text(contents)
    with pytest.raises(ValueError, match=match):
        selection.resolve_cli_locations(config_root=root, environ={}, editable_source_root=None)


def test_wheel_cannot_auto_activate_even_if_git_parent(tmp_path, monkeypatch):
    repo = tmp_path / "fake_repo"
    (repo / "src" / "lisai").mkdir(parents=True)
    (repo / "src" / "lisai" / "__init__.py").touch()
    (repo / "pyproject.toml").write_text('[project]\nname = "lisai"\n')
    import subprocess
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    import lisai
    monkeypatch.setattr(lisai, "__file__", str(repo / "src" / "lisai" / "__init__.py"))
    monkeypatch.setattr(selection.metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda _: None))
    assert selection._source_checkout_for_import() is None


def test_editable_source_validation_with_metadata(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src" / "lisai").mkdir(parents=True)
    (repo / "src" / "lisai" / "__init__.py").touch()
    (repo / "pyproject.toml").write_text('[project]\nname = "lisai"\n')
    import subprocess
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    import lisai
    monkeypatch.setattr(lisai, "__file__", str(repo / "src" / "lisai" / "__init__.py"))
    direct_url = json.dumps({"url": repo.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(selection.metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda _: direct_url))
    assert selection._source_checkout_for_import() == repo


def test_config_show_without_workspace_is_read_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.delenv("LISAI_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("LISAI_HOME", raising=False)
    workspace = tmp_path / "absent"
    assert root_cli.main(["--workspace", str(workspace), "config", "show"]) == 0
    out = capsys.readouterr().out
    assert "not initialized" in out
    assert str(workspace / "configs" / "workspace.yml") in out
    assert "[--workspace]" in out
    assert not workspace.exists()


def test_selected_old_settings_must_match_workspace(settings_factory, tmp_path):
    settings = settings_factory("correct", data_root=tmp_path / "A")
    loc = selection.resolve_cli_locations(
        workspace=tmp_path / "B", config_root=settings.CONFIGS_ROOT,
        environ={}, editable_source_root=None,
    )
    with pytest.raises(ValueError, match="does not match legacy data_root"):
        selection.load_legacy_settings_for_selected_locations(loc)


def test_selected_legacy_settings_can_dispatch_once(settings_factory, tmp_path, monkeypatch):
    settings = settings_factory("correct", data_root=tmp_path / "data")
    seen = []
    from lisai.training import cli as train_cli
    monkeypatch.setattr(train_cli, "run_from_args", lambda args, parser, *, settings: seen.append(settings) or 0)
    # CLI parser closure resolves this function after parse, rather than at import.
    result = root_cli.main([
        "--workspace", str(tmp_path / "data"), "--config-root", str(settings.CONFIGS_ROOT),
        "train", "something.yml",
    ])
    assert result == 0
    assert len(seen) == 1 and seen[0].CONFIGS_ROOT == settings.CONFIGS_ROOT


def test_runner_flags_and_help_are_inert(tmp_path, monkeypatch):
    parser = runner_cli.build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path / "ws"), "--config-root", str(tmp_path / "cfg"), "queue", "list"])
    assert args.workspace == str(tmp_path / "ws")
    assert args.config_root == str(tmp_path / "cfg")
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    assert not (tmp_path / "ws").exists()


def test_runner_hands_selected_paths_to_queue_handler(tmp_path, monkeypatch):
    from lisai_runner.queue import cli as queue_cli
    workspace = tmp_path / "runner_space"
    config_root = tmp_path / "runner_configs"
    expected = object()
    observed = {}
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.setattr(runner_cli, "load_legacy_settings_for_selected_locations", lambda loc: loc)
    monkeypatch.setattr(runner_cli, "Paths", lambda loc: (observed.setdefault("locations", loc), expected)[1])
    monkeypatch.setattr(queue_cli, "list_jobs", lambda **kwargs: observed.update(kwargs) or 0)
    rc = runner_cli.main([
        "--workspace", str(workspace), "--config-root", str(config_root), "queue", "list",
    ])
    assert rc == 0
    assert observed["paths"] is expected
    assert observed["locations"].workspace_root == workspace
    assert observed["locations"].experiment_config_root == config_root


def test_catalog_is_workspace_independent_even_if_external_selector_is_bad(monkeypatch):
    from lisai.promoted_models import cli as models_cli
    monkeypatch.setattr(models_cli, "run_catalog_from_args", lambda args, parser: 42)
    monkeypatch.setenv("LISAI_CONFIG_ROOT", "relative_bad_path")
    assert root_cli.main(["models", "catalog"]) == 42


def test_source_only_verified_checkout_fallback(monkeypatch):
    import lisai
    from importlib.metadata import PackageNotFoundError

    def not_installed(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(selection.metadata, "distribution", not_installed)
    candidate = selection._source_checkout_for_import()
    assert candidate is not None
    assert (candidate / "src" / "lisai" / "__init__.py").resolve() == Path(lisai.__file__).resolve()
