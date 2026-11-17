"""M2 follow-up Patch A: persistent selection and explicit preference ownership."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from lisai import cli as root_cli
from lisai.config import selection
from lisai.config.io.yaml import save_yaml
from lisai.config.models.local_config import DeveloperLocalConfig
from lisai.config.settings import Settings, load_default_settings
from lisai_runner import cli as runner_cli


def _local(repo: Path, workspace: Path, *, mode: str = "repo", custom_root: Path | None = None,
           enabled: bool = True, preferences: dict | None = None) -> Path:
    path = repo / "configs/local.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"dev": {"enabled": enabled}, "workspace": {"root": str(workspace)},
            "config": {"mode": mode}}
    if custom_root is not None:
        data["config"]["root"] = str(custom_root)
    data.update(preferences or {})
    save_yaml(data, path)
    return path


def test_standalone_requires_home_but_can_be_inspected_without_selection(tmp_path):
    with pytest.raises(ValueError, match="LISAI_HOME"):
        selection.resolve_cli_locations(environ={}, editable_source_root=None)
    loc = selection.resolve_cli_locations(environ={}, editable_source_root=None,
                                          require_workspace=False)
    assert loc.workspace_root is None
    assert loc.experiment_config_root is None
    assert loc.local_config_path is None
    assert not loc.developer_mode
    assert "no workspace selected" in selection.describe_locations(loc)


def test_standalone_home_owns_config_root_and_ignores_external_local(tmp_path):
    ws = tmp_path / "workspace"
    _local(ws, tmp_path / "irrelevant", preferences={"console": {"progress_bar": True}})
    external = tmp_path / "other_configs"
    loc = selection.resolve_cli_locations(environ={
        "LISAI_HOME": str(ws), "LISAI_CONFIG_ROOT": str(external)
    }, editable_source_root=None)
    assert loc.workspace_root == ws
    assert loc.experiment_config_root == ws / "configs"
    assert loc.local_config_path is None
    assert loc.config_mode is None
    assert loc.config_root_origin == "workspace default"
    assert loc.workspace_origin == "LISAI_HOME"
    with pytest.raises(ValueError, match="LISAI_HOME must be an absolute"):
        selection.resolve_cli_locations(environ={"LISAI_HOME": "relative"},
                                        editable_source_root=None)


@pytest.mark.parametrize("mode", ["repo", "workspace", "custom"])
def test_developer_three_modes_and_env_noninterference(tmp_path, mode):
    repo, a, b, custom = (tmp_path / part for part in ("checkout", "A", "B", "custom"))
    local_path = _local(repo, a, mode=mode, custom_root=custom if mode == "custom" else None)
    expected = {"repo": repo / "configs", "workspace": a / "configs", "custom": custom}[mode]
    loc = selection.resolve_cli_locations(environ={
        "LISAI_HOME": str(b), "LISAI_CONFIG_ROOT": "relative_invalid_ignored"
    }, editable_source_root=repo)
    assert loc.workspace_root == a
    assert loc.experiment_config_root == expected
    assert loc.local_config_path == local_path
    assert loc.config_mode == mode
    assert loc.developer_mode
    assert mode in loc.config_root_origin
    assert str(local_path) in loc.workspace_origin
    assert f"Config mode:        {mode}" in selection.describe_locations(loc)

    # A persistent workspace edit respects the stored ownership mode.
    _local(repo, b, mode=mode, custom_root=custom if mode == "custom" else None)
    changed = selection.resolve_cli_locations(environ={}, editable_source_root=repo)
    assert changed.workspace_root == b
    assert changed.experiment_config_root == (b / "configs" if mode == "workspace" else expected)


def test_disabled_developer_file_does_not_select_workspace(tmp_path):
    repo, selected, env_root = (tmp_path / part for part in ("repo", "local_selection", "env"))
    _local(repo, selected, enabled=False)
    loc = selection.resolve_cli_locations(environ={"LISAI_HOME": str(env_root)},
                                          editable_source_root=repo)
    assert not loc.developer_mode
    assert loc.workspace_root == env_root
    assert loc.local_config_path is None


@pytest.mark.parametrize("selection_data,match", [
    ({"dev": {"enabled": True}, "workspace": {"root": "/tmp/valid"}}, "config.mode"),
    ({"dev": {"enabled": True}, "config": {"mode": "repo"}}, "workspace.root"),
    ({"dev": {"enabled": True}, "workspace": {"root": "relative"},
      "config": {"mode": "repo"}}, "absolute"),
    ({"dev": {"enabled": True}, "workspace": {"root": "/tmp/valid"},
      "config": {"mode": "invalid"}}, "repo.*workspace.*custom"),
    ({"dev": {"enabled": True}, "workspace": {"root": "/tmp/valid"},
      "config": {"mode": "custom"}}, "config.root"),
    ({"dev": {"enabled": True}, "workspace": {"root": "/tmp/valid"},
      "config": {"mode": "repo", "root": "/tmp/extra"}}, "only valid"),
    ({"dev": {"enabled": True}, "workspace": {"root": "/tmp/valid"},
      "config": {"mode": "custom", "root": "relative"}}, "absolute"),
    ({"dev": {"enabled": "true"}}, "boolean"),
    ({"dev": "bad"}, "mapping"),
    ({"dev": {"enabled": True}, "workspace": {"root": "/tmp/valid"},
      "config": {"mode": "repo"}, "queue": {"override": True}}, "extra_forbidden"),
])
def test_invalid_developer_selection_fails_precisely(tmp_path, selection_data, match):
    repo = tmp_path / "repo"
    target = repo / "configs/local.yml"
    target.parent.mkdir(parents=True)
    save_yaml(selection_data, target)
    with pytest.raises((ValidationError, ValueError), match=match):
        selection.resolve_cli_locations(environ={"LISAI_HOME": str(tmp_path / "fallback")},
                                        editable_source_root=repo)


def test_invalid_disabled_local_does_not_override_standalone(tmp_path):
    repo = tmp_path / "repo"
    target = repo / "configs/local.yml"
    target.parent.mkdir(parents=True)
    save_yaml({"dev": {"enabled": False}, "invalid": True}, target)
    loc = selection.resolve_cli_locations(environ={"LISAI_HOME": str(tmp_path / "ws")},
                                          editable_source_root=repo)
    assert not loc.developer_mode


def test_python_settings_ignore_workspace_custom_local_unless_explicit(tmp_path):
    from lisai.config.workspace import initialize_workspace

    a, b, custom = (tmp_path / item for item in ("A", "B", "custom"))
    initialize_workspace(a, core_only=True)
    initialize_workspace(b, core_only=True)
    dev = tmp_path / "checkout"
    local = _local(dev, a, mode="custom", custom_root=custom,
                   preferences={"console": {"progress_bar": True},
                                "inference": {"output": {"mode": "in_place"}}})
    _local(custom.parent / "elsewhere", b, preferences={"console": {"progress_bar": False}})
    custom.mkdir(parents=True)
    save_yaml({"console": {"progress_bar": True}}, custom / "local.yml")
    for ws in (a, b, a):
        plain = Settings.from_workspace(ws, config_root=custom)
        assert plain.LOCAL_PROGRESS_BAR is None
        assert plain.INFERENCE_OUTPUT_MODE == "default"
        activated = Settings.from_workspace(ws, config_root=custom, local_config=local)
        assert activated.LOCAL_PROGRESS_BAR is True
        assert activated.INFERENCE_OUTPUT_MODE == "in_place"
        assert activated.WORKSPACE_ROOT == ws
        assert activated.CONFIGS_ROOT == custom
    chosen = selection.resolve_cli_locations(environ={}, editable_source_root=dev)
    settings = selection.load_settings_for_selected_locations(chosen)
    assert settings.LOCAL_PROGRESS_BAR is True


def test_api_default_requires_home_and_explicit_workspace_remains_available(tmp_path, monkeypatch):
    from lisai.config.workspace import initialize_workspace

    monkeypatch.delenv("LISAI_HOME", raising=False)
    monkeypatch.setenv("LISAI_CONFIG_ROOT", "unused_invalid")
    with pytest.raises(ValueError, match="LISAI_HOME is not set"):
        load_default_settings()
    ws = tmp_path / "workspace"
    initialize_workspace(ws, core_only=True)
    assert Settings.from_workspace(ws).WORKSPACE_ROOT == ws
    monkeypatch.setenv("LISAI_HOME", str(ws))
    assert load_default_settings().WORKSPACE_ROOT == ws


def test_diagnostics_without_active_workspace_or_file_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.delenv("LISAI_HOME", raising=False)
    assert root_cli.main(["settings", "show"]) == 0
    out = capsys.readouterr().out
    assert "not selected" in out
    assert "LISAI_HOME (unset)" in out
    assert not (tmp_path / "workspace").exists()


def test_no_transient_cli_root_selection_or_old_config_show():
    parser = root_cli.build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--workspace", "/tmp/W", "settings", "show"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--config-root", "/tmp/R", "train", "x"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["config", "show"])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        runner_cli.build_parser().parse_args(["--workspace", "/tmp/W", "queue", "list"])
    assert caught.value.code == 2
    for parser in (root_cli.build_parser(), runner_cli.build_parser()):
        with pytest.raises(SystemExit) as caught:
            parser.parse_args(["--help"])
        assert caught.value.code == 0


def test_cli_dispatch_uses_selected_workspace_once(tmp_path, monkeypatch):
    from lisai.config.workspace import initialize_workspace
    from lisai.training import cli as train_cli

    ws = tmp_path / "workspace"
    initialize_workspace(ws, core_only=True)
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.setenv("LISAI_HOME", str(ws))
    seen = []
    monkeypatch.setattr(train_cli, "run_from_args",
                        lambda args, parser, *, settings: seen.append(settings) or 0)
    assert root_cli.main(["train", "any.yml"]) == 0
    assert len(seen) == 1
    assert seen[0].WORKSPACE_ROOT == ws
    assert seen[0].CONFIGS_ROOT == ws / "configs"


def test_runner_uses_same_selection_and_has_inert_help(tmp_path, monkeypatch):
    from lisai_runner.queue import cli as queue_cli

    ws = tmp_path / "runner_workspace"
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.setenv("LISAI_HOME", str(ws))
    observed = {}
    monkeypatch.setattr(runner_cli, "load_settings_for_selected_locations", lambda loc: loc)
    monkeypatch.setattr(runner_cli, "Paths", lambda loc: observed.setdefault("selection", loc))
    monkeypatch.setattr(queue_cli, "list_jobs", lambda **kwargs: observed.update(kwargs) or 0)
    assert runner_cli.main(["queue", "list"]) == 0
    assert observed["selection"].workspace_root == ws
    assert observed["selection"].experiment_config_root == ws / "configs"


def test_wheel_cannot_auto_activate_even_if_git_parent(tmp_path, monkeypatch):
    repo = tmp_path / "fake_repo"
    (repo / "src" / "lisai").mkdir(parents=True)
    (repo / "src" / "lisai" / "__init__.py").touch()
    (repo / "pyproject.toml").write_text('[project]\nname = "lisai"\n')
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
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    import lisai
    monkeypatch.setattr(lisai, "__file__", str(repo / "src" / "lisai" / "__init__.py"))
    direct_url = json.dumps({"url": repo.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(selection.metadata, "distribution", lambda name: SimpleNamespace(read_text=lambda _: direct_url))
    assert selection._source_checkout_for_import() == repo


def test_source_only_verified_checkout_fallback(monkeypatch):
    import lisai
    from importlib.metadata import PackageNotFoundError

    def not_installed(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(selection.metadata, "distribution", not_installed)
    candidate = selection._source_checkout_for_import()
    assert candidate is not None
    assert (candidate / "src" / "lisai" / "__init__.py").resolve() == Path(lisai.__file__).resolve()


def test_settings_show_reports_invalid_developer_selector(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    path = repo / "configs/local.yml"
    path.parent.mkdir(parents=True)
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(tmp_path / "ws")}}, path)
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: repo)
    assert root_cli.main(["settings", "show"]) == 0
    out = capsys.readouterr().out
    assert "Selection status:   invalid" in out
    assert "config.mode" in out
    assert not (tmp_path / "ws").exists()
