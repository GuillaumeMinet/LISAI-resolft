"""M2 E: explicit workspace lifecycle, data-preservation and CLI boundaries."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lisai.cli import main
from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.selection import resolve_cli_locations
from lisai.config.settings import Settings
from lisai.config.workspace import (
    initialize_developer, initialize_workspace, migrate_legacy_workspace,
    revision_warning, update_workspace,
)

ROOT = Path(__file__).resolve().parents[3]


def test_workspace_explicit_init_is_idempotent_and_preserves_assets(tmp_path):
    root = tmp_path / "space"
    prior = root / "datasets/training/vim/runs/001/weights.pth"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"trained")
    created = initialize_workspace(root)
    assert root / "configs/workspace.yml" in created
    assert (root / "configs/data.yml").exists()
    assert (root / ".lisai/queue").is_dir()
    assert (root / "configs/inference/defaults.yml").is_file()
    assert (root / "configs/inference/post_training.yml").is_file()
    assert (root / "configs/schema/experiment.schema.json").is_file()
    cfg = Settings.from_workspace(root)
    assert cfg.DEFAULTS_REVISION == 1
    assert cfg.WORKSPACE_ROOT == root
    defaults = root / "configs/inference/defaults.yml"
    defaults.write_text("custom: true\n")
    assert initialize_workspace(root) == []
    assert defaults.read_text() == "custom: true\n"
    assert prior.read_bytes() == b"trained"


def test_developer_init_uses_repo_scaffold_not_workspace(tmp_path):
    source = tmp_path / "checkout"
    (source / "src/lisai").mkdir(parents=True)
    (source / "src/lisai/__init__.py").touch()
    (source / "pyproject.toml").write_text('[project]\nname="lisai"\n')
    root = tmp_path / "space"
    initialize_workspace(root, core_only=True)
    assert not (root / "configs/training").exists()
    created = initialize_developer(source, root)
    assert source / "configs/local.yml" in created
    assert load_yaml(source / "configs/local.yml")["dev"]["enabled"] is True
    assert (source / "configs/inference/defaults.yml").exists()
    assert (source / "configs/schema/preprocess.schema.json").exists()
    assert not (root / "configs/training").exists()
    assert initialize_developer(source, root) == []
    initialize_workspace(tmp_path / "other", core_only=True)
    with pytest.raises(ValueError, match="selects"):
        initialize_developer(source, tmp_path / "other")


def test_update_backs_up_both_yaml_and_preserves_user_configs(tmp_path):
    root = tmp_path / "space"
    initialize_workspace(root)
    global_cfg = root / "configs/workspace.yml"
    data_cfg = root / "configs/data.yml"
    before_global = global_cfg.read_bytes()
    before_data = data_cfg.read_bytes()
    local = root / "configs/training/experiment.yml"
    local.write_text("experiment: my-own-values\n")
    weights = root / "models/my-model/weights.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    with global_cfg.open("a") as fp:
        fp.write("\n# personal change\n")
    backups = update_workspace(root)
    assert len(backups) == 2
    for original, backup in backups:
        assert backup.exists()
        assert backup != original
    assert backups[0][1].read_bytes().endswith(b"# personal change\n")
    assert global_cfg.read_bytes() == before_global
    assert data_cfg.read_bytes() == before_data
    assert local.read_text() == "experiment: my-own-values\n"
    assert weights.read_bytes() == b"weights"


def test_update_local_inference_opt_in_and_backup(tmp_path):
    root = tmp_path / "workspace"
    external = tmp_path / "external"
    initialize_workspace(root)
    custom = external / "inference/defaults.yml"
    custom.parent.mkdir(parents=True)
    custom.write_text("my: previous defaults\n")
    post = external / "inference/post_training.yml"
    post.write_text("user: previous override\n")
    update_workspace(root, config_root=external)
    assert custom.read_text() == "my: previous defaults\n"
    backups = update_workspace(root, config_root=external, replace_editable_defaults=True)
    assert {p for p, _ in backups} >= {custom, post}
    assert custom.read_text().startswith("# yaml-language-server:")
    assert next(backup for p, backup in backups if p == custom).read_text() == "my: previous defaults\n"


def _legacy_config_dir(tmp_path: Path, workspace: Path) -> Path:
    root = tmp_path / "legacy"
    root.mkdir()
    save_yaml({"infrastructure": {"data_root": str(workspace)},
               "console": {"progress_bar": True},
               "inference": {"output": {"mode": "default", "save_input_mode": "never"}}},
              root / "local_config.yml")
    for source, target in (("project_config.yml", "project_config.yml"),
                           ("data_config.yml", "data_config.yml")):
        (root / target).write_bytes((ROOT / "configs" / source).read_bytes())
    return root


def test_migrate_legacy_read_only_and_never_moves_data(tmp_path):
    root = tmp_path / "ws"
    weights = root / "models/installed/weight.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weight")
    legacy = _legacy_config_dir(tmp_path, root)
    originals = {p.name: p.read_bytes() for p in legacy.glob("*.yml")}
    converted = migrate_legacy_workspace(legacy, root)
    assert len(converted) == 2
    new = (root / "configs/workspace.yml").read_text()
    assert "{data_root}" not in new
    assert "{workspace_root}" in new
    assert "code_dir" not in new
    assert Settings.from_workspace(root).LOCAL_PROGRESS_BAR is True
    assert weights.read_bytes() == b"weight"
    assert originals == {p.name: p.read_bytes() for p in legacy.glob("*.yml")}
    with pytest.raises(FileExistsError):
        migrate_legacy_workspace(legacy, root)
    with pytest.raises(ValueError, match="does not match"):
        migrate_legacy_workspace(legacy, tmp_path / "other")


def test_revision_outdated_warning_without_auto_update(tmp_path):
    root = tmp_path / "ws"
    initialize_workspace(root)
    config = root / "configs/workspace.yml"
    raw = load_yaml(config)
    raw["defaults_revision"] = 0
    save_yaml(raw, config)
    snapshot = config.read_bytes()
    assert "lisai workspace update" in revision_warning(root)
    assert Settings.from_workspace(root).DEFAULTS_REVISION == 0
    assert config.read_bytes() == snapshot
    update_workspace(root)
    assert revision_warning(root) is None


def test_cli_init_and_show_and_update_confirm(tmp_path, monkeypatch, capsys):
    root = tmp_path / "space"
    assert main(["workspace", "init", str(root)]) == 0
    monkeypatch.setenv("LISAI_HOME", str(root))
    assert main(["settings", "show"]) == 0
    shown = capsys.readouterr().out
    assert "Defaults revision:  1" in shown
    old = (root / "configs/workspace.yml").read_bytes()
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    assert main(["workspace", "update", str(root)]) == 0
    assert not list((root / "configs").glob("workspace.yml.bak.*"))
    monkeypatch.setattr("builtins.input", lambda *args: "y")
    assert main(["workspace", "update", str(root)]) == 0
    assert list((root / "configs").glob("workspace.yml.bak.*"))
    assert (root / "configs/workspace.yml").read_bytes() == old


def test_import_and_help_do_not_initialize(tmp_path, capsys, monkeypatch):
    root = tmp_path / "absent"
    monkeypatch.delenv("LISAI_HOME", raising=False)
    monkeypatch.setattr("lisai.config.selection._source_checkout_for_import", lambda: None)
    with pytest.raises(SystemExit) as result:
        main(["workspace", "--help"])
    assert result.value.code == 0
    assert not root.exists()
    assert main(["settings", "show"]) == 0
    assert "no workspace selected" in capsys.readouterr().out
    assert not root.exists()


def test_cli_dev_init_core_only_and_explicit_activation(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LISAI_CONFIG_ROOT", raising=False)
    root = tmp_path / "workspace"
    source = tmp_path / "checkout"
    (source / "src/lisai").mkdir(parents=True)
    (source / "src/lisai/__init__.py").touch()
    (source / "pyproject.toml").write_text('[project]\nname="lisai"\n')
    monkeypatch.setattr("lisai.config.selection._source_checkout_for_import", lambda: source)
    # Explicit --yes approves core-only bootstrap in the selected workspace.
    monkeypatch.setattr("builtins.input", lambda *_: str(root))
    assert main(["dev", "init", "--yes"]) == 0
    assert (root / "configs/workspace.yml").is_file()
    assert not (root / "configs/training").exists()
    assert (source / "configs/local.yml").is_file()
    assert (source / "configs/training").is_dir()
    monkeypatch.setattr("builtins.input", lambda *_: str(root))
    assert main(["dev", "init", "--yes"]) == 0
    assert main(["settings", "show"]) == 0
    output = capsys.readouterr().out
    assert "Developer" in output
    assert "config.mode=repo" in output


def test_cli_legacy_migration_command(tmp_path, monkeypatch):
    root = tmp_path / "space"
    old = _legacy_config_dir(tmp_path, root)
    monkeypatch.setattr("lisai.config.selection._source_checkout_for_import", lambda: None)
    assert main(["workspace", "migrate", str(root), "--from-legacy", str(old)]) == 0
    assert (root / "configs/workspace.yml").exists()
    assert (root / "configs/data.yml").exists()
    assert main(["workspace", "migrate", str(root), "--from-legacy", str(old)]) == 2
