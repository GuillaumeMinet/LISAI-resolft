"""Patch E: workspace lifecycle ownership, non-destructive preview/update."""
from __future__ import annotations

from pathlib import Path

import pytest

from lisai.cli import main
from lisai.config import selection
from lisai.config.workspace import (
    describe_workspace_update,
    initialize_workspace,
    plan_workspace_update,
    update_workspace,
)


def test_core_only_then_full_is_idempotent_without_overwriting(tmp_path):
    workspace = tmp_path / "W"
    model = workspace / "models" / "trained.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weight")

    first = initialize_workspace(workspace, core_only=True)
    assert workspace / "configs/workspace.yml" in first
    assert (workspace / "configs/data.yml").is_file()
    assert (workspace / ".lisai/queue").is_dir()
    assert not (workspace / "configs/training").exists()
    assert not (workspace / "configs/inference/defaults.yml").exists()
    assert not (workspace / "configs/schema").exists()
    assert initialize_workspace(workspace, core_only=True) == []

    user_data = workspace / "configs/data.yml"
    user_data.write_text(user_data.read_text() + "\n# user edit\n")
    full = initialize_workspace(workspace)
    assert workspace / "configs/inference/defaults.yml" in full
    assert (workspace / "configs/training").is_dir()
    assert (workspace / "configs/preprocess").is_dir()
    assert (workspace / "configs/schema/experiment.schema.json").is_file()
    assert user_data.read_text().endswith("# user edit\n")
    assert model.read_bytes() == b"weight"
    assert initialize_workspace(workspace) == []


def test_cli_core_only_and_full_do_not_change_selection(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.setenv("LISAI_HOME", str(first))
    assert main(["workspace", "init", str(first), "--core-only"]) == 0
    assert not (first / "configs/training").exists()
    assert main(["workspace", "init", str(second)]) == 0
    assert (second / "configs/training").is_dir()
    assert not (first / "configs/training").exists()
    assert main(["workspace", "init", str(first)]) == 0
    assert (first / "configs/training").is_dir()
    assert selection.resolve_cli_locations().workspace_root == first


def test_update_preview_cancel_and_yes_always_show_changes(tmp_path, monkeypatch, capsys):
    root = tmp_path / "workspace"
    initialize_workspace(root)
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    global_yaml = root / "configs/workspace.yml"
    global_yaml.write_text(global_yaml.read_text() + "\n# custom note\n")
    previous = global_yaml.read_bytes()

    plan = plan_workspace_update(root)
    assert plan.requires_writes
    text = describe_workspace_update(plan)
    assert "REPLACE: " + str(global_yaml) in text
    assert "UNCHANGED: " + str(root / "configs/data.yml") in text
    assert "-# custom note" in text
    assert previous == global_yaml.read_bytes()
    assert not list(global_yaml.parent.glob("*.bak.*"))

    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert main(["workspace", "update", str(root)]) == 0
    output = capsys.readouterr().out
    assert "REPLACE:" in output and "Cancelled" in output
    assert global_yaml.read_bytes() == previous
    assert not list(global_yaml.parent.glob("*.bak.*"))

    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("--yes prompted unexpectedly"))
    assert main(["workspace", "update", str(root), "--yes"]) == 0
    output = capsys.readouterr().out
    assert "REPLACE:" in output and "Backup:" in output
    assert not global_yaml.read_text().endswith("# custom note\n")
    backups = list(global_yaml.parent.glob("workspace.yml.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous
    assert not list(global_yaml.parent.glob("data.yml.bak.*"))

    # No-op never rewrites files, creates backups, or asks for confirmation.
    assert main(["workspace", "update", str(root)]) == 0
    assert "No changes required" in capsys.readouterr().out
    assert len(list(global_yaml.parent.glob("workspace.yml.bak.*"))) == 1


def test_update_explicit_root_not_current_and_inference_opt_in(tmp_path, monkeypatch, capsys):
    selected, target, external = tmp_path / "active", tmp_path / "other", tmp_path / "external"
    initialize_workspace(selected)
    initialize_workspace(target)
    monkeypatch.setattr(selection, "_source_checkout_for_import", lambda: None)
    monkeypatch.setenv("LISAI_HOME", str(selected))

    selected_global = selected / "configs/workspace.yml"
    target_global = target / "configs/workspace.yml"
    selected_global.write_text(selected_global.read_text() + "\n# selected edit\n")
    target_global.write_text(target_global.read_text() + "\n# other edit\n")
    selected_before = selected_global.read_bytes()
    target_inference = target / "configs/inference/defaults.yml"
    target_inference.write_text("arbitrary: user inference\n")
    outside = external / "inference/defaults.yml"
    outside.parent.mkdir(parents=True)
    outside.write_text("user: external\n")

    assert main(["workspace", "update", str(target), "--yes"]) == 0
    assert selected_global.read_bytes() == selected_before
    assert "# other edit" not in target_global.read_text()
    assert target_inference.read_text() == "arbitrary: user inference\n"
    assert outside.read_text() == "user: external\n"
    assert selection.resolve_cli_locations().workspace_root == selected

    assert main(["workspace", "update", str(target), "--include-inference", "--yes"]) == 0
    assert target_inference.read_text().startswith("# yaml-language-server:")
    assert outside.read_text() == "user: external\n"
    assert "Inference config root:" in capsys.readouterr().out

    # Explicit external path is a lifecycle operation target, not a selection change.
    assert main(["workspace", "update", str(target), "--external-config-root", str(external), "--yes"]) == 2
    assert outside.read_text() == "user: external\n"
    assert main(["workspace", "update", str(target), "--include-inference",
                 "--external-config-root", str(external), "--yes"]) == 0
    assert outside.read_text().startswith("# yaml-language-server:")
    assert "external (explicit)" in capsys.readouterr().out
    assert selection.resolve_cli_locations().workspace_root == selected


def test_changed_during_update_preview_aborts_before_any_write(tmp_path):
    root = tmp_path / "W"
    initialize_workspace(root)
    data = root / "configs/data.yml"
    config = root / "configs/workspace.yml"
    config.write_text(config.read_text() + "\n# to replace\n")
    plan = plan_workspace_update(root)
    config.write_text(config.read_text() + "# changed during confirmation\n")
    before = config.read_bytes()
    with pytest.raises(ValueError, match="changed since update preview"):
        update_workspace(plan)
    assert config.read_bytes() == before
    assert not list(root.glob("configs/*.bak.*"))
    assert data.is_file()


def test_missing_inference_files_are_only_created_on_explicit_opt_in(tmp_path):
    root = tmp_path / "core"
    initialize_workspace(root, core_only=True)
    inference = root / "configs/inference/defaults.yml"
    plan = plan_workspace_update(root)
    assert inference not in [change.path for change in plan.changes]
    assert not inference.exists()
    update_workspace(plan)
    assert not inference.exists()

    inferred = plan_workspace_update(root, include_inference=True)
    assert {c.action for c in inferred.changes if c.path.name in {"defaults.yml", "post_training.yml"}} == {"CREATE"}
    assert not inference.exists()  # plan is read-only
    update_workspace(inferred)
    assert inference.is_file()
    assert (root / "configs/inference/post_training.yml").is_file()
