"""Final M2 release gates for workspace isolation and boundary selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.selection import resolve_cli_locations
from lisai.config.settings import Settings, load_default_settings
from lisai.config.workspace import initialize_workspace
from lisai.data.dataset_registry import load_dataset_registry
from lisai.infra.paths import Paths
from lisai_runner.queue.storage import default_queue_root


def _prepared(tmp_path: Path):
    a, b = tmp_path / "A", tmp_path / "B"
    initialize_workspace(a)
    initialize_workspace(b)
    for root, name in ((a, "from_A"), (b, "from_B")):
        paths = Paths(Settings.from_workspace(root))
        save_yaml({name: {"usage": "training", "data_format": "single"}}, paths.dataset_registry_path())
        save_yaml({f"model_{name}": {"dir": name}}, paths.promoted_model_registry_path())
    return a, b


def test_a_b_a_separate_data_models_runs_queue_and_inference(tmp_path):
    a, b = _prepared(tmp_path)
    seen = []
    for root in (a, b, a):
        paths = Paths(Settings.from_workspace(root))
        dataset = next(iter(load_dataset_registry(paths.dataset_registry_path())))
        models = load_yaml(paths.promoted_model_registry_path())
        seen.append((dataset, next(iter(models))))
        assert paths.run_dir(dataset_name=dataset, models_subfolder="hdn", exp_name="run_00") == (
            root / f"datasets/training/{dataset}/runs/hdn/run_00"
        )
        assert paths.inference_output_dir(source_name="sample", model_name="model") == (
            root / "inference/sample/model"
        )
        assert paths.tensorboard_dir(dataset_name=dataset) == root / "tensorboard" / dataset
        assert default_queue_root(paths=paths) == root / ".lisai/queue"
    assert seen == [("from_A", "model_from_A"), ("from_B", "model_from_B"),
                    ("from_A", "model_from_A")]


def test_external_experiment_configs_remain_selected_with_workspace_override(tmp_path):
    a, b = _prepared(tmp_path)
    source = tmp_path / "checkout"
    editable = source / "configs"
    editable.mkdir(parents=True)
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(a)},
               "console": {"progress_bar": False}}, editable / "local.yml")

    selected = resolve_cli_locations(workspace=b, environ={}, editable_source_root=source)
    assert selected.workspace_root == b
    assert selected.experiment_config_root == editable
    assert selected.developer_mode
    configured = Settings.from_workspace(b, config_root=selected.experiment_config_root)
    assert configured.CONFIGS_ROOT == editable
    assert configured.local_cfg.console.progress_bar is False
    assert Paths(configured).dataset_registry_path() == b / "datasets/dataset_registry.yml"

    forced = resolve_cli_locations(workspace=b, config_root=b / "configs", environ={},
                                   editable_source_root=source)
    assert forced.experiment_config_root == b / "configs"
    assert not forced.developer_mode
    assert Settings.from_workspace(b, config_root=forced.experiment_config_root).local_cfg.console.progress_bar is None


def test_python_api_ignores_cli_config_root_and_developer_flag(tmp_path, monkeypatch):
    a, b = _prepared(tmp_path)
    repo_configs = tmp_path / "checkout/configs"
    repo_configs.mkdir(parents=True)
    save_yaml({"dev": {"enabled": True}, "workspace": {"root": str(b)},
               "console": {"progress_bar": False}}, repo_configs / "local.yml")
    monkeypatch.setenv("LISAI_HOME", str(a))
    monkeypatch.setenv("LISAI_CONFIG_ROOT", str(repo_configs))
    selected = load_default_settings()
    assert selected.WORKSPACE_ROOT == a
    assert selected.CONFIGS_ROOT == a / "configs"
    assert selected.local_cfg.console.progress_bar is None


def test_explicit_settings_read_does_not_modify_any_workspace_files(tmp_path):
    a, b = _prepared(tmp_path)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for root in (a, b)
              for p in (root / "configs/workspace.yml", root / "configs/data.yml")}
    for root in (a, b, a):
        Settings.from_workspace(root)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == data for p, data in before.items())


def test_missing_workspace_is_actionable_and_not_created(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="lisai workspace init"):
        Settings.from_workspace(missing)
    assert not missing.exists()
