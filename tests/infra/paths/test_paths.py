from __future__ import annotations

from types import SimpleNamespace

from lisai.infra.paths import Paths


def test_paths_exposes_project_root(tmp_path):
    settings = SimpleNamespace(PROJECT_ROOT=tmp_path)

    assert Paths(settings).project_root() == tmp_path.resolve()


def test_dataset_usage_subfolders_control_dataset_location(monkeypatch):
    from lisai.config import settings

    monkeypatch.setitem(
        settings.project_cfg.paths.dataset_usage_subfolders,
        "training",
        "train_custom",
    )
    monkeypatch.setitem(
        settings.project_cfg.paths.dataset_usage_subfolders,
        "evaluation",
        "eval_custom",
    )

    paths = Paths(settings)

    assert paths.dataset_usage_subfolder("training") == "train_custom"
    assert paths.dataset_usage_subfolder("evaluation") == "eval_custom"
    assert paths.dataset_dir(dataset_name="demo", usage="training").parts[-2:] == (
        "train_custom",
        "demo",
    )
    assert paths.dataset_dir(dataset_name="demo", usage="evaluation").parts[-2:] == (
        "eval_custom",
        "demo",
    )
