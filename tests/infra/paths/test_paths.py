from __future__ import annotations

from types import SimpleNamespace

from lisai.infra.paths import Paths


def test_paths_exposes_project_root(tmp_path):
    settings = SimpleNamespace(PROJECT_ROOT=tmp_path)

    assert Paths(settings).project_root() == tmp_path.resolve()


def test_paths_exposes_data_root(tmp_path):
    settings = SimpleNamespace(DATA_ROOT=tmp_path / "data")

    assert Paths(settings).data_root() == (tmp_path / "data").resolve()


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


def test_inference_output_dir_uses_effective_inference_root(monkeypatch, tmp_path):
    from lisai.config import settings

    monkeypatch.setitem(
        settings._ctx.paths.roots,
        "inference_dir",
        str(tmp_path / "predictions"),
    )

    paths = Paths(settings)

    assert paths.inference_root() == (tmp_path / "predictions").resolve()
    assert paths.inference_output_dir(
        source_name="paper_mito",
        model_name="mito_model",
    ) == tmp_path / "predictions" / "paper_mito" / "mito_model"


def test_promoted_model_downloads_dir_uses_project_template(monkeypatch, tmp_path):
    from lisai.config import settings

    monkeypatch.setitem(
        settings._ctx.paths.templates,
        "promoted_model_downloads_dir",
        str(tmp_path / "custom_downloads"),
    )

    assert Paths(settings).promoted_model_downloads_dir() == tmp_path / "custom_downloads"
