from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.data.cli as dataset_cli
from lisai.data.catalog import DatasetCatalog


SHA = "a" * 64


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def data_root(self) -> Path:
        return self.root

    def datasets_root(self) -> Path:
        return self.root / "datasets"

    def dataset_registry_path(self) -> Path:
        return self.root / "datasets" / "dataset_registry.yml"

    def dataset_dir(self, *, dataset_name: str, usage: str = "training", data_subfolder: str = "") -> Path:
        return self.root / "datasets" / usage / dataset_name / data_subfolder

    def noise_model_dir(self, *, noiseModel_name: str) -> Path:
        return self.root / "noise_models" / noiseModel_name


def _catalog() -> DatasetCatalog:
    return DatasetCatalog.model_validate(
        {
            "schema_version": 1,
            "datasets": {
                "train_a": {
                    "usage": "training",
                    "install_path": "datasets/training",
                    "archives": [
                        {"filename": "train_a.zip", "size_bytes": 12_000, "sha256": SHA}
                    ],
                    "registry_entry": {
                        "usage": "training",
                        "data_format": "single",
                        "description": "Training data A",
                    },
                    "dependencies": {
                        "evaluation_datasets": ["eval_a"],
                        "noise_models": ["noise_a"],
                    },
                },
                "eval_a": {
                    "usage": "evaluation",
                    "install_path": "datasets/evaluation",
                    "archives": [
                        {"filename": "eval_a.zip", "size_bytes": 4_000, "sha256": SHA}
                    ],
                    "registry_entry": {
                        "usage": "evaluation",
                        "data_format": "single",
                        "description": "Evaluation data A",
                    },
                    "dependencies": {
                        "evaluation_datasets": [],
                        "noise_models": [],
                    },
                },
            },
            "shared": {
                "noise_models": {
                    "archive": {
                        "filename": "noise_models.zip",
                        "size_bytes": 1_000,
                        "sha256": SHA,
                    },
                    "available": ["noise_a"],
                }
            },
        }
    )


def _patch_remote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cat = _catalog()
    paths = FakePaths(tmp_path / "data")
    monkeypatch.setattr(dataset_cli.download_catalog, "load_catalog", lambda: cat)
    monkeypatch.setattr(dataset_cli, "_paths", lambda: paths)
    return cat, paths


def _fake_result(names, *, noise=True):
    return SimpleNamespace(
        installed=tuple(names),
        already_installed=(),
        noise_models_installed=("noise_a",) if noise else (),
    )


def test_datasets_catalog_lists_download_size(monkeypatch, tmp_path: Path, capsys):
    _patch_remote(monkeypatch, tmp_path)

    assert dataset_cli.main(["catalog"]) == 0

    output = capsys.readouterr().out
    assert "train_a" in output
    assert "evaluation" in output
    assert "11.7 KiB" in output
    assert "Training data A" in output


def test_dataset_download_prompts_for_eval_then_final_plan(monkeypatch, tmp_path: Path, capsys):
    _patch_remote(monkeypatch, tmp_path)
    answers = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    captured = {}

    def fake_download_install(catalog, plan, *, record_id, paths, **kwargs):
        captured["names"] = [entry.name for entry in plan.entries]
        return _fake_result(captured["names"])

    monkeypatch.setattr(dataset_cli, "download_and_install_dataset_plan", fake_download_install)

    assert dataset_cli.main(["download", "train_a"]) == 0

    assert captured["names"] == ["eval_a", "train_a"]
    output = capsys.readouterr().out
    assert "Related evaluation datasets" in output
    assert "train_a" in output
    assert "eval_a" in output
    assert "noise_models.zip" in output
    assert "Total download" in output


def test_dataset_download_no_eval_keeps_only_requested_dataset(monkeypatch, tmp_path: Path):
    _patch_remote(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    captured = {}

    def fake_download_install(catalog, plan, *, record_id, paths, **kwargs):
        captured["names"] = [entry.name for entry in plan.entries]
        return _fake_result(["train_a"])

    monkeypatch.setattr(dataset_cli, "download_and_install_dataset_plan", fake_download_install)

    assert dataset_cli.main(["download", "train_a", "--no-eval"]) == 0
    assert captured["names"] == ["train_a"]


def test_dataset_download_all_requires_only_final_confirmation(monkeypatch, tmp_path: Path):
    _patch_remote(monkeypatch, tmp_path)
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(
        dataset_cli,
        "download_and_install_dataset_plan",
        lambda catalog, plan, *, record_id, paths, **kwargs: _fake_result(
            [entry.name for entry in plan.entries]
        ),
    )

    assert dataset_cli.main(["download", "--all"]) == 0
    assert len(prompts) == 1
    assert "Start download" in prompts[0]


def test_dataset_download_requires_name_or_all(monkeypatch, tmp_path: Path):
    _patch_remote(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        dataset_cli.main(["download"])

    assert exc_info.value.code == 2


def _prepare_install_release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    (release / "train_a.zip").write_bytes(b"train")
    (release / "eval_a.zip").write_bytes(b"eval")
    (release / "noise_models.zip").write_bytes(b"noise")
    (release / "application.zip").write_bytes(b"application")
    (release / "model_a.lisai.zip").write_bytes(b"model")
    return release


def test_dataset_install_archive_installs_only_that_dataset(monkeypatch, tmp_path: Path, capsys):
    cat, _paths = _patch_remote(monkeypatch, tmp_path)
    release = _prepare_install_release(tmp_path)
    monkeypatch.setattr(dataset_cli, "_catalog_for_install_source", lambda _path: cat)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    captured = {}

    def fake_install(catalog, plan, *, source, paths):
        captured["names"] = [entry.name for entry in plan.entries]
        captured["archives"] = set(source.archives)
        return _fake_result(captured["names"])

    monkeypatch.setattr(dataset_cli, "install_dataset_plan", fake_install)

    assert dataset_cli.main(["install", str(release / "train_a.zip")]) == 0

    assert captured["names"] == ["train_a"]
    assert captured["archives"] == {"train_a.zip"}
    output = capsys.readouterr().out
    assert "Optional related evaluation datasets not installed" in output
    assert "Start installation" not in output  # prompt text is consumed by the mocked input


def test_dataset_install_directory_installs_all_catalogued_archives(monkeypatch, tmp_path: Path):
    cat, _paths = _patch_remote(monkeypatch, tmp_path)
    release = _prepare_install_release(tmp_path)
    monkeypatch.setattr(dataset_cli, "_catalog_for_install_source", lambda _path: cat)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "y")
    captured = {}

    def fake_install(catalog, plan, *, source, paths):
        captured["names"] = [entry.name for entry in plan.entries]
        return _fake_result(captured["names"])

    monkeypatch.setattr(dataset_cli, "install_dataset_plan", fake_install)

    assert dataset_cli.main(["install", str(release)]) == 0

    assert captured["names"] == ["eval_a", "train_a"]
    assert len(prompts) == 1
    assert "Start installation" in prompts[0]
