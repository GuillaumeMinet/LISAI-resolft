from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from pydantic import BaseModel

import lisai.promoted_models.package as package_module
from lisai.promoted_models.dependencies import PromotionFile
from lisai.promoted_models.schema import (
    PromotedModelArtifacts,
    PromotedModelDefinition,
    PromotedModelManifest,
    PromotedModelSource,
    PromotedTrainingData,
)


class FakeParams(BaseModel):
    channels: int = 16
    depth: int = 3


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def promoted_models_root(self):
        return self.root / "models"

    def promoted_model_registry_path(self):
        return self.promoted_models_root() / "model_registry.yml"

    def promoted_model_dir(self, *, model_name: str):
        return self.promoted_models_root() / model_name

    def promoted_model_exports_dir(self):
        return self.promoted_models_root() / "exports"

    def dataset_registry_path(self):
        return self.root / "datasets" / "dataset_registry.yml"

    def dataset_dir(self, *, dataset_name: str, data_subfolder: str = "", usage: str = "training"):
        return self.root / "datasets" / usage / dataset_name / data_subfolder


def _resolved_config() -> dict:
    return {
        "experiment": {
            "mode": "train",
            "exp_name": "run_a",
            "task": {"name": "denoising_hdn", "supervised": True, "betaKL": 0.001},
        },
        "routing": {"data_subfolder": "raw"},
        "data": {
            "dataset_name": "DemoData",
            "paired": True,
            "input": "inp",
            "target": "gt",
            "patch_size": 64,
        },
        "model": {
            "architecture": "lvae",
            "parameters": {"num_latents": 2, "z_dims": 16},
        },
        "training": {"n_epochs": 10},
        "normalization": {"norm_prm": {"clip": 0}},
        "model_norm_prm": {"data_mean": 1.0, "data_std": 2.0},
        "noise_model": {"name": "NoiseA"},
        "saving": {
            "enabled": True,
            "state_dict": True,
            "entire_model": False,
            "overwrite_best": True,
        },
    }


def _plan(tmp_path: Path):
    config = tmp_path / "config_train.yaml"
    config.write_text(yaml.safe_dump(_resolved_config(), sort_keys=False), encoding="utf-8")
    loss = tmp_path / "loss.txt"
    loss.write_text("Epoch Train_loss Val_loss\n1 1.0 0.8\n", encoding="utf-8")
    loss_plot = tmp_path / "loss_plot.png"
    loss_plot.write_bytes(b"png")
    checkpoint = tmp_path / "model_best_state_dict.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer_state_dict": {"not": "published"},
        },
        checkpoint,
    )

    manifest = PromotedModelManifest(
        name="demo-model",
        model=PromotedModelDefinition(task="denoising_hdn", architecture="lvae"),
        training_data=PromotedTrainingData(dataset="DemoData"),
        source=PromotedModelSource(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            run_name="run_a",
            run_status="stopped",
            checkpoint_selector="best",
            checkpoint_filename=checkpoint.name,
        ),
        artifacts=PromotedModelArtifacts(
            loss="training/loss.txt",
            loss_plot="training/loss_plot.png",
        ),
    )
    saved_run = SimpleNamespace(model_parameters=FakeParams())
    return SimpleNamespace(
        manifest=manifest,
        checkpoint_path=checkpoint,
        copy_files=(
            PromotionFile(config, "config_train.yaml"),
            PromotionFile(loss, "training/loss.txt"),
            PromotionFile(loss_plot, "training/loss_plot.png"),
        ),
        saved_run=saved_run,
    )


def test_promote_creates_canonical_directory_registry_and_clean_weights(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path)
    paths = FakePaths(tmp_path)
    monkeypatch.setattr(package_module, "build_promotion_plan", lambda *args, **kwargs: plan)

    result = package_module.promote_run(
        tmp_path / "run",
        name="demo-model",
        paths=paths,
    )

    assert result.model_dir == tmp_path / "models" / "demo-model"
    assert (result.model_dir / "README.md").exists()
    assert (result.model_dir / "training" / "loss_plot.png").exists()
    state_dict = torch.load(result.weights_path, map_location="cpu", weights_only=True)
    assert set(state_dict) == {"weight"}

    manifest = yaml.safe_load((result.model_dir / "lisai_model.yaml").read_text())
    assert manifest["name"] == "demo-model"
    assert "version" not in manifest
    assert set(manifest["checksums"]) == {
        "config_train.yaml",
        "training/loss.txt",
        "training/loss_plot.png",
        "weights.pt",
    }

    registry = yaml.safe_load(paths.promoted_model_registry_path().read_text())
    assert registry["models"]["demo-model"]["source_run_id"] == plan.manifest.source.run_id
    assert registry["models"]["demo-model"]["path"] == "demo-model"
    assert registry["models"]["demo-model"]["origin"] == "promoted"


def test_promote_snapshots_training_dataset_documentation_in_model_card(
    tmp_path: Path, monkeypatch
):
    plan = _plan(tmp_path)
    paths = FakePaths(tmp_path)
    monkeypatch.setattr(package_module, "build_promotion_plan", lambda *args, **kwargs: plan)

    registry_path = paths.dataset_registry_path()
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        "DemoData:\n"
        "  data_format: timelapse\n"
        "  usage: training\n"
        "  description: Live demo dataset for model training.\n",
        encoding="utf-8",
    )
    dataset_dir = paths.dataset_dir(dataset_name="DemoData", usage="training")
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "README.md").write_text(
        "# DemoData\n\n"
        "**Sample**\n"
        "- Cell type: U2OS\n\n"
        "**Acquisition**\n"
        "- Pixel size: 30 nm\n",
        encoding="utf-8",
    )

    result = package_module.promote_run(
        tmp_path / "run",
        name="demo-model",
        paths=paths,
    )

    card = (result.model_dir / "README.md").read_text(encoding="utf-8")
    assert "## Training dataset" in card
    assert "- Dataset: `DemoData`" in card
    assert "- Description: Live demo dataset for model training." in card
    assert "**Sample**" in card
    assert "- Cell type: U2OS" in card
    assert "- Pixel size: 30 nm" in card
    assert "# DemoData" not in card


def test_export_zips_existing_promoted_model_under_data_root(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path)
    paths = FakePaths(tmp_path)
    monkeypatch.setattr(package_module, "build_promotion_plan", lambda *args, **kwargs: plan)
    package_module.promote_run(tmp_path / "run", name="demo-model", paths=paths)

    exported = package_module.export_promoted_model("demo-model", paths=paths)

    assert exported.archive_path == tmp_path / "models" / "exports" / "demo-model.lisai.zip"
    assert exported.archive_sha256 == hashlib.sha256(exported.archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(exported.archive_path) as archive:
        assert "lisai_model.yaml" in archive.namelist()
        assert "weights.pt" in archive.namelist()
        assert "training/loss.txt" in archive.namelist()


def test_promote_refuses_duplicate_name_without_overwrite(tmp_path: Path, monkeypatch):
    plan = _plan(tmp_path)
    paths = FakePaths(tmp_path)
    monkeypatch.setattr(package_module, "build_promotion_plan", lambda *args, **kwargs: plan)
    package_module.promote_run(tmp_path / "run", name="demo-model", paths=paths)

    try:
        package_module.promote_run(tmp_path / "run", name="demo-model", paths=paths)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected duplicate promoted-model name to be rejected.")
