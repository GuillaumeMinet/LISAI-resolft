from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import lisai.promoted_models.promotion as promotion
from lisai.runs.schema import CodeState, RunMetadata, RuntimeStats, TrainingSignature


RUN_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def _metadata(tmp_path: Path, *, status: str = "stopped", dataset: str = "dataset_a") -> RunMetadata:
    return RunMetadata(
        run_id=RUN_ID,
        run_name="run_a",
        run_index=0,
        dataset=dataset,
        model_subfolder="lvae",
        status=status,
        closed_cleanly=status in {"completed", "stopped", "failed"},
        created_at=NOW,
        updated_at=NOW,
        ended_at=NOW if status in {"completed", "stopped", "failed"} else None,
        last_heartbeat_at=NOW,
        last_epoch=4,
        max_epoch=10,
        best_val_loss=0.25,
        path=str(tmp_path),
        group_path=None,
        training_signature=TrainingSignature(
            architecture="lvae",
            train_batch_size=2,
            train_patch_size=64,
            trainable_params=1234,
        ),
        runtime_stats=RuntimeStats(total_training_time_sec=100.0),
        code=CodeState(
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            git_remote="https://example.test/lisai.git",
            lisai_version="0.1.0",
        ),
    )


def _raw_config(dataset: str = "dataset_a") -> dict:
    return {
        "experiment": {
            "mode": "train",
            "exp_name": "run_a",
            "task": {"name": "denoising_hdn", "supervised": True, "betaKL": 0.001},
        },
        "routing": {"data_subfolder": "raw"},
        "data": {
            "dataset_name": dataset,
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


class FakePaths:
    def __init__(self, run_dir: Path, noise_root: Path):
        self.run_dir = run_dir
        self.noise_root = noise_root

    def cfg_train_path(self, *, run_dir):
        return Path(run_dir) / "config_train.yaml"

    def checkpoint_path(self, *, run_dir, load_method, best_or_last):
        return Path(run_dir) / "checkpoints" / f"model_{best_or_last}_{load_method}.pt"

    def loss_file_path(self, *, run_dir):
        return Path(run_dir) / "loss.txt"

    def loss_plot_path(self, *, run_dir):
        return Path(run_dir) / "loss_plot.png"

    def noise_model_path(self, *, noiseModel_name):
        return self.noise_root / noiseModel_name / "GMM.npz"

    def noise_model_norm_prm_path(self, *, noiseModel_name):
        return self.noise_root / noiseModel_name / "norm_prm.json"

    def split_manifest_path(self, *, run_dir):
        return Path(run_dir) / "split_manifest.json"


def _prepare_sources(tmp_path: Path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "config_train.yaml").write_text("placeholder", encoding="utf-8")
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "model_best_state_dict.pt").write_bytes(b"checkpoint")
    (run_dir / "loss.txt").write_text("Epoch Train_loss Val_loss\n", encoding="utf-8")
    (run_dir / "loss_plot.png").write_bytes(b"png")

    noise_root = tmp_path / "noise_models"
    noise_dir = noise_root / "NoiseA"
    noise_dir.mkdir(parents=True)
    (noise_dir / "GMM.npz").write_bytes(b"gmm")
    (noise_dir / "norm_prm.json").write_text("{}", encoding="utf-8")
    return run_dir, FakePaths(run_dir, noise_root)


def test_build_promotion_plan_projects_stopped_run_and_publication_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir, paths = _prepare_sources(tmp_path)
    metadata = _metadata(run_dir, status="stopped")
    monkeypatch.setattr(promotion, "read_run_metadata", lambda _run_dir: metadata)
    monkeypatch.setattr(promotion, "load_yaml", lambda _path: _raw_config())

    plan = promotion.build_promotion_plan(
        run_dir,
        name="hdn-vimentin",
        paths=paths,
    )

    assert plan.metadata.status == "stopped"
    assert plan.checkpoint_path.name == "model_best_state_dict.pt"
    assert plan.manifest.name == "hdn-vimentin"
    assert plan.manifest.model.task == "denoising_hdn"
    assert plan.manifest.model.architecture == "lvae"
    assert plan.manifest.training_data.dataset == "dataset_a"
    assert plan.manifest.source.code is not None
    assert plan.manifest.source.code.git_commit == "abc123"
    assert plan.manifest.training.best_val_loss == 0.25
    assert plan.manifest.artifacts.loss == "training/loss.txt"
    assert plan.manifest.artifacts.loss_plot == "training/loss_plot.png"
    assert plan.manifest.artifacts.noise_model is not None
    assert [item.package_path for item in plan.copy_files] == [
        "config_train.yaml",
        "training/loss.txt",
        "training/loss_plot.png",
        "artifacts/noise_model/GMM.npz",
        "artifacts/noise_model/norm_prm.json",
    ]


def test_build_promotion_plan_accepts_completed_run(tmp_path: Path, monkeypatch):
    run_dir, paths = _prepare_sources(tmp_path)
    monkeypatch.setattr(
        promotion,
        "read_run_metadata",
        lambda _run_dir: _metadata(run_dir, status="completed"),
    )
    monkeypatch.setattr(promotion, "load_yaml", lambda _path: _raw_config())

    plan = promotion.build_promotion_plan(run_dir, name="model-a", paths=paths)

    assert plan.manifest.source.run_status == "completed"


def test_build_promotion_plan_rejects_non_terminal_run(tmp_path: Path, monkeypatch):
    run_dir, paths = _prepare_sources(tmp_path)
    monkeypatch.setattr(
        promotion,
        "read_run_metadata",
        lambda _run_dir: _metadata(run_dir, status="running"),
    )

    with pytest.raises(ValueError, match="Only completed, stopped runs can be promoted"):
        promotion.build_promotion_plan(run_dir, name="model-a", paths=paths)


def test_build_promotion_plan_rejects_dataset_mismatch(tmp_path: Path, monkeypatch):
    run_dir, paths = _prepare_sources(tmp_path)
    monkeypatch.setattr(
        promotion,
        "read_run_metadata",
        lambda _run_dir: _metadata(run_dir, dataset="metadata_dataset"),
    )
    monkeypatch.setattr(promotion, "load_yaml", lambda _path: _raw_config(dataset="config_dataset"))

    with pytest.raises(ValueError, match="disagree on dataset name"):
        promotion.build_promotion_plan(run_dir, name="model-a", paths=paths)
