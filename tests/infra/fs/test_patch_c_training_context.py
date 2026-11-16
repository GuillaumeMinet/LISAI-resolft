"""Patch C: training, checkpoint and continuation must retain one workspace."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.training.continue_cli as continue_cli
import lisai.training.run_training as training_mod
from lisai.infra.fs.folders import create_run_dir, create_tb_folder
from lisai.infra.fs.output_folders import prepare_output_folder
from lisai.infra.paths import Paths
from lisai.runs.lifecycle import create_run_metadata, finalize_run_completed
from lisai.training.checkpointing.manager import CheckpointManager
from lisai.training.trainers.base import TrainingOutcome


def test_two_contexts_run_naming_checkpoints_and_tensorboard(settings_factory):
    a, b = settings_factory("a"), settings_factory("b")
    a.project_cfg.naming.run_dir_index_width = 2
    b.project_cfg.naming.run_dir_index_width = 4
    a.project_cfg.naming.exp_name_format = "{name}_A{id:02d}"
    b.project_cfg.naming.exp_name_format = "{name}_B{id:03d}"
    a.project_cfg.run_layout.subdirs["checkpoints"] = "weights_a"
    b.project_cfg.run_layout.subdirs["checkpoints"] = "weights_b"
    a.project_cfg.run_layout.artifacts["loss_file"] = "loss_a.txt"
    b.project_cfg.run_layout.artifacts["loss_file"] = "loss_b.txt"

    runs = []
    for stg, suffix in ((a, "00"), (b, "0000"), (a, "01")):
        paths = Paths(stg)
        run_dir, name = create_run_dir(paths, "dataset", "demo", "Upsamp")
        assert name == f"demo_{suffix}"
        assert run_dir == paths.run_dir(
            dataset_name="dataset", models_subfolder="Upsamp", exp_name=name,
        )
        ckpt = CheckpointManager(run_dir, {"enabled": True}, paths=paths)
        assert ckpt.paths is paths
        assert ckpt.checkpoints_dir == run_dir / ("weights_a" if stg is a else "weights_b")
        assert ckpt.loss_file == run_dir / ("loss_a.txt" if stg is a else "loss_b.txt")
        assert ckpt.checkpoints_dir.is_dir()
        tb_root = paths.tensorboard_dir(dataset_name="dataset", tensorboard_subfolder="x")
        first = tb_root / "demo"
        first.mkdir(parents=True, exist_ok=True)
        tb_dir, tb_name = create_tb_folder(
            tb_root, "demo", exist_ok=False,
            name_format=stg.NAMING.exp_name_format,
        )
        assert tb_name == (("demo_A01" if suffix == "00" else "demo_A02") if stg is a else "demo_B001")
        assert tb_dir.is_dir()
        runs.append(run_dir)

    assert runs[0] != runs[1]
    assert a.WORKSPACE_ROOT != b.WORKSPACE_ROOT


def test_numbered_folder_accepts_selected_format(tmp_path):
    requested = tmp_path / "prediction"
    requested.mkdir()
    out = prepare_output_folder(
        requested, if_exists_policy="numbered", numbering_format="{name}_custom{id:03d}",
    )
    assert out.path.name == "prediction_custom001"


def test_training_passes_same_paths_to_runtime_and_trainer(settings_factory, monkeypatch):
    a, b = settings_factory("train_a"), settings_factory("train_b")
    observed = []
    cfg = SimpleNamespace(
        training=SimpleNamespace(progress_bar=False),
        experiment=SimpleNamespace(exp_name="test", mode="train"),
        model=SimpleNamespace(architecture="unet"),
        data=SimpleNamespace(dataset_name="dataset"),
        routing=SimpleNamespace(models_subfolder="Upsamp"),
    )
    monkeypatch.setattr(training_mod, "resolve_config", lambda _path, *, stg: cfg)
    monkeypatch.setattr(training_mod, "resolve_progress_bar", lambda default, override, *, stg: default)

    def fake_runtime(c, *, paths):
        assert c is cfg
        observed.append(("runtime", paths.settings))
        return SimpleNamespace(
            paths=paths, run_dir=None, run_name="test", device="cpu", writer=None,
            callbacks=[], console_filter=None, file_filter=None, logger=None,
        )

    monkeypatch.setattr(training_mod, "initialize_runtime", fake_runtime)
    monkeypatch.setattr(training_mod.setup, "prepare_data", lambda c, runtime: SimpleNamespace(
        data_norm_prm={}, model_norm_prm={}, split_manifest=None,
        train_loader=(), val_loader=(), patch_info=None,
    ))
    monkeypatch.setattr(training_mod.setup, "save_training_config", lambda *a: None)
    monkeypatch.setattr(training_mod.setup, "build_model", lambda cfg, device, paths, prm: (object(), None))
    monkeypatch.setattr(training_mod, "count_trainable_parameters", lambda model: 0)

    class FakeTrainer:
        def train(self):
            return TrainingOutcome(reason="completed", last_completed_epoch=0)

    def fake_get_trainer(*, paths, **kwargs):
        observed.append(("trainer", paths.settings))
        return FakeTrainer()

    monkeypatch.setattr(training_mod, "get_trainer", fake_get_trainer)
    monkeypatch.setattr(training_mod, "_finalize_training_result", lambda *args, **kwargs: None)
    for stg in (a, b, a):
        result = training_mod.run_training("unused.yml", settings=stg)
        assert isinstance(result, FakeTrainer)

    assert observed == [
        ("runtime", a), ("trainer", a),
        ("runtime", b), ("trainer", b),
        ("runtime", a), ("trainer", a),
    ]


def test_continue_uses_selected_paths_for_discovery_and_training(settings_factory, monkeypatch):
    a, b = settings_factory("continue_a"), settings_factory("continue_b")
    runs = {}
    for stg in (a, b):
        paths = Paths(stg)
        run_dir, _ = create_run_dir(paths, "dataset", "resume", "Upsamp")
        create_run_metadata(run_dir, paths=paths, dataset="dataset", model_subfolder="Upsamp")
        finalize_run_completed(run_dir)
        runs[stg] = run_dir

    forwarded = []
    monkeypatch.setattr(
        continue_cli, "run_training_from_config_dict",
        lambda config, *, settings, **kw: forwarded.append((settings, Path(config["load_model"]["model_full_path"]))),
    )
    for stg in (a, b, a):
        assert continue_cli.continue_run(
            selector="resume_00", dataset="dataset", assume_yes=True, settings=stg,
        ) == 0
    assert forwarded == [(a, runs[a]), (b, runs[b]), (a, runs[a])]
