from __future__ import annotations

from pathlib import Path

import pytest

import lisai.training.checkpointing.manager as manager_mod


class FakePaths:
    def __init__(self, _settings):
        pass

    def checkpoints_dir(self, *, run_dir):
        return Path(run_dir) / "checkpoints"

    def loss_file_path(self, *, run_dir):
        return Path(run_dir) / "loss.txt"


def test_checkpoint_manager_disabled_without_run_dir():
    cm = manager_mod.CheckpointManager(run_dir=None, saving_prm={"enabled": True}, paths=FakePaths(None))

    assert cm.enabled is False
    assert cm.checkpoints_dir is None
    assert cm.loss_file is None


def test_update_loss_file_standard_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manager_mod, "Paths", FakePaths)
    run_dir = tmp_path / "run_standard"

    cm = manager_mod.CheckpointManager(run_dir=run_dir, saving_prm={"enabled": True}, paths=FakePaths(None), is_lvae=False)
    cm.update_loss_file(epoch=0, train_metrics={"loss": 1.0}, val_metrics={"loss": 2.0})
    cm.update_loss_file(epoch=1, train_metrics={"loss": 0.8}, val_metrics={"loss": 1.5})

    lines = cm.loss_file.read_text(encoding="utf-8").strip().splitlines()

    assert "Epoch" in lines[0]
    assert "Train_loss" in lines[0]
    assert "Val_loss" in lines[0]
    assert len(lines) == 3
    assert "0" in lines[1]
    assert "1" in lines[2]


def test_update_loss_file_lvae_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(manager_mod, "Paths", FakePaths)
    run_dir = tmp_path / "run_lvae"

    cm = manager_mod.CheckpointManager(run_dir=run_dir, saving_prm={"enabled": True}, paths=FakePaths(None), is_lvae=True)
    cm.update_loss_file(
        epoch=3,
        train_metrics={"loss": 0.5, "recons_loss": 0.3, "kl_loss": 0.2},
        val_metrics={"loss": 0.6},
    )

    lines = cm.loss_file.read_text(encoding="utf-8").strip().splitlines()

    assert "Recons_Loss" in lines[0]
    assert "KL_Loss" in lines[0]
    assert "0.3" in lines[1]
    assert "0.2" in lines[1]


def test_save_state_dict_writes_best_and_last_and_cleans_best_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(manager_mod, "Paths", FakePaths)
    saved = []

    def fake_torch_save(obj, path):
        payload = dict(obj) if isinstance(obj, dict) else obj
        saved.append((payload, Path(path).name))

    monkeypatch.setattr(manager_mod.torch, "save", fake_torch_save)

    cm = manager_mod.CheckpointManager(
        run_dir=tmp_path / "run_state",
        paths=FakePaths(None),
        saving_prm={"enabled": True, "state_dict": True, "entire_model": False, "overwrite_best": True},
    )
    state_dict = {"epoch": 5, "model_state_dict": {"w": 1}}

    cm.save(state_dict=state_dict, model=object(), best_loss=0.123, is_best=True)

    names = [name for _, name in saved]
    assert "model_best_state_dict.pt" in names
    assert "model_last_state_dict.pt" in names

    last_payload = next(payload for payload, name in saved if name == "model_last_state_dict.pt")
    assert last_payload["best_loss"] == 0.123
    assert "best_loss" not in state_dict


def test_save_entire_model_uses_epoch_name_when_not_overwriting_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(manager_mod, "Paths", FakePaths)
    saved_names = []

    def fake_torch_save(obj, path):
        saved_names.append(Path(path).name)

    monkeypatch.setattr(manager_mod.torch, "save", fake_torch_save)

    cm = manager_mod.CheckpointManager(
        run_dir=tmp_path / "run_full",
        paths=FakePaths(None),
        saving_prm={"enabled": True, "state_dict": False, "entire_model": True, "overwrite_best": False},
    )

    cm.save(state_dict={"epoch": 2}, model=object(), best_loss=1.0, is_best=True)

    assert "model_epoch_2.pt" in saved_names
    assert "model_last.pt" in saved_names
