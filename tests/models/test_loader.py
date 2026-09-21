from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from lisai.models import loader
from lisai.models.params import AnyModelParams, UNetParams


@dataclass(frozen=True)
class DummyTrainingSpec:
    architecture: str
    parameters: AnyModelParams
    mode: str
    patch_size: int | None = None
    downsamp_factor: int | None = 1
    origin_run_dir: Path | None = None
    checkpoint_method: str | None = None
    checkpoint_selector: str | None = None
    checkpoint_epoch: int | None = None
    checkpoint_filename: str | None = None


class DummyModel:
    def __init__(self):
        self.loaded_state = None
        self.device = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        self.device = device
        return self



def _base_spec(**overrides) -> DummyTrainingSpec:
    data = {
        "architecture": "unet",
        "parameters": UNetParams(),
        "mode": "train",
    }
    data.update(overrides)
    return DummyTrainingSpec(**data)



def test_origin_checkpoint_path_prefers_explicit_filename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = {}

    class FakePaths:
        def __init__(self):
            calls["init_called"] = True

        def checkpoint_path(self, **kwargs):
            calls["kwargs"] = kwargs
            return Path(kwargs["run_dir"]) / "checkpoints" / kwargs["model_name"]

    monkeypatch.setattr(loader, "Paths", FakePaths)

    spec = _base_spec(
        mode="continue_training",
        origin_run_dir=tmp_path / "origin",
        checkpoint_filename="manual_checkpoint.pt",
        checkpoint_method="state_dict",
        checkpoint_selector="last",
    )

    out = loader._origin_checkpoint_path(spec)

    assert out.name == "manual_checkpoint.pt"
    assert calls["init_called"] is True
    assert calls["kwargs"]["model_name"] == "manual_checkpoint.pt"
    assert "load_method" not in calls["kwargs"]


def test_origin_checkpoint_path_best_falls_back_to_highest_epoch_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class FakePaths:
        def checkpoint_path(self, **kwargs):
            run_dir = Path(kwargs["run_dir"])
            load_method = kwargs["load_method"]
            if kwargs.get("epoch_number") is not None:
                middle = f"epoch_{kwargs['epoch_number']}"
            else:
                middle = kwargs["best_or_last"]
            if load_method == "state_dict":
                name = f"model_{middle}_state_dict.pt"
            else:
                name = f"model_{middle}.pt"
            return run_dir / "checkpoints" / name

    monkeypatch.setattr(loader, "Paths", FakePaths)
    checkpoints = tmp_path / "origin" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "model_epoch_3_state_dict.pt").write_bytes(b"old-best")
    expected = checkpoints / "model_epoch_8_state_dict.pt"
    expected.write_bytes(b"new-best")

    spec = _base_spec(
        mode="retrain",
        origin_run_dir=tmp_path / "origin",
        checkpoint_method="state_dict",
        checkpoint_selector="best",
    )

    assert loader._origin_checkpoint_path(spec) == expected



def test_prepare_model_for_training_requires_architecture():
    spec = _base_spec(architecture="")

    with pytest.raises(ValueError, match="architecture"):
        loader.prepare_model_for_training(spec=spec, device=torch.device("cpu"))



def test_prepare_model_for_training_full_model_load_returns_loaded_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    checkpoint = tmp_path / "model_full.pt"
    checkpoint.write_bytes(b"test")

    loaded_model = object()
    spec = _base_spec(
        mode="continue_training",
        origin_run_dir=tmp_path,
        checkpoint_method="full_model",
    )

    monkeypatch.setattr(loader, "_origin_checkpoint_path", lambda _spec: checkpoint)
    monkeypatch.setattr(
        loader,
        "init_model",
        lambda **kwargs: pytest.fail("init_model should not be called for full_model load"),
    )
    monkeypatch.setattr(loader.torch, "load", lambda path, map_location: loaded_model)

    model, state = loader.prepare_model_for_training(
        spec=spec,
        device=torch.device("cpu"),
        model_norm_prm={"data_mean": 0.0},
    )

    assert model is loaded_model
    assert state is None



def test_prepare_model_for_training_loads_model_state_dict_and_returns_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    checkpoint = tmp_path / "model_state_dict.pt"
    checkpoint.write_bytes(b"test")

    model = DummyModel()
    loaded_state = {"model_state_dict": {"weights": torch.tensor([1.0])}, "epoch": 7}
    spec = _base_spec(
        mode="continue_training",
        origin_run_dir=tmp_path,
        checkpoint_method="state_dict",
    )

    monkeypatch.setattr(loader, "_origin_checkpoint_path", lambda _spec: checkpoint)
    monkeypatch.setattr(loader, "init_model", lambda **kwargs: model)
    monkeypatch.setattr(loader.torch, "load", lambda path, map_location, **kwargs: loaded_state)

    out_model, out_state = loader.prepare_model_for_training(
        spec=spec,
        device=torch.device("cpu"),
        model_norm_prm={"data_mean": 0.0},
    )

    assert out_model is model
    assert model.loaded_state == loaded_state["model_state_dict"]
    assert out_state == loaded_state



def test_prepare_model_for_training_plain_state_dict_returns_none_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    checkpoint = tmp_path / "plain_state_dict.pt"
    checkpoint.write_bytes(b"test")

    model = DummyModel()
    plain_state_dict = {"weights": torch.tensor([2.0])}
    spec = _base_spec(
        mode="continue_training",
        origin_run_dir=tmp_path,
        checkpoint_method="state_dict",
    )

    monkeypatch.setattr(loader, "_origin_checkpoint_path", lambda _spec: checkpoint)
    monkeypatch.setattr(loader, "init_model", lambda **kwargs: model)
    monkeypatch.setattr(loader.torch, "load", lambda path, map_location, **kwargs: plain_state_dict)

    out_model, out_state = loader.prepare_model_for_training(
        spec=spec,
        device=torch.device("cpu"),
        model_norm_prm={"data_mean": 0.0},
    )

    assert out_model is model
    assert model.loaded_state == plain_state_dict
    assert out_state is None
