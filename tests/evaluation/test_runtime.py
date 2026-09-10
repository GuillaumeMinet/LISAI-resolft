from __future__ import annotations

from pathlib import Path

import pytest
import torch

import lisai.evaluation.runtime as runtime_mod
from lisai.evaluation.saved_run import SavedTrainingRun
from lisai.models.params import UNetParams


class FakePaths:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path_value = checkpoint_path
        self.calls = []

    def checkpoint_path(self, **kwargs):
        self.calls.append(kwargs)
        return self.checkpoint_path_value


class SelectorAwarePaths:
    def __init__(self, *, best_path: Path, last_path: Path):
        self.best_path = best_path
        self.last_path = last_path
        self.calls = []

    def checkpoint_path(self, **kwargs):
        self.calls.append(kwargs)
        selector = kwargs.get("best_or_last")
        if selector == "best":
            return self.best_path
        if selector == "last":
            return self.last_path
        raise AssertionError(f"Unexpected checkpoint selector: {selector}")



def _make_saved_run(*, checkpoint_methods=('state_dict',), default_tiling_size=128) -> SavedTrainingRun:
    return SavedTrainingRun(
        run_dir=Path('/runs/dataset_a/exp_a'),
        experiment_name='exp_a',
        dataset_name='dataset_a',
        data_subfolder='raw',
        data_cfg={'dataset_name': 'dataset_a', 'patch_size': 64},
        model_architecture='unet',
        model_parameters=UNetParams(),
        data_norm_prm={'clip': 0},
        model_norm_prm={'data_mean': 1.0},
        noise_model_name=None,
        checkpoint_methods=checkpoint_methods,
        patch_size=64,
        downsamp_factor=1,
        upsampling_factor=1,
        context_length=None,
        default_tiling_size=default_tiling_size,
    )


def test_resolve_tiling_size_uses_saved_default_for_auto_and_null():
    saved_run = _make_saved_run(default_tiling_size=300)

    assert runtime_mod.resolve_tiling_size(saved_run, "auto") == 300
    assert runtime_mod.resolve_tiling_size(saved_run, None) == 300


def test_resolve_tiling_size_supports_forced_size_and_off():
    saved_run = _make_saved_run(default_tiling_size=300)

    assert runtime_mod.resolve_tiling_size(saved_run, 2000) == 2000
    assert runtime_mod.resolve_tiling_size(saved_run, "512") == 512
    assert runtime_mod.resolve_tiling_size(saved_run, "off") is None


@pytest.mark.parametrize("tiling_size", [0, -1, True, "small"])
def test_resolve_tiling_size_rejects_invalid_values(tiling_size):
    with pytest.raises(ValueError, match="tiling_size"):
        runtime_mod.resolve_tiling_size(_make_saved_run(), tiling_size)



def test_initialize_runtime_builds_inference_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint_path = tmp_path / 'checkpoint.pt'
    checkpoint_path.write_text('ok', encoding='utf-8')
    fake_paths = FakePaths(checkpoint_path)
    model_obj = object()

    monkeypatch.setattr(runtime_mod, 'Paths', lambda _settings: fake_paths)
    monkeypatch.setattr(
        runtime_mod,
        '_load_state_dict_model',
        lambda saved_run, checkpoint_path, device, paths: (model_obj, 17),
    )

    runtime = runtime_mod.initialize_runtime(
        saved_run=_make_saved_run(),
        device='cpu',
        best_or_last='best',
        epoch_number=None,
        tiling_size=None,
    )

    assert runtime.device == torch.device('cpu')
    assert runtime.model is model_obj
    assert runtime.checkpoint_path == checkpoint_path
    assert runtime.load_method == 'state_dict'
    assert runtime.tiling_size == 128
    assert runtime.resolved_epoch == 17
    assert fake_paths.calls == [
        {
            'run_dir': Path('/runs/dataset_a/exp_a'),
            'load_method': 'state_dict',
            'best_or_last': 'best',
        }
    ]



def test_initialize_runtime_loads_full_model_and_applies_tiling_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint_path = tmp_path / 'model_epoch_9.pt'
    checkpoint_path.write_text('ok', encoding='utf-8')
    fake_paths = FakePaths(checkpoint_path)

    class FakeModel:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

    model_obj = FakeModel()

    monkeypatch.setattr(runtime_mod, 'Paths', lambda _settings: fake_paths)
    monkeypatch.setattr(runtime_mod.torch, 'load', lambda path, map_location: model_obj)

    runtime = runtime_mod.initialize_runtime(
        saved_run=_make_saved_run(checkpoint_methods=('full_model',), default_tiling_size=128),
        device='cpu',
        tiling_size=256,
    )

    assert runtime.model is model_obj
    assert runtime.load_method == 'full_model'
    assert runtime.tiling_size == 256
    assert runtime.resolved_epoch == 9
    assert model_obj.evaluated is True


def test_initialize_runtime_with_both_selector_falls_back_to_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    best_path = tmp_path / "model_best_state_dict.pt"
    last_path = tmp_path / "model_last_state_dict.pt"
    last_path.write_text("ok", encoding="utf-8")
    fake_paths = SelectorAwarePaths(best_path=best_path, last_path=last_path)
    model_obj = object()

    monkeypatch.setattr(runtime_mod, "Paths", lambda _settings: fake_paths)
    monkeypatch.setattr(
        runtime_mod,
        "_load_state_dict_model",
        lambda saved_run, checkpoint_path, device, paths: (model_obj, 11),
    )

    runtime = runtime_mod.initialize_runtime(
        saved_run=_make_saved_run(),
        device="cpu",
        best_or_last="both",
        epoch_number=None,
        tiling_size=None,
    )

    assert runtime.model is model_obj
    assert runtime.checkpoint_path == last_path
    assert fake_paths.calls == [
        {
            "run_dir": Path("/runs/dataset_a/exp_a"),
            "load_method": "state_dict",
            "best_or_last": "best",
        },
        {
            "run_dir": Path("/runs/dataset_a/exp_a"),
            "load_method": "state_dict",
            "best_or_last": "last",
        },
    ]
