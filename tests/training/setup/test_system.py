from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import lisai.training.runtime as runtime_mod
import lisai.training.setup.run_dir as run_dir_mod


class FakeLogger:
    def __init__(self):
        self.info_calls = []
        self.warning_calls = []

    def info(self, msg):
        self.info_calls.append(msg)

    def warning(self, msg):
        self.warning_calls.append(msg)


def _make_cfg(
    *,
    architecture: str = "unet",
    progress_bar: bool = False,
    device: str = "cpu",
    tensorboard_enabled: bool = False,
    tensorboard_subfolder: str | None = None,
    models_subfolder: str = "runs",
    validation_images: bool = True,
    validation_freq: int = 10,
):
    return SimpleNamespace(
        experiment=SimpleNamespace(exp_name="exp_raw", mode="train"),
        model=SimpleNamespace(architecture=architecture),
        training=SimpleNamespace(progress_bar=progress_bar, device=device),
        tensorboard=SimpleNamespace(enabled=tensorboard_enabled),
        data=SimpleNamespace(dataset_name="dataset_x"),
        routing=SimpleNamespace(
            tensorboard_subfolder=tensorboard_subfolder,
            models_subfolder=models_subfolder,
        ),
        saving=SimpleNamespace(
            validation_images=validation_images,
            validation_freq=validation_freq,
        ),
    )


def test_initialize_runtime_falls_back_to_cpu_and_builds_validation_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run_a"
    captured = {}
    fake_logger = FakeLogger()
    console_filter = SimpleNamespace(enable=True)
    file_filter = SimpleNamespace(enable=True)

    class FakePaths:
        def log_file_path(self, *, run_dir):
            return Path(run_dir) / "train.log"

        def validation_images_dir(self, *, run_dir):
            return Path(run_dir) / "validation_images"

        def tensorboard_dir(self, *, dataset_name, tensorboard_subfolder):
            return tmp_path / "tensorboard" / dataset_name / tensorboard_subfolder

    def fake_setup_logger(
        *,
        name,
        level,
        log_file,
        use_tqdm,
        file_format=None,
        file_datefmt=None,
        file_enabled=True,
        **_,
    ):
        captured["setup_logger"] = {
            "name": name,
            "level": level,
            "log_file": Path(log_file),
            "use_tqdm": use_tqdm,
            "file_format": file_format,
            "file_datefmt": file_datefmt,
            "file_enabled": file_enabled,
        }
        return fake_logger, console_filter, file_filter

    monkeypatch.setattr(runtime_mod, "Paths", lambda _settings: FakePaths())
    monkeypatch.setattr(run_dir_mod, "prepare_run_dir", lambda cfg, ctx: (run_dir, "exp_unique"))
    monkeypatch.setattr(runtime_mod, "setup_logger", fake_setup_logger)
    monkeypatch.setattr(runtime_mod.torch.cuda, "is_available", lambda: False)

    cfg = _make_cfg(
        architecture="unet3d",
        progress_bar=True,
        device="cuda",
        tensorboard_enabled=False,
        validation_images=True,
        validation_freq=7,
    )

    ctx = runtime_mod.initialize_runtime(cfg)

    assert ctx.run_name == "exp_unique"
    assert ctx.run_dir == run_dir
    assert str(ctx.device) == "cpu"
    assert ctx.writer is None
    assert len(ctx.callbacks) == 1
    assert isinstance(ctx.callbacks[0], runtime_mod.ValidationImagesCallback)
    assert ctx.callbacks[0].freq == 7
    assert ctx.callbacks[0].volumetric is True

    assert captured["setup_logger"]["name"] == "lisai"
    assert captured["setup_logger"]["log_file"] == run_dir / "train.log"
    assert captured["setup_logger"]["use_tqdm"] is True
    assert captured["setup_logger"]["file_format"] == "%(asctime)s %(message)s"
    assert captured["setup_logger"]["file_datefmt"] == "%Y-%m-%d %H:%M:%S"
    assert captured["setup_logger"]["file_enabled"] is False

    assert any("CUDA requested but not available" in msg for msg in fake_logger.warning_calls)

    ctx.enable_console_logs(False)
    ctx.enable_file_logs(False)
    assert console_filter.enable is False
    assert file_filter.enable is False


def test_initialize_runtime_creates_tensorboard_writer_and_tensorboard_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run_b"
    tb_root = tmp_path / "tb_root"
    tb_folder = tb_root / "exp_unique"
    captured = {}
    fake_logger = FakeLogger()
    console_filter = SimpleNamespace(enable=True)
    file_filter = SimpleNamespace(enable=True)

    class FakeSummaryWriter:
        def __init__(self, log_dir):
            self.log_dir = log_dir

    fake_tb_module = types.ModuleType("torch.utils.tensorboard")
    fake_tb_module.SummaryWriter = FakeSummaryWriter

    class FakePaths:
        def log_file_path(self, *, run_dir):
            return Path(run_dir) / "train.log"

        def validation_images_dir(self, *, run_dir):
            return Path(run_dir) / "validation_images"

        def tensorboard_dir(self, *, dataset_name, tensorboard_subfolder):
            captured["tensorboard_dir"] = {
                "dataset_name": dataset_name,
                "tensorboard_subfolder": tensorboard_subfolder,
            }
            return tb_root

    def fake_setup_logger(
        *,
        name,
        level,
        log_file,
        use_tqdm,
        file_format=None,
        file_datefmt=None,
        file_enabled=True,
        **_,
    ):
        captured["setup_logger"] = {
            "name": name,
            "level": level,
            "log_file": Path(log_file),
            "use_tqdm": use_tqdm,
            "file_format": file_format,
            "file_datefmt": file_datefmt,
            "file_enabled": file_enabled,
        }
        return fake_logger, console_filter, file_filter

    def fake_create_tb_folder(root, exp_name, exist_ok):
        captured["create_tb_folder"] = {
            "root": Path(root),
            "exp_name": exp_name,
            "exist_ok": exist_ok,
        }
        return tb_folder, "exp_unique"

    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tb_module)
    monkeypatch.setattr(runtime_mod, "Paths", lambda _settings: FakePaths())
    monkeypatch.setattr(run_dir_mod, "prepare_run_dir", lambda cfg, ctx: (run_dir, "exp_unique"))
    monkeypatch.setattr(runtime_mod, "setup_logger", fake_setup_logger)
    monkeypatch.setattr(runtime_mod, "create_tb_folder", fake_create_tb_folder)
    monkeypatch.setattr(runtime_mod.torch.cuda, "is_available", lambda: True)

    cfg = _make_cfg(
        architecture="unet",
        progress_bar=False,
        device="cpu",
        tensorboard_enabled=True,
        tensorboard_subfolder="",
        models_subfolder="models_fallback",
        validation_images=True,
    )

    ctx = runtime_mod.initialize_runtime(cfg)

    assert isinstance(ctx.writer, FakeSummaryWriter)
    assert Path(ctx.writer.log_dir) == tb_folder
    assert len(ctx.callbacks) == 2
    assert isinstance(ctx.callbacks[0], runtime_mod.TensorBoardCallback)
    assert isinstance(ctx.callbacks[1], runtime_mod.ValidationImagesCallback)

    assert captured["tensorboard_dir"] == {
        "dataset_name": "dataset_x",
        "tensorboard_subfolder": "models_fallback",
    }
    assert captured["create_tb_folder"] == {
        "root": tb_root,
        "exp_name": "exp_unique",
        "exist_ok": True,
    }
    assert captured["setup_logger"]["file_enabled"] is False
