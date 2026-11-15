from __future__ import annotations

from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.training.cli as training_cli
from lisai.config.io.config_paths import ConfigPathResolver
from lisai.config.settings import load_default_settings


def test_resolve_config_path_supports_training_subpath():
    expected = (load_default_settings().TRAINING_CONFIG_DIR / "examples" / "vim_denoising_unet.yml").resolve()

    assert training_cli.resolve_config_path("examples/vim_denoising_unet.yml", resolver=ConfigPathResolver("training", stg=load_default_settings())) == expected


def test_resolve_config_path_supports_training_subpath_without_extension():
    expected = (load_default_settings().TRAINING_CONFIG_DIR / "examples" / "vim_denoising_unet.yml").resolve()

    assert training_cli.resolve_config_path("examples/vim_denoising_unet", resolver=ConfigPathResolver("training", stg=load_default_settings())) == expected


def test_resolve_config_path_lists_available_configs_when_missing():
    with pytest.raises(FileNotFoundError, match="Training config not found: missing_training_config") as exc_info:
        training_cli.resolve_config_path("missing_training_config", resolver=ConfigPathResolver("training", stg=load_default_settings()))

    message = str(exc_info.value)
    assert "Available configs:" in message
    assert "examples/vim_denoising_unet.yml" in message
    assert "presets/denoising_care.yml" in message


def test_resolve_config_path_refuses_presets_and_templates():
    with pytest.raises(ValueError, match="must be instantiated"):
        training_cli.resolve_config_path("presets/denoising_hdn_unsup", resolver=ConfigPathResolver("training", stg=load_default_settings()))

    with pytest.raises(ValueError, match="must be instantiated"):
        training_cli.resolve_config_path("templates/base_training", resolver=ConfigPathResolver("training", stg=load_default_settings()))


def test_training_cli_main_accepts_config_flag(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path, *, settings=None):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["--config", "examples/vim_denoising_unet"])

    assert exit_code == 0
    assert captured["config_path"].name == "vim_denoising_unet.yml"


def test_training_cli_main_accepts_extensionless_config_name(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path, *, settings=None):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["examples/vim_denoising_unet"])

    assert exit_code == 0
    assert captured["config_path"].name == "vim_denoising_unet.yml"


def test_training_cli_passes_progress_bar_override(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path, *, progress_bar=None, settings=None):
        captured["config_path"] = config_path
        captured["progress_bar"] = progress_bar

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = training_cli.main(["examples/vim_denoising_unet", "--no-progress-bar"])

    assert exit_code == 0
    assert captured["config_path"].name == "vim_denoising_unet.yml"
    assert captured["progress_bar"] is False


def test_root_cli_train_dispatches_extensionless_config_to_training(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_run_training(config_path, *, settings=None):
        captured["config_path"] = config_path

    monkeypatch.setattr(training_cli, "run_training", fake_run_training)

    exit_code = root_cli.main(["train", "examples/vim_denoising_unet"])

    assert exit_code == 0
    assert captured["config_path"].name == "vim_denoising_unet.yml"
