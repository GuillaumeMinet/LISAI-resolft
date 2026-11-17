from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.models.local_config import LocalConfig, LegacyLocalConfig
from lisai.config.settings import Settings, prepare_legacy_config_dir


def test_existing_local_config_without_inference_section_uses_defaults():
    cfg = LegacyLocalConfig.model_validate(
        {
            "infrastructure": {
                "data_root": "/tmp/lisai",
            }
        }
    )

    assert cfg.inference.output.mode == "default"
    assert cfg.inference.output.save_input_mode == "if_not_in_place"
    assert cfg.inference.inference_dir == "default"
    assert cfg.console.progress_bar is None


def test_local_inference_output_mode_accepts_source_relative_modes():
    folder_inside = LegacyLocalConfig.model_validate(
        {
            "infrastructure": {"data_root": "/tmp/lisai"},
            "inference": {"output": {"mode": "folder_inside"}},
        }
    )
    folder_outside = LegacyLocalConfig.model_validate(
        {
            "infrastructure": {"data_root": "/tmp/lisai"},
            "inference": {"output": {"mode": "folder_outside"}},
        }
    )

    assert folder_inside.inference.output.mode == "folder_inside"
    assert folder_outside.inference.output.mode == "folder_outside"


def test_first_time_setup_writes_inference_defaults(monkeypatch, tmp_path: Path):
    local_path = tmp_path / "configs" / "local_config.yml"
    data_root = tmp_path / "data"
    monkeypatch.setattr("builtins.input", lambda _prompt: str(data_root))

    raw = prepare_legacy_config_dir(local_path.parent, interactive=True)

    assert raw["inference"] == {
        "output": {
            "mode": "default",
            "save_input_mode": "if_not_in_place",
        },
        "inference_dir": "default",
    }
    assert raw["console"] == {"progress_bar": None}
    written = load_yaml(local_path)
    assert written["inference"] == raw["inference"]
    assert written["console"] == raw["console"]
    assert local_path.read_text(encoding="utf-8").startswith(
        "# yaml-language-server: $schema=./schema/legacy-local-config.schema.json\n"
    )


def test_existing_flat_output_mode_is_migrated_to_nested_output(tmp_path: Path):
    local_path = tmp_path / "configs" / "local_config.yml"
    local_path.parent.mkdir(parents=True)
    save_yaml(
        {
            "infrastructure": {"data_root": "/tmp/lisai"},
            "inference": {
                "output_mode": "folder_inside",
                "inference_dir": "default",
            },
        },
        local_path,
    )

    loaded = prepare_legacy_config_dir(local_path.parent)

    assert loaded["inference"] == {
        "output": {
            "mode": "folder_inside",
            "save_input_mode": "if_not_in_place",
        },
        "inference_dir": "default",
    }
    assert load_yaml(local_path)["inference"] == loaded["inference"]


def test_existing_local_config_gets_schema_hint_without_rewriting_values(tmp_path: Path):
    local_path = tmp_path / "configs" / "local_config.yml"
    local_path.parent.mkdir(parents=True)
    expected = {
        "infrastructure": {"data_root": "/tmp/lisai"},
        "inference": {
            "output": {"mode": "folder_inside", "save_input_mode": "never"},
            "inference_dir": "default",
        },
    }
    save_yaml(expected, local_path)

    loaded = prepare_legacy_config_dir(local_path.parent)

    assert loaded == expected
    assert local_path.read_text(encoding="utf-8").startswith(
        "# yaml-language-server: $schema=./schema/legacy-local-config.schema.json\n"
    )


def test_local_inference_dir_overrides_project_root(tmp_path: Path, global_settings):
    settings = Settings.__new__(Settings)
    settings.WORKSPACE_ROOT = tmp_path / "data"
    settings.project_cfg = global_settings.project_cfg
    settings.local_cfg = LocalConfig.model_validate(
        {
            "inference": {
                "output": {"mode": "default"},
                "inference_dir": str(tmp_path / "custom_predictions"),
            },
        }
    )

    ctx = settings._build_context()

    assert Path(ctx.paths.roots.inference_dir) == (tmp_path / "custom_predictions").resolve()


def test_default_local_inference_dir_uses_project_inference_root(tmp_path: Path, global_settings):
    settings = Settings.__new__(Settings)
    settings.WORKSPACE_ROOT = tmp_path / "data"
    settings.project_cfg = global_settings.project_cfg
    settings.local_cfg = LocalConfig.model_validate(
        {
            "inference": {
                "output": {"mode": "default"},
                "inference_dir": "default",
            },
        }
    )

    ctx = settings._build_context()

    assert Path(ctx.paths.roots.inference_dir) == (tmp_path / "data" / "inference").resolve()


def test_local_inference_configs_are_created_from_builtin_defaults(tmp_path: Path):
    config_dir = tmp_path / "configs"
    prepare_legacy_config_dir(config_dir, data_root=tmp_path / "data")

    defaults_path = config_dir / "inference" / "local" / "defaults.yml"
    post_training_path = config_dir / "inference" / "local" / "post_training.yml"
    defaults = load_yaml(defaults_path)
    post_training = load_yaml(post_training_path)

    assert defaults["apply"]["inference"]["tiling_size"] == "auto"
    assert defaults["apply"]["postprocess"]["color_code"]["enabled"] is False
    assert defaults["evaluate"]["checkpoint"]["best_or_last"] == "best"
    assert post_training["evaluate"]["checkpoint"]["best_or_last"] == "both"
    assert post_training["evaluate"]["metrics"] == ["psnr", "ssim"]
    assert post_training["evaluate"]["saving"]["overwrite"] is True


def test_local_inference_configs_are_never_overwritten(tmp_path: Path):
    config_dir = tmp_path / "configs"
    local_dir = config_dir / "inference" / "local"
    local_dir.mkdir(parents=True)
    defaults_path = local_dir / "defaults.yml"
    post_training_path = local_dir / "post_training.yml"
    save_yaml({"apply": {"inference": {"tiling_size": 777}}}, defaults_path)
    save_yaml({"evaluate": {"metrics": ["custom"]}}, post_training_path)

    prepare_legacy_config_dir(config_dir, data_root=tmp_path / "data")

    assert load_yaml(defaults_path)["apply"]["inference"]["tiling_size"] == 777
    assert load_yaml(post_training_path)["evaluate"]["metrics"] == ["custom"]


def test_explicit_loader_does_not_run_legacy_migration(tmp_path: Path):
    from shutil import copyfile

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    repo_configs = Path(__file__).resolve().parents[1] / "fixtures/legacy_configs"
    for name in ("project_config.yml", "data_config.yml"):
        copyfile(repo_configs / name, config_dir / name)
    local_path = config_dir / "local_config.yml"
    save_yaml(
        {"infrastructure": {"data_root": str(tmp_path / "data")},
         "inference": {"output_mode": "folder_inside"}},
        local_path,
    )
    original = local_path.read_bytes()
    with pytest.raises(ValidationError, match="output_mode"):
        Settings.from_config_dir(config_dir)
    assert local_path.read_bytes() == original
    prepare_legacy_config_dir(config_dir)
    assert Settings.from_config_dir(config_dir).INFERENCE_OUTPUT_MODE == "folder_inside"


def test_missing_local_config_requires_explicit_setup(tmp_path: Path, monkeypatch):
    def forbid_prompt(*_args):
        raise AssertionError("unexpected input prompt")

    monkeypatch.setattr("builtins.input", forbid_prompt)
    config_dir = tmp_path / "configs"
    with pytest.raises(FileNotFoundError, match="local_config.yml"):
        prepare_legacy_config_dir(config_dir)
    assert not config_dir.exists()
