from __future__ import annotations

from pathlib import Path

from lisai.config.io.yaml import load_yaml, save_yaml
from lisai.config.models.local_config import LocalConfig
from lisai.config.settings import Settings


def test_existing_local_config_without_inference_section_uses_defaults():
    cfg = LocalConfig.model_validate(
        {
            "infrastructure": {
                "data_root": "/tmp/lisai",
            }
        }
    )

    assert cfg.inference.output_mode == "default"
    assert cfg.inference.inference_dir == "default"


def test_local_inference_output_mode_accepts_source_relative_modes():
    folder_inside = LocalConfig.model_validate(
        {
            "infrastructure": {"data_root": "/tmp/lisai"},
            "inference": {"output_mode": "folder_inside"},
        }
    )
    folder_outside = LocalConfig.model_validate(
        {
            "infrastructure": {"data_root": "/tmp/lisai"},
            "inference": {"output_mode": "folder_outside"},
        }
    )

    assert folder_inside.inference.output_mode == "folder_inside"
    assert folder_outside.inference.output_mode == "folder_outside"


def test_first_time_setup_writes_inference_defaults(monkeypatch, tmp_path: Path):
    settings = Settings.__new__(Settings)
    settings._local_yaml_path = tmp_path / "configs" / "local_config.yml"
    data_root = tmp_path / "data"
    monkeypatch.setattr("builtins.input", lambda _prompt: str(data_root))

    raw = settings._load_or_setup_infrastructure()

    assert raw["inference"] == {
        "output_mode": "default",
        "inference_dir": "default",
    }
    written = load_yaml(settings._local_yaml_path)
    assert written["inference"] == raw["inference"]
    assert settings._local_yaml_path.read_text(encoding="utf-8").startswith(
        "# yaml-language-server: $schema=./schema/local-config.schema.json\n"
    )


def test_existing_local_config_gets_schema_hint_without_rewriting_values(tmp_path: Path):
    settings = Settings.__new__(Settings)
    settings._local_yaml_path = tmp_path / "configs" / "local_config.yml"
    settings._local_yaml_path.parent.mkdir(parents=True)
    expected = {
        "infrastructure": {"data_root": "/tmp/lisai"},
        "inference": {"output_mode": "folder_inside", "inference_dir": "default"},
    }
    save_yaml(expected, settings._local_yaml_path)

    loaded = settings._load_or_setup_infrastructure()

    assert loaded == expected
    assert settings._local_yaml_path.read_text(encoding="utf-8").startswith(
        "# yaml-language-server: $schema=./schema/local-config.schema.json\n"
    )


def test_local_inference_dir_overrides_project_root(tmp_path: Path):
    from lisai.config import settings as global_settings

    settings = Settings.__new__(Settings)
    settings.PROJECT_ROOT = tmp_path
    settings.project_cfg = global_settings.project_cfg
    settings.local_cfg = LocalConfig.model_validate(
        {
            "infrastructure": {"data_root": str(tmp_path / "data")},
            "inference": {
                "output_mode": "default",
                "inference_dir": str(tmp_path / "custom_predictions"),
            },
        }
    )

    ctx = settings._build_context()

    assert Path(ctx.paths.roots.inference_dir) == (tmp_path / "custom_predictions").resolve()


def test_default_local_inference_dir_uses_project_inference_root(tmp_path: Path):
    from lisai.config import settings as global_settings

    settings = Settings.__new__(Settings)
    settings.PROJECT_ROOT = tmp_path
    settings.project_cfg = global_settings.project_cfg
    settings.local_cfg = LocalConfig.model_validate(
        {
            "infrastructure": {"data_root": str(tmp_path / "data")},
            "inference": {
                "output_mode": "default",
                "inference_dir": "default",
            },
        }
    )

    ctx = settings._build_context()

    assert Path(ctx.paths.roots.inference_dir) == (tmp_path / "data" / "inference").resolve()
