from __future__ import annotations

from pathlib import Path

from lisai.config.io.yaml import load_yaml
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
