from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest

from lisai.config.io.yaml import save_yaml
from lisai.config.settings import Settings


@pytest.fixture
def settings_factory(tmp_path):
    """Isolated legacy configuration trees, without machine-specific roots."""
    source = Path(__file__).resolve().parents[2] / "configs"

    def create(name: str, *, data_root: Path | None = None) -> Settings:
        config_dir = tmp_path / name / "configs"
        config_dir.mkdir(parents=True)
        for filename in ("project_config.yml", "data_config.yml"):
            copyfile(source / filename, config_dir / filename)
        save_yaml(
            {"infrastructure": {"data_root": str(data_root or (tmp_path / name / "data"))}},
            config_dir / "local_config.yml",
        )
        return Settings.from_config_dir(config_dir)

    return create
