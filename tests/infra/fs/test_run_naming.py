from __future__ import annotations

from pathlib import Path

import pytest

from lisai.config import settings
from lisai.infra.fs.folders import create_run_dir
from lisai.infra.fs.run_naming import allocate_run_dir_name, format_run_dir_name


def test_allocate_run_dir_name_starts_at_zero_and_increments(tmp_path: Path):
    runs_root = tmp_path / "runs"
    (runs_root / "demo_00").mkdir(parents=True, exist_ok=True)
    (runs_root / "demo_01").mkdir(parents=True, exist_ok=True)
    (runs_root / "demo_05").mkdir(parents=True, exist_ok=True)
    (runs_root / "demo_extra").mkdir(parents=True, exist_ok=True)

    name, idx = allocate_run_dir_name(runs_root, "demo")

    assert idx == 6
    assert name == "demo_06"


def test_format_run_dir_name_uses_configurable_width(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.project_cfg.naming, "run_dir_index_width", 3)
    assert format_run_dir_name("abc", 1) == "abc_001"


def test_create_run_dir_uses_name_plus_index(tmp_path: Path):
    class FakePaths:
        def __init__(self, root: Path):
            self.root = root

        def run_dir(self, *, dataset_name: str, models_subfolder: str, exp_name: str) -> Path:
            return self.root / dataset_name / "runs" / models_subfolder / exp_name

    paths = FakePaths(tmp_path / "datasets")

    run_a, name_a = create_run_dir(paths=paths, ds_name="Gag", exp_name="demo", subfolder="Upsamp")
    run_b, name_b = create_run_dir(paths=paths, ds_name="Gag", exp_name="demo", subfolder="Upsamp")

    assert name_a == "demo_00"
    assert name_b == "demo_01"
    assert run_a.name == "demo_00"
    assert run_b.name == "demo_01"
