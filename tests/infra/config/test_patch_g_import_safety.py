"""M1 acceptance: imports and parser help must never initialize a workspace."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_HELP_SCRIPT = r'''
import builtins
import importlib
from pathlib import Path


def fail_input(*_args, **_kwargs):
    raise AssertionError("LISAI import/help attempted interactive setup")


_original_path_open = Path.open


def checked_open(self, *args, **kwargs):
    path = str(self).replace("\\", "/")
    if "/configs/" in path and path.endswith((".yml", ".yaml")):
        raise AssertionError(f"LISAI import/help read configuration: {path}")
    return _original_path_open(self, *args, **kwargs)


builtins.input = fail_input
Path.open = checked_open

import lisai
import lisai.api
import lisai.config
import lisai.config.io.resolver
import lisai.config.json_schema
import lisai.training.cli
import lisai.preprocess.cli
import lisai.evaluation.cli
import lisai_runner.queue.cli
from lisai.cli import build_parser
from lisai.config.settings import Settings

module = importlib.import_module("lisai.config.settings")
assert not hasattr(module, "settings")
assert not any(isinstance(x, Settings) for x in vars(lisai.config).values())
parser = build_parser()
for arguments in (["--help"], ["train", "--help"], ["preprocess", "--help"],
                  ["apply", "--help"], ["evaluate", "--help"],
                  ["models", "--help"], ["configs", "--help"]):
    try:
        parser.parse_args(arguments)
    except SystemExit as exc:
        assert exc.code == 0, (arguments, exc.code)
    else:
        raise AssertionError(f"Expected --help to exit cleanly: {arguments}")
print("IMPORT_PARSER_SAFE")
'''


def test_import_and_cli_help_are_inert_without_local_config(tmp_path: Path):
    empty_cwd = tmp_path / "outside_checkout"
    empty_cwd.mkdir()
    src = Path(__file__).resolve().parents[3] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    completed = subprocess.run(
        [sys.executable, "-c", _HELP_SCRIPT],
        cwd=empty_cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "IMPORT_PARSER_SAFE" in completed.stdout
    assert list(empty_cwd.iterdir()) == []


def test_explicit_settings_never_initialize_missing_config(tmp_path: Path, monkeypatch):
    from lisai.config.settings import Settings

    monkeypatch.setattr("builtins.input", lambda *_args: pytest.fail("unexpected prompt"))
    empty = tmp_path / "missing" / "configs"
    with pytest.raises(FileNotFoundError, match="local_config.yml"):
        Settings.from_config_dir(empty)
    assert not empty.exists()
