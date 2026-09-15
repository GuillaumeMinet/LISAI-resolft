from __future__ import annotations

from pathlib import Path

import lisai.infra.cli.open_path as open_path


def test_try_open_path_uses_available_desktop_opener(monkeypatch, tmp_path: Path):
    target = tmp_path / "folder"
    target.mkdir()
    launched = []

    monkeypatch.delattr(open_path.os, "startfile", raising=False)
    monkeypatch.setattr(
        open_path.shutil,
        "which",
        lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None,
    )
    monkeypatch.setattr(
        open_path.subprocess,
        "Popen",
        lambda command, **_kwargs: launched.append(command),
    )

    assert open_path.try_open_path(target) is True
    assert launched == [["/usr/bin/xdg-open", str(target.resolve())]]


def test_try_open_path_returns_false_without_available_opener(monkeypatch, tmp_path: Path):
    target = tmp_path / "folder"
    target.mkdir()

    monkeypatch.delattr(open_path.os, "startfile", raising=False)
    monkeypatch.setattr(open_path.shutil, "which", lambda _name: None)

    assert open_path.try_open_path(target) is False
