from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def try_open_path(path: Path) -> bool:
    """Open a filesystem path with the platform file explorer when available."""
    resolved = path.resolve()

    startfile = getattr(os, "startfile", None)
    if callable(startfile):
        try:
            startfile(str(resolved))
            return True
        except OSError:
            pass

    commands: list[list[str]] = []
    explorer = shutil.which("explorer.exe") or shutil.which("explorer")
    if explorer is not None:
        target = _to_windows_path(resolved)
        commands.append([explorer, target if target is not None else str(resolved)])

    xdg_open = shutil.which("xdg-open")
    if xdg_open is not None:
        commands.append([xdg_open, str(resolved)])

    for command in commands:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False


def _to_windows_path(path: Path) -> str | None:
    wslpath_cmd = shutil.which("wslpath")
    if wslpath_cmd is None:
        return None
    try:
        completed = subprocess.run(
            [wslpath_cmd, "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    converted = completed.stdout.strip()
    return converted or None


__all__ = ["try_open_path"]
