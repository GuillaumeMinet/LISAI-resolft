from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from lisai.infra.paths import Paths

from .schema import CodeState


def collect_code_state(repo_dir: str | Path | None = None, *, paths: Paths | None = None) -> CodeState:
    """Collect a best-effort snapshot of the source state for run metadata."""
    if repo_dir is None and paths is None:
        raise ValueError("collect_code_state requires paths when repo_dir is not supplied")
    cwd = Path(repo_dir).resolve() if repo_dir is not None else paths.project_root()
    git_commit = _git_output(cwd, "rev-parse", "HEAD")
    git_branch = _git_output(cwd, "branch", "--show-current")
    git_remote = _git_output(cwd, "config", "--get", "remote.origin.url")
    status = _git_output(cwd, "status", "--porcelain", empty_as_none=False)

    return CodeState(
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=None if status is None else bool(status.strip()),
        git_remote=git_remote,
        lisai_version=_lisai_version(),
    )

def _git_output(cwd: Path, *args: str, empty_as_none: bool = True) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    text = completed.stdout.strip()
    if empty_as_none and not text:
        return None
    return text


def _lisai_version() -> str | None:
    try:
        return version("lisai")
    except PackageNotFoundError:
        return None


__all__ = ["collect_code_state"]
