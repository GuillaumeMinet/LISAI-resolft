from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from lisai.infra.paths import Paths

from .schema import CodeState


def collect_code_state(repo_dir: str | Path | None = None, *, paths: Paths | None = None) -> CodeState:
    """Record installed LISAI version and verified source checkout Git state.

    ``repo_dir`` explicitly selects a repository for compatibility/testing;
    ordinary training passes ``paths`` but never treats it as a Git location.
    """
    # Production callers never infer a source checkout from a workspace or cwd.
    # The shared verifier checks the *imported* LISAI package, installation
    # metadata, project layout, and Git root. An installed wheel has no source
    # checkout and therefore records version-only provenance.
    from lisai.config.selection import _source_checkout_for_import

    cwd = Path(repo_dir).resolve() if repo_dir is not None else _source_checkout_for_import()
    git_commit = _git_output(cwd, "rev-parse", "HEAD") if cwd is not None else None
    git_branch = _git_output(cwd, "branch", "--show-current") if cwd is not None else None
    git_remote = _git_output(cwd, "config", "--get", "remote.origin.url") if cwd is not None else None
    status = _git_output(cwd, "status", "--porcelain", empty_as_none=False) if cwd is not None else None

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
