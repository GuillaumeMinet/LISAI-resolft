"""Workspace actions reusable by GUI hosts, without Qt or host settings."""
from __future__ import annotations

from pathlib import Path


def validate_workspace(workspace: str | Path):
    """Validate a *full* LISAI workspace, without creating or changing files."""
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"LISAI workspace directory does not exist: {root}")
    for name in ("workspace.yml", "data.yml"):
        required = root / "configs" / name
        if not required.is_file():
            raise ValueError(
                f"Incomplete LISAI workspace: missing {required}. "
                f"Run 'lisai workspace init {root}' in a terminal."
            )
    for name in ("defaults.yml", "post_training.yml"):
        required = root / "configs" / "inference" / name
        if not required.is_file():
            raise ValueError(
                f"LISAI workspace is missing inference/{name} (possibly core-only). "
                f"Complete it with 'lisai workspace init {root}'."
            )
    try:
        from lisai.config.settings import Settings

        return Settings.from_workspace(root)
    except Exception as error:
        raise ValueError(f"Cannot load LISAI workspace {root}: {error}") from error


def create_workspace(workspace: str | Path):
    """Explicit full workspace creation/repair, independent of LISAI CLI selection.

    Only initialize an empty directory or an already recognizable workspace;
    refuse to populate an unrelated nonempty directory by accident.
    """
    requested = Path(workspace).expanduser()
    if not requested.is_absolute():
        raise ValueError("Enter an absolute workspace directory path.")
    root = requested.resolve()
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"Workspace target is not a directory: {root}")
        if any(root.iterdir()) and not (root / "configs" / "workspace.yml").is_file():
            raise ValueError(
                f"{root} is not empty and does not contain a LISAI workspace. "
                "Choose an empty directory to avoid modifying unrelated files."
            )
    from lisai.config.workspace import initialize_workspace

    initialize_workspace(root)
    return validate_workspace(root)
