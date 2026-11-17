"""Compute editor-schema hints for authored YAMLs without guessing their depth."""
from __future__ import annotations

import os
from pathlib import Path


def schema_header(target: Path, schema_root: Path, schema_filename: str) -> str:
    """Return a truthful relative schema comment, or nothing across drives."""
    try:
        relative = Path(os.path.relpath(schema_root / schema_filename, target.parent))
    except ValueError:  # e.g. unrelated drives on Windows
        return ""
    return f"# yaml-language-server: $schema={relative.as_posix()}"
