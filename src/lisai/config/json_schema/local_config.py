from __future__ import annotations

import json
from pathlib import Path

from ..models.local_config import LocalConfig


def local_config_json_schema() -> dict:
    """Return the JSON schema for the untracked machine-local LISAI config."""
    return LocalConfig.model_json_schema()


def write_local_config_json_schema(output_path: str | Path) -> Path:
    """Write the local-config JSON schema to disk and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(local_config_json_schema(), indent=2) + "\n", encoding="utf-8")
    return path
