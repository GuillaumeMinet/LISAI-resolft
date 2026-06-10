from __future__ import annotations

from copy import deepcopy
from typing import Any

CONFIG_METADATA_KEY = "metadata"


def split_config_metadata(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return a copy of ``cfg`` without catalog metadata plus the metadata block."""
    body = deepcopy(cfg)
    metadata = body.pop(CONFIG_METADATA_KEY, None)
    if metadata is None:
        return body, None
    if not isinstance(metadata, dict):
        raise ValueError(f"`{CONFIG_METADATA_KEY}` must be a mapping when provided.")
    return body, dict(metadata)


def strip_config_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    body, _ = split_config_metadata(cfg)
    return body

