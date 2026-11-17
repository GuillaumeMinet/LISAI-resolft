from pathlib import Path
from lisai._compat import Traversable

import yaml


def load_yaml(path: str | Path | Traversable) -> dict:
    """Load a YAML mapping from a regular file or a read-only package resource."""
    source = Path(path) if isinstance(path, (str, Path)) else path
    if not source.is_file():
        raise FileNotFoundError(f"YAML config not found: {source}")

    with source.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config {source} must be a dictionary")

    return cfg

def save_yaml(
    cfg: dict,
    path: str | Path,
    *,
    default_flow_style: bool | None = False,
):
    """Save a config dictionary to YAML."""
    path = Path(path)
    with open(path, "w") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            default_flow_style=default_flow_style,
        )