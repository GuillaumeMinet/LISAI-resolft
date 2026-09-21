from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """ Load yaml config file. """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a dictionary")

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