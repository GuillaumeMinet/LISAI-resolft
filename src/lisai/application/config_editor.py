"""Editing user training YAMLs using the existing LISAI configuration resolver."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def list_user_training_configs(settings) -> list[tuple[str, Path]]:
    """Return only writable workspace training configurations; no built-in presets."""
    from lisai.config.io.config_paths import ConfigPathResolver

    return [(name, Path(file)) for name, file in
            ConfigPathResolver("training", stg=settings).user_config_paths()]


def validate_training_yaml_text(text: str, settings):
    """Parse/validate authored YAML with LISAI's actual training resolver."""
    import yaml
    from lisai.config.catalog import validate_training_config_dict
    from lisai.config.io.metadata import strip_config_metadata
    from lisai.config.io.resolver import resolve_config_dict

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid training YAML syntax: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Training YAML must be a mapping (object).")
    validate_training_config_dict(document)
    return resolve_config_dict(strip_config_metadata(document), stg=settings)


def managed_training_config_path(settings, value: str | Path) -> Path:
    """Restrict all GUI saves and loads to the workspace's writable config root."""
    root = Path(settings.TRAINING_CONFIG_DIR).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    target = path.resolve()
    if not target.is_relative_to(root) or target.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Choose a .yml/.yaml file inside the workspace training config directory.")
    if not target.name or target.name.startswith("."):
        raise ValueError("Invalid training config filename.")
    return target


def save_training_yaml_text(
    text: str, config_path: str | Path, *, settings, overwrite: bool = False,
) -> Path:
    """Validate first; exclusive-create on Save As and atomic-replace on Save."""
    validate_training_yaml_text(text, settings)
    target = managed_training_config_path(settings, config_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Training configuration already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{target.name}.",
            suffix=".tmp", dir=target.parent, delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text if text.endswith("\n") else text + "\n")
        # Recheck to avoid replacing a newly-created file when Save As is used.
        if not overwrite:
            with target.open("x", encoding="utf-8") as output:
                output.write(Path(temp_name).read_text(encoding="utf-8"))
        else:
            os.replace(temp_name, target)
            temp_name = None
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return target
