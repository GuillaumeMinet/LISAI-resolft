"""Editing user training YAMLs using the existing LISAI configuration resolver."""
from __future__ import annotations

import hashlib
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
    from lisai.config.io.resources import BUILTIN_NAMESPACES
    if target.relative_to(root).parts[0] in BUILTIN_NAMESPACES.get("training", ()):
        raise ValueError("Packaged training config namespaces are reserved for built-in resources.")
    return target


def validate_training_yaml_authoring(text: str) -> list[str]:
    """Validate editable structure and report unresolved scaffold placeholders."""
    import yaml
    from lisai.config.catalog import PLACEHOLDER_RE, _changeme_paths, validate_training_template_dict

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid training YAML syntax: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("Training YAML must be a mapping (object).")
    validate_training_template_dict(document)
    placeholders = _changeme_paths(document)

    def collect_other_placeholders(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from collect_other_placeholders(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from collect_other_placeholders(child, f"{prefix}[{index}]")
        elif isinstance(value, str) and PLACEHOLDER_RE.fullmatch(value):
            yield prefix or "<root>"

    placeholders.extend(collect_other_placeholders(document))
    return (["Unfilled template fields: " + ", ".join(placeholders)] if placeholders else [])


def _file_revision(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_training_yaml_text(
    text: str, config_path: str | Path, *, settings, overwrite: bool = False,
    validation: str = "trainable", expected_revision: str | None = None,
) -> Path:
    """Validate first; exclusive-create on Save As and atomic-replace on Save."""
    if validation == "trainable":
        validate_training_yaml_text(text, settings)
    elif validation == "authoring":
        validate_training_yaml_authoring(text)
    else:
        raise ValueError(f"Unknown YAML validation level: {validation}")
    target = managed_training_config_path(settings, config_path)
    if expected_revision is not None:
        if not target.is_file() or _file_revision(target) != expected_revision:
            raise RuntimeError(
                f"Training YAML changed on disk: {target}. Reload or use Save As."
            )
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
            created = False
            try:
                with target.open("x", encoding="utf-8") as output:
                    created = True
                    output.write(Path(temp_name).read_text(encoding="utf-8"))
            except BaseException:
                if created:
                    target.unlink(missing_ok=True)
                raise
        else:
            os.replace(temp_name, target)
            temp_name = None
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return target
