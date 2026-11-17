"""Explicit, non-destructive workspace initialization, updates and legacy import.

Never called during Settings construction, imports or normal command execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
import shutil

import yaml

from lisai.config.io.yaml import load_yaml
from lisai.config.models import DeveloperLocalConfig
from lisai.config.settings import (
    CURRENT_DEFAULTS_REVISION, Settings, WorkspaceFileConfig,
    _legacy_project_templates,
)
from lisai.config.user_configs import refresh_user_schemas, seed_inference_local_configs


def _resource(name: str):
    return files("lisai.resources").joinpath("configs", "workspace", name)


def _yaml_text(value: dict) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _write_new(path: Path, content: str) -> bool:
    """Create once, without overwriting even on repeated init."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return True


def _workspace_default(name: str) -> dict:
    return load_yaml(_resource(name))


def _validate_globals(workspace: Path) -> None:
    Settings.from_workspace(workspace)


def _scaffold_root(config_root: Path, *, workspace: Path) -> list[Path]:
    created = []
    for relative in ("training/local", "preprocess/local", "inference/local", "schema"):
        folder = config_root / relative
        if not folder.exists():
            folder.mkdir(parents=True)
            created.append(folder)
    created.extend(seed_inference_local_configs(config_root))
    # Explicit initialization also refreshes generated schemas. This is a cache,
    # not user-authored config. Use the authoritative workspace registry.
    refresh_user_schemas(stg=Settings.from_workspace(workspace, config_root=config_root))
    return created


def initialize_workspace(workspace: str | Path, *, core_only: bool = False) -> list[Path]:
    """Create missing workspace infrastructure, never replace existing files/data.

    core_only is used exclusively by dev init. Workspace init always scaffolds
    the workspace's own editable configs, even in activated developer mode.
    """
    root = Path(workspace).expanduser().resolve()
    created: list[Path] = []
    for filename in ("workspace.yml", "data.yml"):
        target = root / "configs" / filename
        if _write_new(target, _resource(filename).read_text(encoding="utf-8")):
            created.append(target)
    _validate_globals(root)
    for name in ("datasets", "models", "noise_models", "inference", "tensorboard", ".lisai/queue"):
        target = root / name
        if not target.exists():
            target.mkdir(parents=True)
            created.append(target)
    if not core_only:
        created.extend(_scaffold_root(root / "configs", workspace=root))
    return created


def initialize_developer(source_root: str | Path, workspace: str | Path) -> list[Path]:
    """Explicit developer configuration for a verified checkout, no env writes."""
    source = Path(source_root).resolve()
    root = Path(workspace).expanduser().resolve()
    if not (source / "pyproject.toml").is_file() or not (source / "src/lisai/__init__.py").is_file():
        raise ValueError(f"Not a LISAI source checkout: {source}")
    _validate_globals(root)
    config_root = source / "configs"
    target = config_root / "local.yml"
    contents = _yaml_text({"dev": {"enabled": True}, "workspace": {"root": str(root)},
                           "config": {"mode": "repo"}})
    DeveloperLocalConfig.model_validate(yaml.safe_load(contents))
    created = [target] if _write_new(target, contents) else []
    if target.is_file():
        local = DeveloperLocalConfig.model_validate(load_yaml(target))
        if not local.dev.enabled:
            raise ValueError(f"Developer activation is disabled in existing {target}; edit it explicitly")
        if local.workspace.root != str(root):
            raise ValueError(f"Existing {target} selects {local.workspace.root!r}, not {str(root)!r}; edit it explicitly")
    created.extend(_scaffold_root(config_root, workspace=root))
    return created


def _backup(path: Path) -> Path:
    # Never overwrite earlier backups, even for repeated updates in one second.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = path.with_name(path.name + f".bak.{stamp}")
    shutil.copy2(path, candidate)
    return candidate


def update_workspace(workspace: str | Path, *, config_root: str | Path | None = None,
                     replace_editable_defaults: bool = False) -> list[tuple[Path, Path]]:
    """Explicitly replace generated defaults with backups; never touch user experiments.

    User may choose to replace seeded inference files in the specified writable
    config root, but those files are otherwise untouched.
    """
    root = Path(workspace).expanduser().resolve()
    _validate_globals(root)
    replacements: dict[Path, str] = {}
    for filename in ("workspace.yml", "data.yml"):
        path = root / "configs" / filename
        replacements[path] = _resource(filename).read_text(encoding="utf-8")
    if replace_editable_defaults:
        from lisai.config.models.inference import ResolvedInferenceConfig
        from lisai.config.models.inference.presets import POST_TRAINING_OVERRIDES
        from lisai.config.user_configs import INFERENCE_DEFAULTS_HINT, INFERENCE_OVERRIDES_HINT
        editable = Path(config_root).expanduser().resolve() if config_root else root / "configs"
        replacements[editable / "inference/local/defaults.yml"] = (
            INFERENCE_DEFAULTS_HINT + "\n" + _yaml_text(ResolvedInferenceConfig().model_dump(mode="json")))
        replacements[editable / "inference/local/post_training.yml"] = (
            INFERENCE_OVERRIDES_HINT + "\n" + _yaml_text(POST_TRAINING_OVERRIDES))
    # Write backups before replacing any file. Missing generated files are added.
    backups = []
    for path, contents in replacements.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = _backup(path)
            backups.append((path, backup))
        path.write_text(contents, encoding="utf-8")
    return backups


def migrate_legacy_workspace(legacy_config_root: str | Path, workspace: str | Path) -> list[Path]:
    """Explicit conversion of old three-file YAMLs, preserving originals/data.

    The old data_root must match the destination workspace. No implicit data
    copying, renaming, or active data_root aliases are introduced.
    """
    old = Path(legacy_config_root).expanduser().resolve()
    root = Path(workspace).expanduser().resolve()
    settings = Settings.from_config_dir(old)
    if settings.WORKSPACE_ROOT != root:
        raise ValueError(
            f"Legacy data_root {settings.WORKSPACE_ROOT} does not match destination {root}; "
            "choose that workspace explicitly; no data will be moved automatically")
    project = _legacy_project_templates(load_yaml(old / "project_config.yml"))
    project["paths"]["roots"].pop("code_dir", None)
    project["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    project["inference"] = settings.local_cfg.inference.model_dump(mode="json")
    project["console"] = settings.local_cfg.console.model_dump(mode="json")
    WorkspaceFileConfig.model_validate(project)
    documents = {
        root / "configs/workspace.yml": _yaml_text(project),
        root / "configs/data.yml": (old / "data_config.yml").read_text(encoding="utf-8"),
    }
    existing = [p for p in documents if p.exists()]
    if existing:
        raise FileExistsError("Migration will not overwrite existing workspace files: " + ", ".join(map(str, existing)))
    created = []
    for path, content in documents.items():
        if _write_new(path, content):
            created.append(path)
    _validate_globals(root)
    return created


def revision_warning(workspace: str | Path) -> str | None:
    path = Path(workspace).expanduser().resolve() / "configs/workspace.yml"
    if not path.is_file():
        return None
    raw = load_yaml(path)
    revision = WorkspaceFileConfig.model_validate(raw).defaults_revision
    if revision < CURRENT_DEFAULTS_REVISION:
        return (f"Workspace defaults revision {revision} is older than packaged revision "
                f"{CURRENT_DEFAULTS_REVISION}. Run `lisai workspace update` to review and apply updates.")
    return None
