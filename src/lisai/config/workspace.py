"""Explicit, non-destructive workspace initialization, updates and legacy import.

Never called during Settings construction, imports or normal command execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
import shutil

import yaml

from lisai.config.io.yaml import load_yaml
from lisai.config.io.schema_hints import schema_header
from lisai.config.models import DataConfig, DeveloperLocalConfig
from lisai.config.settings import (
    CURRENT_DEFAULTS_REVISION, Settings, WorkspaceFileConfig,
    _legacy_project_templates,
)
from lisai.config.user_configs import (
    inference_seed_files, refresh_user_schemas, seed_inference_local_configs,
)


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
    for relative in ("training", "preprocess", "inference", "schema"):
        folder = config_root / relative
        if not folder.exists():
            folder.mkdir(parents=True)
            created.append(folder)
    created.extend(seed_inference_local_configs(config_root))
    # Explicit initialization also refreshes generated schemas. This is a cache,
    # not user-authored config. Use the authoritative workspace registry.
    refresh_user_schemas(stg=Settings.from_workspace(workspace, config_root=config_root))
    return created


def open_workspace(workspace: str | Path) -> Path:
    """Open a selected workspace in the native file explorer, without modifying it."""
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Selected workspace directory does not exist: {root}")
    if sys.platform == "win32":
        os.startfile(str(root))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(root)])
    else:
        subprocess.Popen(["xdg-open", str(root)])
    return root


def initialize_workspace(workspace: str | Path, *, core_only: bool = False) -> list[Path]:
    """Create missing workspace infrastructure, never replace existing files/data.

    Core-only init is also public. Full init always scaffolds the workspace's
    own editable configs, even in activated developer mode.
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
    header = schema_header(target, config_root / "schema", "local-config.schema.json")
    if header:
        contents = header + "\n" + contents
    created = [target] if _write_new(target, contents) else []
    if target.is_file():
        local = DeveloperLocalConfig.model_validate(load_yaml(target))
        if not local.dev.enabled:
            raise ValueError(f"Developer activation is disabled in existing {target}; edit it explicitly")
        if local.workspace.root != str(root):
            raise ValueError(f"Existing {target} selects {local.workspace.root!r}, not {str(root)!r}; edit it explicitly")
    created.extend(_scaffold_root(config_root, workspace=root))
    return created


@dataclass(frozen=True)
class WorkspaceUpdateChange:
    """A proposed replacement, preserving the bytes reviewed by the caller."""

    path: Path
    previous: bytes | None
    replacement: bytes

    @property
    def action(self) -> str:
        if self.previous is None:
            return "CREATE"
        return "UNCHANGED" if self.previous == self.replacement else "REPLACE"

    def diff(self) -> str:
        old = self.previous.decode("utf-8").splitlines(keepends=True) if self.previous is not None else []
        new = self.replacement.decode("utf-8").splitlines(keepends=True)
        return "".join(difflib.unified_diff(
            old, new, fromfile=str(self.path) + " (current)",
            tofile=str(self.path) + " (proposed)",
        ))


@dataclass(frozen=True)
class WorkspaceUpdatePlan:
    workspace: Path
    inference_root: Path | None
    changes: tuple[WorkspaceUpdateChange, ...]

    @property
    def requires_writes(self) -> bool:
        return any(change.action != "UNCHANGED" for change in self.changes)


def plan_workspace_update(
    workspace: str | Path,
    *,
    include_inference: bool = False,
    external_config_root: str | Path | None = None,
) -> WorkspaceUpdatePlan:
    """Validate and preview all changes without creating anything.

    Workspace-global files always belong to *workspace*. Inference files are
    workspace-owned unless an external root was deliberately supplied together
    with ``include_inference``. Active developer settings are never consulted.
    """
    if external_config_root is not None and not include_inference:
        raise ValueError("--external-config-root requires --include-inference")
    root = Path(workspace).expanduser().resolve()
    _validate_globals(root)
    replacements: dict[Path, str] = {}
    for name in ("workspace.yml", "data.yml"):
        source = _workspace_default(name)
        if name == "workspace.yml":
            WorkspaceFileConfig.model_validate(source)
        else:
            DataConfig.model_validate(source)
        replacements[root / "configs" / name] = _resource(name).read_text(encoding="utf-8")
    editable = None
    if include_inference:
        editable = (Path(external_config_root).expanduser().resolve()
                    if external_config_root is not None else root / "configs")
        replacements.update(inference_seed_files(editable))
    changes = []
    for path, text in replacements.items():
        if path.exists() and not path.is_file():
            raise ValueError(f"Update target is not a file: {path}")
        previous = path.read_bytes() if path.is_file() else None
        changes.append(WorkspaceUpdateChange(path, previous, text.encode("utf-8")))
    return WorkspaceUpdatePlan(root, editable, tuple(changes))


def describe_workspace_update(plan: WorkspaceUpdatePlan) -> str:
    """Exact targets and readable diffs, including unchanged files."""
    lines = [f"Workspace update target: {plan.workspace}"]
    if plan.inference_root is not None:
        owner = "external (explicit)" if plan.inference_root != plan.workspace / "configs" else "workspace"
        lines.append(f"Inference config root: {plan.inference_root} [{owner}]")
    for change in plan.changes:
        lines.append(f"{change.action}: {change.path}")
        if change.action != "UNCHANGED":
            lines.append(change.diff().rstrip("\n"))
    if not plan.requires_writes:
        lines.append("No changes required.")
    return "\n".join(lines)


def _backup(path: Path) -> Path:
    """Create a uniquely named backup without overwriting earlier backups."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for suffix in range(100):
        candidate = path.with_name(f"{path.name}.bak.{stamp}.{suffix}")
        try:
            with candidate.open("xb") as out, path.open("rb") as inp:
                shutil.copyfileobj(inp, out)
            shutil.copystat(path, candidate)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"Unable to allocate a unique backup for {path}")


def _atomic_write(path: Path, contents: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def update_workspace(plan: WorkspaceUpdatePlan) -> list[tuple[Path, Path]]:
    """Apply a previously previewed plan. Caller is responsible for approval.

    Detect concurrent changes *before* any writes. Back up all changing existing
    files before starting replacements, then replace each file atomically.
    """
    for change in plan.changes:
        current = change.path.read_bytes() if change.path.is_file() else None
        if current != change.previous:
            raise ValueError(f"{change.path} changed since update preview; retry the update")
    updates = [change for change in plan.changes if change.action != "UNCHANGED"]
    backups: list[tuple[Path, Path]] = []
    for change in updates:
        if change.previous is not None:
            backups.append((change.path, _backup(change.path)))
    for change in updates:
        change.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(change.path, change.replacement)
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
