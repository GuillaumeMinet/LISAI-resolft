"""CLI-only selection of workspace and writable experiment-config locations.

Selection reads only local selector fields and never creates files. The explicit
loader below constructs Settings from the selected workspace/config root.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

import yaml


@dataclass(frozen=True)
class ResolvedLocations:
    workspace_root: Path
    experiment_config_root: Path
    workspace_origin: str
    config_root_origin: str
    local_config_path: Path | None = None
    source_root: Path | None = None
    developer_mode: bool = False

    @property
    def workspace_config(self) -> Path:
        return self.workspace_root / "configs" / "workspace.yml"

    @property
    def data_config(self) -> Path:
        return self.workspace_root / "configs" / "data.yml"


def _absolute(value: str | os.PathLike[str], *, label: str, from_file: bool = False) -> Path:
    path = Path(value).expanduser()
    if (from_file or label.startswith("LISAI_")) and not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {value!s}")
    return path.resolve()


def _source_checkout_for_import() -> Path | None:
    """Verify an editable checkout belongs to the imported LISAI package.

    A wheel or an unrelated parent Git repository is never treated as editable.
    When distribution metadata is unavailable (e.g. PYTHONPATH=repo/src), a
    validated src layout + Git root is the documented fallback.
    """
    import lisai

    package_path = Path(lisai.__file__).resolve()
    if package_path.name != "__init__.py" or package_path.parent.name != "lisai":
        return None
    src = package_path.parent.parent
    if src.name != "src":
        return None
    repo = src.parent
    pyproject = repo / "pyproject.toml"
    if not pyproject.is_file() or not (repo / ".git").exists():
        return None
    project_text = pyproject.read_text(encoding="utf-8")
    if not re.search(r'^name\s*=\s*["\']lisai["\']\s*$', project_text, re.MULTILINE):
        return None
    if (repo / "src" / "lisai" / "__init__.py").resolve() != package_path:
        return None
    try:
        git_root = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    if Path(git_root).resolve() != repo.resolve():
        return None

    try:
        distribution = metadata.distribution("lisai")
    except metadata.PackageNotFoundError:
        # Source-only import, including a test checkout before pip install -e.
        return repo
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None  # A regular distribution cannot activate a parent checkout.
    try:
        payload = json.loads(direct_url)
        if payload.get("dir_info", {}).get("editable") is not True:
            return None
        parsed = urlparse(payload["url"])
        if parsed.scheme != "file":
            return None
        # On Windows a local file URI looks like file:///C:/path.
        local_path = unquote(parsed.path)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", local_path):
            local_path = local_path[1:]
        editable_root = Path(local_path).resolve()
    except (ValueError, KeyError, TypeError):
        return None
    return repo if editable_root == repo.resolve() else None


def _read_local_selector(path: Path) -> tuple[bool, Path | None]:
    """Read only selector fields; typed local preferences are implemented in B."""
    if not path.is_file():
        return False, None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    dev = raw.get("dev", {})
    workspace = raw.get("workspace", {})
    if not isinstance(dev, dict) or not isinstance(workspace, dict):
        raise ValueError(f"dev and workspace must be mappings in {path}")
    enabled = dev.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"dev.enabled must be boolean in {path}")
    root = workspace.get("root")
    if root is not None and (not isinstance(root, str) or not root.strip()):
        raise ValueError(f"workspace.root must be a nonempty absolute path in {path}")
    resolved = _absolute(root, label=f"{path}: workspace.root", from_file=True) if root else None
    return enabled, resolved


def resolve_cli_locations(
    *,
    workspace: str | os.PathLike[str] | None = None,
    config_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    editable_source_root: Path | None | bool = False,
) -> ResolvedLocations:
    """Resolve two *independent* roots without loading workspace YAML or writing.

    ``editable_source_root=False`` (the default) requests verified discovery;
    ``None`` disables discovery, and a Path supplies a verified source root for
    deterministic tests/callers. This is a boundary helper, not a global context.
    """
    env = os.environ if environ is None else environ
    source = _source_checkout_for_import() if editable_source_root is False else editable_source_root
    source = Path(source).resolve() if source is not None else None

    external_root: Path | None = None
    root_origin: str | None = None
    if config_root is not None:
        external_root = _absolute(config_root, label="--config-root")
        root_origin = "--config-root"
    elif env.get("LISAI_CONFIG_ROOT"):
        external_root = _absolute(env["LISAI_CONFIG_ROOT"], label="LISAI_CONFIG_ROOT")
        root_origin = "LISAI_CONFIG_ROOT"
    elif source is not None:
        candidate = source / "configs"
        active, _ = _read_local_selector(candidate / "local.yml")
        if active:
            external_root, root_origin = candidate.resolve(), "developer activation"

    local_path = external_root / "local.yml" if external_root is not None else None
    local_root: Path | None = None
    local_dev = False
    if local_path is not None:
        local_dev, local_root = _read_local_selector(local_path)

    if workspace is not None:
        selected_workspace = _absolute(workspace, label="--workspace")
        workspace_origin = "--workspace"
    elif local_root is not None:
        selected_workspace = local_root
        workspace_origin = f"{local_path}: workspace.root"
    elif env.get("LISAI_HOME"):
        selected_workspace = _absolute(env["LISAI_HOME"], label="LISAI_HOME")
        workspace_origin = "LISAI_HOME"
    else:
        selected_workspace = (Path.home() / "LISAI").resolve()
        workspace_origin = "~/LISAI"

    if external_root is None:
        external_root = selected_workspace / "configs"
        root_origin = "workspace default"
    assert root_origin is not None
    return ResolvedLocations(
        workspace_root=selected_workspace,
        experiment_config_root=external_root,
        workspace_origin=workspace_origin,
        config_root_origin=root_origin,
        local_config_path=local_path,
        source_root=source,
        developer_mode=bool(
            source is not None and external_root == source / "configs" and local_dev
        ),
    )


def load_settings_for_selected_locations(locations: ResolvedLocations):
    """Load both selected roots once at the CLI boundary, with no fallback."""
    from lisai.config.settings import Settings

    return Settings.from_workspace(
        locations.workspace_root,
        config_root=locations.experiment_config_root,
        local_config=locations.local_config_path if (
            locations.local_config_path is not None and locations.local_config_path.is_file()
        ) else None,
    )



def describe_locations(locations: ResolvedLocations) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("lisai")
    except PackageNotFoundError:
        package_version = "source checkout (no installed metadata)"
    workspace_present = locations.workspace_config.is_file() and locations.data_config.is_file()
    from lisai.config.settings import CURRENT_DEFAULTS_REVISION
    revision_status = "unavailable"
    if locations.workspace_config.is_file():
        try:
            from lisai.config.io.yaml import load_yaml
            revision = load_yaml(locations.workspace_config).get("defaults_revision", 0)
            revision_status = f"{revision} (packaged: {CURRENT_DEFAULTS_REVISION})"
            if isinstance(revision, int) and revision < CURRENT_DEFAULTS_REVISION:
                revision_status += " — update available (lisai workspace update)"
        except (OSError, ValueError):
            revision_status = "invalid workspace.yml"
    return "\n".join((
        f"Mode:               {'Developer' if locations.developer_mode else 'Normal'}",
        f"LISAI version:      {package_version}",
        f"LISAI package:      {Path(__file__).resolve().parents[1]}",
        f"Workspace:          {locations.workspace_root}  [{locations.workspace_origin}]",
        f"Global settings:    {locations.workspace_config}"
        f"  [{'present' if locations.workspace_config.is_file() else 'missing'}]",
        f"Data settings:      {locations.data_config}"
        f"  [{'present' if locations.data_config.is_file() else 'missing'}]",
        f"Experiment root:    {locations.experiment_config_root}  [{locations.config_root_origin}]",
        f"Local preferences:  {locations.local_config_path or '(none)'}",
        "Built-in configs:   LISAI installed package resources",
        f"Defaults revision:  {revision_status}",
        f"Workspace status:   {'initialized' if workspace_present else 'not initialized'}",
        "Workspace init:     lisai workspace init <PATH>" if not workspace_present else "",
    )).rstrip()
