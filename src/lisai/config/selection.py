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
from typing import TYPE_CHECKING, Mapping
from urllib.parse import unquote, urlparse

import yaml

if TYPE_CHECKING:
    from lisai.config.models.local_config import DeveloperLocalConfig


@dataclass(frozen=True)
class ResolvedLocations:
    """CLI-boundary selection, including the intentionally unselected state."""

    workspace_root: Path | None
    experiment_config_root: Path | None
    workspace_origin: str
    config_root_origin: str
    local_config_path: Path | None = None
    source_root: Path | None = None
    developer_mode: bool = False
    config_mode: str | None = None

    @property
    def workspace_config(self) -> Path | None:
        return self.workspace_root / "configs" / "workspace.yml" if self.workspace_root else None

    @property
    def data_config(self) -> Path | None:
        return self.workspace_root / "configs" / "data.yml" if self.workspace_root else None


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



@dataclass(frozen=True)
class DeveloperSetupStatus:
    """Whether a checkout's local.yml activates a *complete* developer selection.

    This is a selection-readiness check, not workspace initialization: normal
    commands separately load/validate workspace.yml and data.yml.
    """

    enabled: bool
    complete: bool
    local: DeveloperLocalConfig | None = None
    reason: str | None = None


def developer_setup_status(
    source_root: str | Path, *, document: Mapping[str, object] | None = None,
) -> DeveloperSetupStatus:
    """Inspect developer selection without reading the environment or writing files.

    An activated-but-incomplete checkout must never fall back to LISAI_HOME.
    ``document`` lets `settings set` assess a proposed selector before writing.
    """
    from lisai.config.models.local_config import DeveloperLocalConfig

    path = Path(source_root).resolve() / "configs" / "local.yml"
    if document is None:
        if not path.is_file():
            return DeveloperSetupStatus(False, False, reason="local.yml is absent")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raw = document
    if raw is None:
        return DeveloperSetupStatus(False, False, reason="local.yml is empty")
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    dev = raw.get("dev", {})
    if not isinstance(dev, dict):
        raise ValueError(f"dev must be a mapping in {path}")
    enabled = dev.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"dev.enabled must be boolean in {path}")
    if not enabled:
        return DeveloperSetupStatus(False, False, reason="dev.enabled is not true")
    # Strict validation still rejects wrong paths, unsupported keys and invalid
    # configuration modes. Missing selection fields alone are allowed at init.
    local = DeveloperLocalConfig.model_validate(raw)
    missing = []
    if local.workspace.root is None:
        missing.append("workspace.root")
    if local.config is None:
        missing.append("config.mode")
    return DeveloperSetupStatus(
        True, not missing, local,
        f"Missing {', '.join(missing)}" if missing else None,
    )


def resolve_cli_locations(
    *,
    environ: Mapping[str, str] | None = None,
    editable_source_root: Path | None | bool = False,
    require_workspace: bool = True,
) -> ResolvedLocations:
    """Resolve persistent CLI selection once, without settings reads or writes.

    ``editable_source_root=False`` requests verified discovery; ``None``
    disables it; a Path supplies a verified root for deterministic tests.
    No explicit command-line root overrides or config-root environment variable
    participate in this selection.
    """
    env = os.environ if environ is None else environ
    source = _source_checkout_for_import() if editable_source_root is False else editable_source_root
    source = Path(source).resolve() if source is not None else None

    selected_workspace: Path | None = None
    experiment_root: Path | None = None
    local_path: Path | None = None
    developer_mode = False
    config_mode: str | None = None
    workspace_origin = "LISAI_HOME (unset)"
    config_origin = "workspace default (unavailable)"

    if source is not None:
        candidate = source / "configs" / "local.yml"
        setup = developer_setup_status(source)
        if setup.enabled and not setup.complete:
            raise ValueError(
                f"Developer setup is incomplete ({setup.reason}) in {candidate}. "
                "Run 'lisai settings set --workspace PATH' to complete selection "
                "(and --config-mode repo|workspace|custom if needed)."
            )
        local = setup.local if setup.complete else None
        if local is not None:
            assert local.config is not None and local.workspace.root is not None
            developer_mode = True
            local_path = candidate
            selected_workspace = Path(local.workspace.root)
            config_mode = local.config.mode
            workspace_origin = f"{candidate}: workspace.root"
            if config_mode == "repo":
                experiment_root = source / "configs"
                config_origin = "developer config.mode=repo"
            elif config_mode == "workspace":
                experiment_root = selected_workspace / "configs"
                config_origin = "developer config.mode=workspace"
            else:
                assert local.config.root is not None
                experiment_root = Path(local.config.root)
                config_origin = "developer config.mode=custom"

    if not developer_mode and env.get("LISAI_HOME"):
        selected_workspace = _absolute(env["LISAI_HOME"], label="LISAI_HOME")
        experiment_root = selected_workspace / "configs"
        workspace_origin = "LISAI_HOME"
        config_origin = "workspace default"

    if selected_workspace is None and require_workspace:
        raise ValueError(
            "No LISAI workspace selected. Set LISAI_HOME to an initialized workspace "
            "(standalone CLI), or use 'lisai dev init' then "
            "'lisai settings set --workspace PATH' in a verified editable checkout."
        )

    return ResolvedLocations(
        workspace_root=selected_workspace,
        experiment_config_root=experiment_root,
        workspace_origin=workspace_origin,
        config_root_origin=config_origin,
        local_config_path=local_path,
        source_root=source,
        developer_mode=developer_mode,
        config_mode=config_mode,
    )


def load_settings_for_selected_locations(locations: ResolvedLocations):
    """Load already selected W/R and optional D preferences exactly once."""
    from lisai.config.settings import Settings

    if locations.workspace_root is None or locations.experiment_config_root is None:
        raise ValueError("No LISAI workspace selected; set LISAI_HOME or activate developer mode")
    return Settings.from_workspace(
        locations.workspace_root,
        config_root=locations.experiment_config_root,
        local_config=locations.local_config_path if locations.developer_mode else None,
    )


def describe_locations(locations: ResolvedLocations) -> str:
    """Read-only diagnostics, including an unselected/uninitialized workspace."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("lisai")
    except PackageNotFoundError:
        package_version = "source checkout (no installed metadata)"

    workspace = locations.workspace_root
    workspace_config = locations.workspace_config
    data_config = locations.data_config
    present = bool(workspace_config and data_config and workspace_config.is_file() and data_config.is_file())
    revision_status = "unavailable"
    if workspace_config is not None and workspace_config.is_file():
        try:
            from lisai.config.io.yaml import load_yaml
            from lisai.config.settings import CURRENT_DEFAULTS_REVISION

            revision = load_yaml(workspace_config).get("defaults_revision", 0)
            revision_status = f"{revision} (packaged: {CURRENT_DEFAULTS_REVISION})"
            if isinstance(revision, int) and revision < CURRENT_DEFAULTS_REVISION:
                revision_status += " — update available (lisai workspace update)"
        except (OSError, ValueError):
            revision_status = "invalid workspace.yml"

    validation = "no workspace selected" if workspace is None else "not initialized"
    if present:
        try:
            load_settings_for_selected_locations(locations)
            validation = "valid"
        except (OSError, ValueError) as exc:
            validation = f"invalid: {exc}"

    rows = [
        f"Mode:               {'Developer' if locations.developer_mode else 'Standalone'}",
        f"LISAI version:      {package_version}",
        f"LISAI package:      {Path(__file__).resolve().parents[1]}",
        f"Workspace:          {workspace or '(not selected)'}  [{locations.workspace_origin}]",
        f"Global settings:    {workspace_config or '(not selected)'}"
        f"  [{'present' if workspace_config and workspace_config.is_file() else 'missing'}]",
        f"Data settings:      {data_config or '(not selected)'}"
        f"  [{'present' if data_config and data_config.is_file() else 'missing'}]",
        f"Experiment root:    {locations.experiment_config_root or '(not selected)'}"
        f"  [{locations.config_root_origin}]",
        f"Config mode:        {locations.config_mode or '(standalone)'}",
        f"Local preferences:  {locations.local_config_path or '(none)'}",
        "Built-in configs:   LISAI installed package resources",
        f"Defaults revision:  {revision_status}",
        f"Workspace status:   {validation}",
    ]
    if not present:
        rows.append("Workspace init:     lisai workspace init <PATH>")
    return "\n".join(rows)
