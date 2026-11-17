"""Explicit, persistent developer CLI selection edits.

Normal settings loading is strictly read-only. This command alone is allowed to
rewrite the verified source checkout's ``configs/local.yml``, after validating
and displaying the complete proposed selection.
"""
from __future__ import annotations

import copy
import difflib
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from lisai.config.models.local_config import DeveloperLocalConfig
from lisai.config.io.schema_hints import schema_header
from lisai.config.selection import resolve_cli_locations
from lisai.config.settings import Settings


def _developer_file() -> Path:
    # Import the module, not the function, so source verification is easy to
    # override in tests and remains the single CLI verification mechanism.
    from lisai.config import selection

    source = selection._source_checkout_for_import()
    if source is None:
        raise ValueError("'lisai settings set' requires a verified editable LISAI checkout")
    local_path = Path(source).resolve() / "configs" / "local.yml"
    if not local_path.is_file():
        raise ValueError(
            f"No developer configuration at {local_path}. "
            "Run 'lisai dev init' to activate this checkout first."
        )
    return local_path


def _read_active_file(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    dev = raw.get("dev")
    if not isinstance(dev, dict) or dev.get("enabled") is not True:
        raise ValueError(
            f"Developer activation is not enabled in {path}; "
            "use 'lisai dev init' or enable dev.enabled explicitly first."
        )
    return raw


def _input_path(value: str, *, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} cannot be empty")
    # CLI paths may be relative to the invoking cwd, but their persisted form
    # is always absolute; subsequent CLI invocations never depend on cwd.
    return str(Path(value).expanduser().resolve())


def _new_selection(raw: dict[str, Any], args: Any) -> DeveloperLocalConfig:
    updated = copy.deepcopy(raw)
    if args.workspace is not None:
        new_workspace = updated.get("workspace")
        if not isinstance(new_workspace, dict):
            new_workspace = {}
        new_workspace["root"] = _input_path(args.workspace, label="--workspace")
        updated["workspace"] = new_workspace

    existing = updated.get("config")
    if not isinstance(existing, dict):
        existing = {}
    mode = args.config_mode or existing.get("mode")
    if mode is None:
        raise ValueError(
            "Missing config.mode. Specify --config-mode repo, workspace or custom "
            "to complete the developer selection."
        )
    if args.config_root is not None and mode != "custom":
        raise ValueError("--config-root requires --config-mode custom (or an existing custom mode)")
    if args.config_mode is not None:
        existing["mode"] = args.config_mode
    if mode == "custom":
        if args.config_root is not None:
            existing["root"] = _input_path(args.config_root, label="--config-root")
    else:
        # Switching out of custom mode intentionally forgets its old root.
        existing.pop("root", None)
    updated["config"] = existing
    return DeveloperLocalConfig.model_validate(updated)


def _root_for_mode(source: Path, selection: DeveloperLocalConfig) -> tuple[Path, Path]:
    assert selection.workspace.root is not None and selection.config is not None
    workspace = Path(selection.workspace.root)
    if selection.config.mode == "repo":
        config_root = source / "configs"
    elif selection.config.mode == "workspace":
        config_root = workspace / "configs"
    else:
        assert selection.config.root is not None
        config_root = Path(selection.config.root)
    return workspace, config_root


def _atomic_replace(path: Path, text: str) -> None:
    """Replace only once validation and user confirmation have completed."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".local-",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def set_developer_settings(args: Any) -> int:
    """Validate, preview, confirm and persist a developer W/R selection."""
    if args.workspace is None and args.config_mode is None and args.config_root is None:
        raise ValueError("Specify at least one of --workspace, --config-mode or --config-root")

    path = _developer_file()
    source = path.parent.parent
    raw = _read_active_file(path)
    old_bytes = path.read_bytes()
    # We deliberately do not validate the old selector: this command must be
    # usable to repair an activated local.yml missing config.mode/root.
    try:
        current = resolve_cli_locations(environ={}, editable_source_root=source)
        old_state = (
            f"Mode: {current.config_mode}\n"
            f"Workspace: {current.workspace_root}\n"
            f"Experiment root: {current.experiment_config_root}"
        )
    except (ValueError, OSError) as exc:
        old_state = f"Invalid or incomplete: {exc}"

    proposed = _new_selection(raw, args)
    workspace, config_root = _root_for_mode(source, proposed)
    if config_root.exists() and not config_root.is_dir():
        raise ValueError(f"Experiment config root is not a directory: {config_root}")
    # A core-only W is valid; only its two authoritative global files are
    # necessary. Missing experiment directories may be scaffolded later.
    try:
        Settings.from_workspace(workspace, config_root=config_root)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise ValueError(f"Proposed workspace is not initialized/valid: {exc}") from exc

    proposed_raw = copy.deepcopy(raw)
    original_workspace = raw.get("workspace")
    proposed_raw["workspace"] = {
        **(original_workspace if isinstance(original_workspace, dict) else {}),
        "root": str(workspace),
    }
    config_values: dict[str, Any] = {"mode": proposed.config.mode}
    if proposed.config.root is not None:
        config_values["root"] = str(config_root)
    proposed_raw["config"] = config_values
    # Keep the original file, including its comments and formatting, intact
    # when the requested persistent selection is already in effect.
    unchanged = proposed_raw == raw
    # Preserve unrelated authored (whitelisted) preference values, and other
    # settings accepted by DeveloperLocalConfig. Do not add default fields.
    contents = (old_bytes.decode("utf-8") if unchanged else
                yaml.safe_dump(proposed_raw, sort_keys=False, allow_unicode=True))
    if not unchanged:
        # safe_dump does not retain comments. Restore the editor hint on every
        # deliberate edit so local.yml continues to validate against the
        # generated developer selector schema.
        header = schema_header(path, path.parent / "schema", "local-config.schema.json")
        if header:
            contents = header + "\n" + contents
    if not contents.endswith("\n"):
        contents += "\n"

    print(f"Developer settings file: {path}")
    print("Current selection:")
    print("  " + old_state.replace("\n", "\n  "))
    print("Proposed selection:")
    print(f"  Mode: {proposed.config.mode}")
    print(f"  Workspace: {workspace}")
    print(f"  Experiment root: {config_root}")
    print(f"  Config scaffold: {'present' if config_root.is_dir() else 'missing (no files will be created)'}")
    diff = list(difflib.unified_diff(
        old_bytes.decode("utf-8").splitlines(keepends=True),
        contents.splitlines(keepends=True),
        fromfile=str(path) + " (current)", tofile=str(path) + " (proposed)",
    ))
    print("Proposed file changes:")
    if diff:
        print("".join(diff), end="" if diff[-1].endswith("\n") else "\n")
    else:
        print("  No changes")
        return 0

    if not args.yes:
        answer = input("Write these persistent developer settings? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled; no files changed.")
            return 0
    # Detect a concurrent manual edit between preview and confirmation.
    if path.read_bytes() != old_bytes:
        raise ValueError(f"{path} changed during confirmation; retry settings set")
    _atomic_replace(path, contents)
    print(f"Updated: {path}")
    return 0
