from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from lisai.config.cli import add_configs_subparser
from lisai.config.selection import (
    describe_locations,
    load_settings_for_selected_locations,
    resolve_cli_locations,
)
from lisai.data.cli import add_datasets_subparser
from lisai.evaluation.cli import add_apply_subparser, add_evaluate_subparser
from lisai.promoted_models.cli import add_models_subparser
from lisai.preprocess.cli import add_preprocess_subparser
from lisai.runs.cli import add_runs_subparser
from lisai.training.cli import add_train_subparser
from lisai.training.continue_cli import add_continue_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lisai", description="LISAI command line interface.")
    parser.add_argument("--workspace", metavar="PATH", help="Select workspace data and global settings.")
    parser.add_argument("--config-root", metavar="PATH", help="Select writable experiment configuration root.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    config_parser = subparsers.add_parser("config", help="Inspect selected configuration locations.")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("show", help="Show selected roots and their origins.")

    workspace_parser = subparsers.add_parser("workspace", help="Explicitly initialize, update or migrate workspaces.")
    workspace_actions = workspace_parser.add_subparsers(dest="workspace_action", required=True)
    init_parser = workspace_actions.add_parser("init", help="Initialize a workspace without replacing existing files.")
    init_parser.add_argument("path", nargs="?", help="Workspace path (defaults to the selected workspace).")
    update_parser = workspace_actions.add_parser("update", help="Back up and update workspace defaults explicitly.")
    update_parser.add_argument("--yes", action="store_true", help="Confirm replacing generated defaults.")
    update_parser.add_argument("--include-inference", action="store_true", help="Also back up and replace seeded local inference defaults.")
    migrate_parser = workspace_actions.add_parser("migrate", help="Convert legacy three-file settings without moving data.")
    migrate_parser.add_argument("--from-legacy", required=True, metavar="CONFIGS", help="Old configs/ directory.")
    dev_parser = subparsers.add_parser("dev", help="Explicit editable-checkout developer setup.")
    dev_actions = dev_parser.add_subparsers(dest="dev_action", required=True)
    dev_init = dev_actions.add_parser("init", help="Activate source checkout using local.yml.")
    dev_init.add_argument("--yes", action="store_true", help="Confirm initializing an absent workspace and developer configuration.")

    add_train_subparser(subparsers)
    add_continue_subparser(subparsers)
    add_configs_subparser(subparsers)
    add_datasets_subparser(subparsers)
    add_runs_subparser(subparsers)
    add_models_subparser(subparsers)
    add_preprocess_subparser(subparsers)
    add_apply_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command in ("workspace", "dev"):
        return _run_setup(args)
    # Catalog listing is independent of local configuration, even if it is absent.
    if (args.command == "models" and getattr(args, "models_command", None) == "catalog"
            or args.command == "datasets" and getattr(args, "datasets_command", None) == "catalog"):
        return args.handler(args)
    try:
        locations = resolve_cli_locations(workspace=args.workspace, config_root=args.config_root)
        if args.command == "config" and args.config_command == "show":
            print(describe_locations(locations))
            return 0
        settings = load_settings_for_selected_locations(locations)
        from lisai.config.workspace import revision_warning
        warning = revision_warning(locations.workspace_root)
        if warning:
            print(f"LISAI: {warning}", file=sys.stderr)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LISAI configuration: {exc}", file=sys.stderr)
        return 2
    # One selection at the operation boundary; direct module/API callers retain
    # their own optional default for backward compatibility.
    args._lisai_settings = settings
    args._lisai_locations = locations
    return args.handler(args)


def _run_setup(args: argparse.Namespace) -> int:
    """Explicit write commands; normal parsing and Settings remain read-only."""
    from lisai.config.selection import _source_checkout_for_import
    from lisai.config.workspace import (
        initialize_developer, initialize_workspace, migrate_legacy_workspace,
        update_workspace,
    )
    try:
        locations = resolve_cli_locations(workspace=args.workspace, config_root=args.config_root)
        root = locations.workspace_root
        if args.command == "workspace":
            if args.workspace_action == "init":
                root = Path(args.path).expanduser().resolve() if args.path else root
                created = initialize_workspace(root)
            elif args.workspace_action == "migrate":
                created = migrate_legacy_workspace(args.from_legacy, root)
            else:
                if not args.yes:
                    answer = input(
                        f"Back up and replace generated workspace defaults in {root}? [y/N] "
                    ).strip().lower()
                    if answer not in {"y", "yes"}:
                        print("Cancelled; no files changed.")
                        return 0
                backups = update_workspace(
                    root, config_root=locations.experiment_config_root,
                    replace_editable_defaults=args.include_inference,
                )
                for original, backup in backups:
                    print(f"Backup: {original} -> {backup}")
                print(f"Updated workspace defaults in {root} (revision recorded in workspace.yml)")
                return 0
        else:
            source = _source_checkout_for_import()
            if source is None:
                raise ValueError(
                    "Developer setup requires a verified editable LISAI checkout; "
                    "use --config-root or LISAI_CONFIG_ROOT for explicit configuration selection."
                )
            if not (root / "configs/workspace.yml").is_file() or not (root / "configs/data.yml").is_file():
                if not args.yes:
                    answer = input(f"Initialize workspace core in {root}? [y/N] ").strip().lower()
                    if answer not in {"y", "yes"}:
                        print("Cancelled; no files changed.")
                        return 0
                initialize_workspace(root, core_only=True)
            if not args.yes:
                answer = input(
                    f"Configure developer settings in {source / 'configs'} for workspace {root}? [y/N] "
                ).strip().lower()
                if answer not in {"y", "yes"}:
                    print("Cancelled; developer settings unchanged.")
                    return 0
            created = initialize_developer(source, root)
        for path in created:
            print(f"Created: {path}")
        print(f"Workspace: {root}; created {len(created)} new paths; existing files preserved.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"LISAI setup: {exc}", file=sys.stderr)
        return 2
