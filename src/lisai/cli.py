from __future__ import annotations

import argparse
import sys
from typing import Sequence

from lisai.config.cli import add_configs_subparser
from lisai.config.selection import (
    describe_locations,
    load_legacy_settings_for_selected_locations,
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
    # Catalog listing is independent of local configuration, even if it is absent.
    if (args.command == "models" and getattr(args, "models_command", None) == "catalog"
            or args.command == "datasets" and getattr(args, "datasets_command", None) == "catalog"):
        return args.handler(args)
    try:
        locations = resolve_cli_locations(workspace=args.workspace, config_root=args.config_root)
        if args.command == "config" and args.config_command == "show":
            print(describe_locations(locations))
            return 0
        settings = load_legacy_settings_for_selected_locations(locations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LISAI configuration: {exc}", file=sys.stderr)
        return 2
    # One selection at the operation boundary; direct module/API callers retain
    # their own optional default for backward compatibility.
    args._lisai_settings = settings
    args._lisai_locations = locations
    return args.handler(args)
