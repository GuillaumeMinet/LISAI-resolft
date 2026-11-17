from __future__ import annotations

import argparse
import sys
from typing import Sequence

from lisai.config.selection import load_settings_for_selected_locations, resolve_cli_locations
from lisai.infra.paths import Paths
from lisai_runner.queue.cli import add_queue_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lisai-runner", description="LISAI RUNNER command line interface.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    add_queue_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        locations = resolve_cli_locations()
        settings = load_settings_for_selected_locations(locations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"LISAI runner configuration: {exc}", file=sys.stderr)
        return 2
    args._lisai_paths = Paths(settings)
    args._lisai_locations = locations
    return args.handler(args)
