from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Sequence

from lisai.config import settings
from lisai.infra.cli.prompts import prompt_yes_no
from lisai.runs.cli import add_run_filter_arguments
from lisai.runs.listing import (
    is_run_likely_active,
    is_run_likely_stale,
)
from lisai.runs.scanner import DiscoveredRun
from lisai.runs.selection import resolve_discovered_run_selector

from .run_training import run_training_from_config_dict


def continue_run(
    *,
    selector: str | None = None,
    run_id: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    assume_yes: bool = False,
    force: bool = False,
    stdin=None,
    stdout=None,
    stderr=None,
    now: datetime | None = None,
    progress_bar: bool | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    selected_run = resolve_discovered_run_selector(
        selector=selector,
        run_id=run_id,
        dataset=dataset,
        model_subfolder=model_subfolder,
        stdin=in_stream,
        stdout=out,
        stderr=err,
    )
    if selected_run is None:
        return 1

    if is_run_likely_active(selected_run, now=now):
        timeout_minutes = settings.project_cfg.run_tracking.active_heartbeat_timeout_minutes
        if not force:
            print(
                "Selected run still appears active based on a recent heartbeat. "
                f"Refusing to continue. Rerun with --force if you are sure it is not actually running "
                f"(current timeout: {timeout_minutes} minutes).",
                file=err,
            )
            return 1
        print(
            "warning: forcing continuation even though the selected run still appears active "
            f"(heartbeat timeout: {timeout_minutes} minutes).",
            file=err,
        )
    elif is_run_likely_stale(selected_run, now=now):
        print(
            "warning: selected run appears stale (running + not closed cleanly + old heartbeat). "
            "Continuing will reuse the same run directory.",
            file=err,
        )

    already_confirmed = False
    if not selected_run.path_consistent:
        issue_text = ", ".join(selected_run.consistency_issues) or "unknown inconsistency"
        print(
            "warning: selected run has inconsistent path metadata "
            f"(likely moved/renamed folder): {issue_text}",
            file=err,
        )
        if assume_yes and force:
            print(
                "warning: continuing with --yes --force despite inconsistent path metadata.",
                file=err,
            )
            already_confirmed = True
        elif assume_yes:
            print(
                "Selected run has inconsistent path metadata. "
                "Rerun without --yes to confirm interactively, or use --yes --force.",
                file=err,
            )
            return 1
        else:
            confirmed = prompt_yes_no(
                "Selected run has inconsistent path metadata. Continue anyway? [y/N]: ",
                stdin=in_stream,
                stdout=out,
                require_interactive=True,
            )
            if confirmed is None:
                print(
                    "Confirmation required. Rerun with --yes --force to continue non-interactively.",
                    file=err,
                )
                return 1
            if not confirmed:
                print("Continue cancelled.", file=err)
                return 1
            already_confirmed = True

    if not assume_yes:
        prompt = None
        if not already_confirmed:
            prompt = "Continue training this run in place? [y/N]: "

        if prompt is not None:
            confirmed = prompt_yes_no(
                prompt,
                stdin=in_stream,
                stdout=out,
                require_interactive=True,
            )
            if confirmed is None:
                print("Confirmation required. Rerun with --yes to continue non-interactively.", file=err)
                return 1
            if not confirmed:
                print("Continue cancelled.", file=err)
                return 1

    config = _build_continue_training_config(selected_run)
    if progress_bar is None:
        run_training_from_config_dict(config)
    else:
        run_training_from_config_dict(config, progress_bar=progress_bar)
    return 0


def _build_continue_training_config(run: DiscoveredRun) -> dict:
    return {
        "experiment": {"mode": "continue_training"},
        "load_model": {
            "canonical_load": False,
            "model_full_path": str(run.run_dir.resolve()),
            "load_method": "state_dict",
            "best_or_last": "last",
        },
    }


def run_from_args(args: argparse.Namespace) -> int:
    return continue_run(
        selector=args.run,
        run_id=args.run_id,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        assume_yes=args.yes,
        force=args.force,
        progress_bar=getattr(args, "progress_bar", None),
    )


def add_continue_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("--run-id", help="Stable run identifier to continue in place.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow continuation even if the selected run still appears active from a recent heartbeat, "
            "and permit non-interactive continuation of path-inconsistent runs with --yes."
        ),
    )
    parser.add_argument(
        "--progress-bar",
        dest="progress_bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the local/configured tqdm progress-bar preference for the continued training run.",
    )
    return parser


def add_continue_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "continue",
        help="Continue an existing training run without changing its config.",
        description="Continue an existing training run without changing its config.",
    )
    add_continue_arguments(parser)
    parser.set_defaults(handler=run_from_args)
    return parser


def build_parser(*, prog: str = "lisai continue") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue an existing training run without changing its config.",
        prog=prog,
    )
    add_continue_arguments(parser)
    parser.set_defaults(handler=run_from_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = [
    "add_continue_subparser",
    "build_parser",
    "continue_run",
    "main",
]
