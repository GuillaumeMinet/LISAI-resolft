from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

from .listing import (
    filter_runs,
    has_path_inconsistencies,
    render_runs_table,
    write_invalid_run_warnings,
)
from .plotting import show_loss_plot_for_run
from .scanner import DiscoveredRun, InvalidRunMetadata, scan_runs
from .schema import RUN_STATUSES
from .selection import resolve_discovered_run_selector

_LIVE_INTERVAL_MIN_SECONDS = 1.0


def list_runs(
    *,
    run_id: str | None = None,
    run_dir_name: str | None = None,
    exp_name: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
    promoted: bool = False,
    full: bool = False,
    live: bool = False,
    interval_seconds: float = 2.0,
    stdout=None,
    stderr=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    if not math.isfinite(interval_seconds):
        raise ValueError("interval_seconds must be finite.")

    resolved_interval_seconds = interval_seconds
    interval_warning: str | None = None
    if live and interval_seconds < _LIVE_INTERVAL_MIN_SECONDS:
        resolved_interval_seconds = _LIVE_INTERVAL_MIN_SECONDS
        interval_warning = (
            f"warning: --interval {interval_seconds:g}s is below the minimum "
            f"{_LIVE_INTERVAL_MIN_SECONDS:g}s; using {_LIVE_INTERVAL_MIN_SECONDS:g}s."
        )

    if live and not _is_interactive(out):
        print(
            "warning: --live requires interactive terminal output; showing a single snapshot instead.",
            file=err,
        )
        live = False

    if live:
        emitted_invalid_keys: set[tuple[str, str, str]] = set()
        try:
            while True:
                _render_runs_snapshot(
                    run_id=run_id,
                    run_dir_name=run_dir_name,
                    exp_name=exp_name,
                    dataset=dataset,
                    model_subfolder=model_subfolder,
                    status=status,
                    promoted=promoted,
                    full=full,
                    stdout=out,
                    stderr=err,
                    live=True,
                    refresh_interval_seconds=resolved_interval_seconds,
                    emitted_invalid_keys=emitted_invalid_keys,
                    top_notice=interval_warning,
                )
                time.sleep(resolved_interval_seconds)
        except KeyboardInterrupt:
            # Keep shell prompt on a clean line after Ctrl+C in live mode.
            print(file=out, flush=True)
            return 0

    _render_runs_snapshot(
        run_id=run_id,
        run_dir_name=run_dir_name,
        exp_name=exp_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
        promoted=promoted,
        full=full,
        stdout=out,
        stderr=err,
        live=False,
        refresh_interval_seconds=resolved_interval_seconds,
        emitted_invalid_keys=None,
        top_notice=interval_warning,
    )
    return 0


def _render_runs_snapshot(
    *,
    run_id: str | None,
    run_dir_name: str | None,
    exp_name: str | None,
    dataset: str | None,
    model_subfolder: str | None,
    status: str | None,
    promoted: bool,
    full: bool,
    stdout,
    stderr,
    live: bool,
    refresh_interval_seconds: float,
    emitted_invalid_keys: set[tuple[str, str, str]] | None,
    top_notice: str | None,
) -> None:
    scan_result = scan_runs()
    filtered_runs = filter_runs(
        scan_result.runs,
        run_id=run_id,
        run_dir_name=run_dir_name,
        exp_name=exp_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
    )

    if promoted:
        from lisai.promoted_models.registry import promoted_source_run_ids

        promoted_ids = promoted_source_run_ids()
        filtered_runs = [run for run in filtered_runs if run.metadata.run_id in promoted_ids]

    snapshot_lines: list[str] = []
    if top_notice is not None:
        snapshot_lines.append(top_notice)
    snapshot_lines.append(
        _format_listing_title(
            run_id=run_id,
            run_dir_name=run_dir_name,
            exp_name=exp_name,
            dataset=dataset,
            model_subfolder=model_subfolder,
            status=status,
            promoted=promoted,
            live=live,
            refresh_interval_seconds=refresh_interval_seconds,
        )
    )

    body = "No runs found."
    if filtered_runs:
        body = render_runs_table(filtered_runs, full=full)
        if has_path_inconsistencies(filtered_runs):
            body = "\n".join(
                [
                    body,
                    "",
                    "Some listed runs have inconsistent path metadata (likely moved/renamed folders).",
                ]
            )

    snapshot_lines.append(body)
    snapshot = "\n".join(snapshot_lines)

    if live:
        _print_live_snapshot(snapshot, stdout=stdout)
    else:
        print(snapshot, file=stdout)

    if emitted_invalid_keys is None:
        write_invalid_run_warnings(scan_result.invalid, stderr=stderr)
        return

    new_invalid = _filter_new_invalid_warnings(scan_result.invalid, emitted_invalid_keys)
    if new_invalid:
        write_invalid_run_warnings(new_invalid, stderr=stderr)


def _format_listing_title(
    *,
    run_id: str | None,
    run_dir_name: str | None,
    exp_name: str | None,
    dataset: str | None,
    model_subfolder: str | None,
    status: str | None,
    promoted: bool,
    live: bool,
    refresh_interval_seconds: float,
) -> str:
    filter_parts: list[str] = []
    if dataset:
        filter_parts.append(f"Dataset: '{dataset}'")
    if model_subfolder:
        filter_parts.append(f"Subfolder: '{model_subfolder}'")
    if status:
        filter_parts.append(f"Status: '{status}'")
    if promoted:
        filter_parts.append("Promoted only")
    if run_dir_name:
        filter_parts.append(f"run_dir='{run_dir_name}'")
    if exp_name:
        filter_parts.append(f"exp_name~='{exp_name}'")
    if run_id:
        filter_parts.append(f"run_id={run_id}")

    title = "LISAI runs listing"
    if filter_parts:
        title = f"{title} - {' | '.join(filter_parts)}\n"
    if live:
        title = f"{title}LIVE MODE ({refresh_interval_seconds:g}s refresh) - Ctrl+C to stop live\n"
    return title


def _print_live_snapshot(snapshot: str, *, stdout) -> None:
    # ANSI: move cursor to top-left and clear the rest of the screen.
    stdout.write("\x1b[H\x1b[J")
    stdout.write(snapshot)
    stdout.write("\n")
    flush = getattr(stdout, "flush", None)
    if callable(flush):
        flush()


def _filter_new_invalid_warnings(
    invalid_runs: Iterable[InvalidRunMetadata],
    seen_keys: set[tuple[str, str, str]],
) -> list[InvalidRunMetadata]:
    new_entries: list[InvalidRunMetadata] = []
    for invalid in invalid_runs:
        key = (str(invalid.metadata_path), invalid.kind, invalid.message)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_entries.append(invalid)
    return new_entries


def _is_interactive(stream: Any) -> bool:
    is_tty = getattr(stream, "isatty", None)
    return callable(is_tty) and bool(is_tty())


def _seconds_value(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seconds value: {value!r}.") from exc
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError("Seconds value must be finite.")
    return seconds


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_runs(
        run_id=args.run_id,
        run_dir_name=args.run_dir_name,
        exp_name=args.exp_name,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        status=args.status,
        promoted=args.promoted,
        full=args.full,
        live=args.live,
        interval_seconds=args.interval,
    )


def _resolve_run_from_args(args: argparse.Namespace) -> DiscoveredRun | None:
    return resolve_discovered_run_selector(
        selector=args.run,
        run_id=args.run_id,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _try_open_path(path: Path) -> bool:
    resolved = path.resolve()

    startfile = getattr(os, "startfile", None)
    if callable(startfile):
        try:
            startfile(str(resolved))
            return True
        except OSError:
            pass

    commands: list[list[str]] = []
    explorer = shutil.which("explorer.exe") or shutil.which("explorer")
    if explorer is not None:
        target = _to_windows_path(resolved)
        commands.append([explorer, target if target is not None else str(resolved)])

    xdg_open = shutil.which("xdg-open")
    if xdg_open is not None:
        commands.append([xdg_open, str(resolved)])

    for command in commands:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False


def _to_windows_path(path: Path) -> str | None:
    wslpath_cmd = shutil.which("wslpath")
    if wslpath_cmd is None:
        return None
    try:
        completed = subprocess.run(
            [wslpath_cmd, "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    converted = completed.stdout.strip()
    return converted or None


def run_open_from_args(args: argparse.Namespace) -> int:
    selected = _resolve_run_from_args(args)
    if selected is None:
        return 1
    if _try_open_path(selected.run_dir):
        return 0
    print(selected.run_dir)
    return 0


def run_plot_from_args(args: argparse.Namespace) -> int:
    selected = _resolve_run_from_args(args)
    if selected is None:
        return 1

    architecture = None
    if selected.metadata.training_signature is not None:
        architecture = selected.metadata.training_signature.architecture
    return show_loss_plot_for_run(
        run_dir=selected.run_dir,
        dataset=selected.dataset,
        model_subfolder=selected.model_subfolder,
        architecture=architecture,
        stderr=sys.stderr,
        open_saved_plot=_try_open_path,
    )


def run_promote_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    selected = _resolve_run_from_args(args)
    if selected is None:
        return 1

    from lisai.promoted_models import promote_run

    try:
        promoted = promote_run(
            selected.run_dir,
            name=args.name,
            checkpoint=args.checkpoint,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")

    print(f"Promoted model: {promoted.manifest.name}")
    print(f"Model directory: {promoted.model_dir}")
    print(f"Source run: {promoted.manifest.source.run_id}")
    return 0


def add_run_filter_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_identity: bool = True,
    include_status: bool = False,
) -> argparse.ArgumentParser:
    if include_identity:
        parser.add_argument("--run-id", help="Filter runs by stable run_id.")
        parser.add_argument(
            "--run-dir",
            "--run_dir",
            dest="run_dir_name",
            help="Filter runs by full run folder name.",
        )
        parser.add_argument(
            "--exp-name",
            "--exp_name",
            dest="exp_name",
            help="Partially filter runs by semantic experiment name.",
        )
    parser.add_argument("--dataset", help="Filter runs by dataset name.")
    parser.add_argument(
        "--model-subfolder",
        "--models-subfolder",
        "--subfolder",
        dest="model_subfolder",
        help="Filter runs by training model_subfolder.",
    )
    if include_status:
        parser.add_argument("--status", choices=RUN_STATUSES, help="Filter runs by persisted status.")
    return parser


def _add_runs_list_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_run_filter_arguments(parser, include_status=True)
    parser.add_argument(
        "--promoted",
        action="store_true",
        help="Show only runs that are the source of a locally promoted model.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Include extra metadata columns "
            "(failure, path_consistent, closed_cleanly, start_time, last_seen, run_id)."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refresh the runs table continuously (interactive terminals only).",
    )
    parser.add_argument(
        "--interval",
        type=_seconds_value,
        default=2.0,
        metavar="SECONDS",
        help="Refresh interval for --live mode (default: 2.0).",
    )
    parser.set_defaults(handler=run_list_from_args)
    return parser


def _add_runs_plot_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("--run-id", help="Stable run identifier to plot.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.set_defaults(handler=run_plot_from_args)
    return parser


def _add_runs_open_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("--run-id", help="Stable run identifier to open.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.set_defaults(handler=run_open_from_args)
    return parser


def _add_runs_promote_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("--run-id", help="Stable run identifier to promote.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.add_argument(
        "--name",
        required=True,
        help="Unique public name for the locally promoted model.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="State-dict checkpoint to promote (default: best).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing locally promoted model with the same name.",
    )
    parser.set_defaults(handler=lambda args, p=parser: run_promote_from_args(args, p))
    return parser


def add_runs_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "runs",
        help="Inspect locally tracked training runs.",
        description="Inspect locally tracked training runs.",
    )
    runs_subparsers = parser.add_subparsers(dest="runs_command")
    runs_subparsers.required = True

    list_parser = runs_subparsers.add_parser(
        "list",
        help="List locally tracked training runs.",
        description="List locally tracked training runs.",
    )
    _add_runs_list_arguments(list_parser)

    plot_parser = runs_subparsers.add_parser(
        "plot",
        help="Plot train/val losses for a selected run.",
        description="Plot train/val losses for a selected run.",
    )
    _add_runs_plot_arguments(plot_parser)

    open_parser = runs_subparsers.add_parser(
        "open",
        help="Open a selected run folder in file explorer.",
        description="Open a selected run folder in file explorer.",
    )
    _add_runs_open_arguments(open_parser)

    promote_parser = runs_subparsers.add_parser(
        "promote",
        help="Promote a training run into the local reusable model library.",
        description="Promote a completed or stopped run into the local reusable model library.",
    )
    _add_runs_promote_arguments(promote_parser)
    return parser


def build_parser(*, prog: str = "lisai runs") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect locally tracked training runs.", prog=prog)
    subparsers = parser.add_subparsers(dest="runs_command")
    subparsers.required = True

    list_parser = subparsers.add_parser(
        "list",
        help="List locally tracked training runs.",
        description="List locally tracked training runs.",
    )
    _add_runs_list_arguments(list_parser)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot train/val losses for a selected run.",
        description="Plot train/val losses for a selected run.",
    )
    _add_runs_plot_arguments(plot_parser)

    open_parser = subparsers.add_parser(
        "open",
        help="Open a selected run folder in file explorer.",
        description="Open a selected run folder in file explorer.",
    )
    _add_runs_open_arguments(open_parser)

    promote_parser = subparsers.add_parser(
        "promote",
        help="Promote a training run into the local reusable model library.",
        description="Promote a completed or stopped run into the local reusable model library.",
    )
    _add_runs_promote_arguments(promote_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = ["add_run_filter_arguments", "add_runs_subparser", "build_parser", "list_runs", "main"]
