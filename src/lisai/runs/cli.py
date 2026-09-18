from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Iterable
from typing import Sequence

from lisai.infra.cli.open_path import try_open_path as _try_open_path
from lisai.infra.cli.prompts import is_interactive, prompt_yes_no
from lisai.infra.cli.selection import resolve_partial_name

from .listing import (
    filter_runs,
    has_path_inconsistencies,
    render_runs_table,
    render_external_runs_table,
    write_invalid_run_warnings,
)
from .plotting import show_loss_plot_for_run
from .scanner import DiscoveredRun, InvalidRunMetadata, ScanResults, scan_runs
from .schema import RUN_STATUSES
from .external import filter_external_runs, scan_external_runs
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
    kind: Sequence[str] | None = None,
    promoted: bool = False,
    kept: bool = False,
    full: bool = False,
    recent: int | None = None,
    live: bool = False,
    interval_seconds: float = 2.0,
    stdin=None,
    stdout=None,
    stderr=None,
) -> int:
    in_stream = sys.stdin if stdin is None else stdin
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

    if live and not is_interactive(out):
        print(
            "warning: --live requires interactive terminal output; showing a single snapshot instead.",
            file=err,
        )
        live = False

    initial_scan: ScanResults | None = None
    initial_external_scan = None
    resolved_dataset = dataset
    if dataset is not None:
        initial_scan = scan_runs()
        initial_external_scan = scan_external_runs()
        selected_kinds = set(kind or ("lisai", "external"))
        dataset_names = [run.dataset for run in initial_scan.runs] if "lisai" in selected_kinds else []
        if "external" in selected_kinds:
            dataset_names.extend(run.dataset for run in initial_external_scan.runs)
        resolved_dataset = resolve_partial_name(
            dataset,
            dataset_names,
            entity_name="dataset",
            stdin=in_stream,
            stdout=out,
            stderr=err,
            help_hint="Use 'lisai runs list' without --dataset to inspect available datasets and runs.",
        )
        if resolved_dataset is None:
            write_invalid_run_warnings(initial_scan.invalid, stderr=err)
            return 1

    if live:
        emitted_invalid_keys: set[tuple[str, str, str]] = set()
        try:
            while True:
                _render_runs_snapshot(
                    run_id=run_id,
                    run_dir_name=run_dir_name,
                    exp_name=exp_name,
                    dataset=resolved_dataset,
                    model_subfolder=model_subfolder,
                    status=status,
                    kind=kind,
                    promoted=promoted,
                    kept=kept,
                    full=full,
                    recent=recent,
                    stdout=out,
                    stderr=err,
                    live=True,
                    refresh_interval_seconds=resolved_interval_seconds,
                    emitted_invalid_keys=emitted_invalid_keys,
                    top_notice=interval_warning,
                    scan_result=initial_scan,
                    external_scan_result=initial_external_scan,
                )
                initial_scan = None
                initial_external_scan = None
                time.sleep(resolved_interval_seconds)
        except KeyboardInterrupt:
            # Keep shell prompt on a clean line after Ctrl+C in live mode.
            print(file=out, flush=True)
            return 0

    _render_runs_snapshot(
        run_id=run_id,
        run_dir_name=run_dir_name,
        exp_name=exp_name,
        dataset=resolved_dataset,
        model_subfolder=model_subfolder,
        status=status,
        kind=kind,
        promoted=promoted,
        kept=kept,
        full=full,
        recent=recent,
        stdout=out,
        stderr=err,
        live=False,
        refresh_interval_seconds=resolved_interval_seconds,
        emitted_invalid_keys=None,
        top_notice=interval_warning,
        scan_result=initial_scan,
        external_scan_result=initial_external_scan,
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
    kind: Sequence[str] | None,
    promoted: bool,
    kept: bool,
    full: bool,
    recent: int | None = None,
    stdout,
    stderr,
    live: bool,
    refresh_interval_seconds: float,
    emitted_invalid_keys: set[tuple[str, str, str]] | None,
    top_notice: str | None,
    scan_result: ScanResults | None = None,
    external_scan_result=None,
) -> None:
    resolved_scan = scan_runs() if scan_result is None else scan_result
    resolved_external_scan = scan_external_runs() if external_scan_result is None else external_scan_result
    filtered_runs = filter_runs(
        resolved_scan.runs,
        run_id=run_id,
        run_dir_name=run_dir_name,
        exp_name=exp_name,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
        kept=True if kept else None,
    )

    kinds = set(kind or ("lisai", "external"))
    if "lisai" not in kinds:
        filtered_runs = []
    external_runs = filter_external_runs(
        resolved_external_scan.runs,
        dataset=dataset,
        run_name=exp_name or run_dir_name,
    )
    # Filters tied to LISAI training metadata do not apply to imported external runs.
    if "external" not in kinds or status is not None or promoted or kept or model_subfolder is not None or run_id is not None:
        external_runs = []

    if promoted:
        from lisai.promoted_models.registry import promoted_source_run_ids

        promoted_ids = promoted_source_run_ids()
        filtered_runs = [run for run in filtered_runs if run.metadata.run_id in promoted_ids]

    if recent is not None:
        filtered_runs = filtered_runs[:recent]
        external_runs = external_runs[:recent]

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
            kind=kind,
            promoted=promoted,
            kept=kept,
            recent=recent,
            live=live,
            refresh_interval_seconds=refresh_interval_seconds,
        )
    )

    body = "No LISAI runs found." if external_runs else "No runs found."
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
    if external_runs:
        snapshot_lines.extend([
            "",
            "Imported external runs",
            "These runs were imported into LISAI; commands such as continue, evaluate, apply, etc. are not available.",
            render_external_runs_table(external_runs),
        ])
    snapshot = "\n".join(snapshot_lines)

    if live:
        _print_live_snapshot(snapshot, stdout=stdout)
    else:
        print(snapshot, file=stdout)

    if emitted_invalid_keys is None:
        write_invalid_run_warnings(resolved_scan.invalid, stderr=stderr)
        return

    new_invalid = _filter_new_invalid_warnings(resolved_scan.invalid, emitted_invalid_keys)
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
    kind: Sequence[str] | None,
    promoted: bool,
    kept: bool,
    recent: int | None = None,
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
    if kind:
        filter_parts.append(f"Kind: {', '.join(kind)}")
    if promoted:
        filter_parts.append("Promoted only")
    if kept:
        filter_parts.append("Kept only")
    if run_dir_name:
        filter_parts.append(f"run_dir='{run_dir_name}'")
    if exp_name:
        filter_parts.append(f"exp_name~='{exp_name}'")
    if run_id:
        filter_parts.append(f"run_id={run_id}")
    if recent is not None:
        filter_parts.append(f"Recent: {recent}")

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


def _seconds_value(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seconds value: {value!r}.") from exc
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError("Seconds value must be finite.")
    return seconds

def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}.") from exc

    if result <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")

    return result

def run_list_from_args(args: argparse.Namespace) -> int:
    return list_runs(
        run_id=args.run_id,
        run_dir_name=args.run_dir_name,
        exp_name=args.exp_name,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        status=args.status,
        kind=args.kind,
        promoted=args.promoted,
        kept=args.kept,
        full=args.full,
        recent=args.recent,
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


def _resolve_external_open_from_args(args: argparse.Namespace):
    if args.run_id is not None:
        return None
    selector = (args.run or "").strip().casefold()
    scan = scan_external_runs()
    candidates = list(scan.runs)
    if args.dataset:
        dataset_query = args.dataset.strip().casefold()
        dataset_names = sorted({run.dataset for run in candidates})
        exact = [name for name in dataset_names if name.casefold() == dataset_query]
        partial = [name for name in dataset_names if dataset_query in name.casefold()]
        names = exact or partial
        if len(names) == 1:
            candidates = [run for run in candidates if run.dataset == names[0]]
        elif not names:
            candidates = []
    if selector:
        exact = [run for run in candidates if run.name.casefold() == selector]
        candidates = exact or [run for run in candidates if selector in run.name.casefold()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(f"{run.dataset}/{run.name}" for run in candidates)
        print(f"External run selector is ambiguous: {names}", file=sys.stderr)
    return None


def run_open_from_args(args: argparse.Namespace) -> int:
    selected_external = None
    if getattr(args, "kind", None) != "lisai":
        selected_external = _resolve_external_open_from_args(args)
        if getattr(args, "kind", None) == "external" and selected_external is None:
            return 1

    if selected_external is not None:
        if _try_open_path(selected_external.run_dir):
            return 0
        print(selected_external.run_dir)
        return 0

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


def _run_set_kept_from_args(args: argparse.Namespace, *, kept: bool) -> int:
    selected = _resolve_run_from_args(args)
    if selected is None:
        return 1

    from .retention import set_run_kept

    was_kept = selected.metadata.kept
    set_run_kept(selected.run_dir, kept=kept)
    if was_kept == kept:
        state = "kept" if kept else "not kept"
        print(f"Run is already {state}: {selected.run_dir.name}")
    else:
        action = "Kept" if kept else "Unkept"
        print(f"{action} run: {selected.run_dir.name}")
    return 0


def run_keep_from_args(args: argparse.Namespace) -> int:
    return _run_set_kept_from_args(args, kept=True)


def run_unkeep_from_args(args: argparse.Namespace) -> int:
    return _run_set_kept_from_args(args, kept=False)


def run_prune_from_args(args: argparse.Namespace) -> int:
    from .pruning import (
        archive_run_directory,
        build_prune_plan,
        delete_run_directory,
    )
    from .schema import utc_now

    scan_result = scan_runs()
    resolved_dataset = args.dataset
    if args.dataset is not None:
        resolved_dataset = resolve_partial_name(
            args.dataset,
            (run.dataset for run in scan_result.runs),
            entity_name="dataset",
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            help_hint="Use 'lisai runs list' without --dataset to inspect available datasets and runs.",
        )
        if resolved_dataset is None:
            write_invalid_run_warnings(scan_result.invalid, stderr=sys.stderr)
            return 1
        args.dataset = resolved_dataset

    scoped_runs = filter_runs(
        scan_result.runs,
        run_id=args.run_id,
        run_dir_name=args.run_dir_name,
        exp_name=args.exp_name,
        dataset=resolved_dataset,
        model_subfolder=args.model_subfolder,
        status=args.status,
    )
    plan = build_prune_plan(scoped_runs)

    _print_prune_summary(args, plan)
    write_invalid_run_warnings(scan_result.invalid, stderr=sys.stderr)
    if not plan.candidates:
        print("No unkept terminal runs to prune.")
        return 0

    if not args.yes:
        prompt = (
            "Permanently delete these runs? [y/N] "
            if args.delete
            else "Archive these runs? [y/N] "
        )
        if not prompt_yes_no(prompt):
            print("Prune cancelled.")
            return 0

    failures = 0
    archived_at = utc_now()
    for run in plan.candidates:
        try:
            if args.delete:
                delete_run_directory(run.run_dir)
            else:
                archive_run_directory(run.run_dir, archived_at=archived_at)
        except OSError as exc:
            failures += 1
            print(f"Failed to prune {run.run_dir}: {exc}", file=sys.stderr)

    succeeded = len(plan.candidates) - failures
    action = "Deleted" if args.delete else "Archived"
    print(f"{action} {succeeded} run(s).")
    return 1 if failures else 0


def _print_prune_summary(args: argparse.Namespace, plan) -> None:
    print("LISAI runs prune")
    scope_parts: list[str] = []
    if args.dataset:
        scope_parts.append(f"dataset={args.dataset!r}")
    if args.model_subfolder:
        scope_parts.append(f"subfolder={args.model_subfolder!r}")
    if args.status:
        scope_parts.append(f"status={args.status!r}")
    if args.run_dir_name:
        scope_parts.append(f"run_dir={args.run_dir_name!r}")
    if args.exp_name:
        scope_parts.append(f"exp_name~={args.exp_name!r}")
    if args.run_id:
        scope_parts.append(f"run_id={args.run_id}")
    print(f"Scope: {' | '.join(scope_parts) if scope_parts else 'all discovered runs'}")
    print(f"Matched runs: {len(plan.matched)}")
    print(f"Kept (protected): {len(plan.kept)}")
    print(f"Non-terminal (protected): {len(plan.non_terminal)}")
    print(f"Candidates: {len(plan.candidates)}")
    print(
        "Action: permanently delete"
        if args.delete
        else "Action: archive to local _archive folders"
    )

    if plan.candidates:
        label = "Runs to delete:" if args.delete else "Runs to archive:"
        print(label)
        for run in plan.candidates:
            print(f"  {run.dataset}/{run.model_subfolder}/{run.run_dir.name}")


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
            inference_config=args.inference_config,
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
    parser.add_argument("--dataset", help="Filter runs by dataset name or unique partial name.")
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
        "--kind",
        choices=["lisai", "external"],
        action="append",
        help="Filter by run kind. Repeat to include multiple kinds.",
    )
    parser.add_argument(
        "--promoted",
        action="store_true",
        help="Show only runs that are the source of a locally promoted model.",
    )
    parser.add_argument(
        "--kept",
        action="store_true",
        help="Show only runs marked as kept.",
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
        "--recent",
        type=_positive_int,
        metavar="N",
        help="Show only the N most recently seen runs after applying other filters"
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
    parser.add_argument("--kind", choices=["lisai", "external"], help="Restrict run selection to one run kind.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.set_defaults(handler=run_open_from_args)
    return parser


def _add_runs_keep_arguments(
    parser: argparse.ArgumentParser,
    *,
    kept: bool,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: run_dir_name, partial exp_name, or dataset[/subfolder]/run_dir_name. "
            "Use --run-id as an alternative."
        ),
    )
    parser.add_argument("--run-id", help="Stable run identifier to select.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.set_defaults(handler=run_keep_from_args if kept else run_unkeep_from_args)
    return parser


def _add_runs_prune_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_run_filter_arguments(parser, include_identity=True, include_status=True)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete prune candidates instead of archiving them locally.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply the prune plan without the confirmation prompt.",
    )
    parser.set_defaults(handler=run_prune_from_args)
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
        "--inference-config",
        help=(
            "Optional inference config path or config name to store as the promoted model's "
            "default apply config. It can also be attached later with 'lisai models set-config'."
        ),
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

    keep_parser = runs_subparsers.add_parser(
        "keep",
        help="Mark a training run to be retained by pruning.",
        description="Mark a training run to be retained by pruning.",
    )
    _add_runs_keep_arguments(keep_parser, kept=True)

    unkeep_parser = runs_subparsers.add_parser(
        "unkeep",
        help="Remove the keep marker from a training run.",
        description="Remove the keep marker from a training run.",
    )
    _add_runs_keep_arguments(unkeep_parser, kept=False)

    prune_parser = runs_subparsers.add_parser(
        "prune",
        help="Archive or delete unkept training runs in a selected scope.",
        description="Archive or delete unkept training runs in a selected scope.",
    )
    _add_runs_prune_arguments(prune_parser)

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

    keep_parser = subparsers.add_parser(
        "keep",
        help="Mark a training run to be retained by pruning.",
        description="Mark a training run to be retained by pruning.",
    )
    _add_runs_keep_arguments(keep_parser, kept=True)

    unkeep_parser = subparsers.add_parser(
        "unkeep",
        help="Remove the keep marker from a training run.",
        description="Remove the keep marker from a training run.",
    )
    _add_runs_keep_arguments(unkeep_parser, kept=False)

    prune_parser = subparsers.add_parser(
        "prune",
        help="Archive or delete unkept training runs in a selected scope.",
        description="Archive or delete unkept training runs in a selected scope.",
    )
    _add_runs_prune_arguments(prune_parser)

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
