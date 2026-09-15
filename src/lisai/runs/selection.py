from __future__ import annotations

import sys
from collections.abc import Sequence

from lisai.infra.cli.selection import resolve_ambiguous_matches

from .listing import filter_runs, matches_exp_name, render_runs_table, write_invalid_run_warnings
from .scanner import DiscoveredRun, ScanResults, scan_runs

_RUN_SELECTOR_HINT = (
    "Rerun with --dataset/--subfolder or with --run-id to disambiguate."
)


def resolve_ambiguous_run_matches(
    matches: Sequence[DiscoveredRun],
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    rerun_hint: str = _RUN_SELECTOR_HINT,
) -> DiscoveredRun | None:
    return resolve_ambiguous_matches(
        matches,
        render_matches=lambda entries: render_runs_table(entries, include_selection_index=True),
        heading="Multiple matching runs found:",
        rerun_hint=rerun_hint,
        selection_name="run",
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )


def resolve_discovered_run_selector(
    *,
    selector: str | None = None,
    run_id: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    scan_result: ScanResults | None = None,
    allow_partial_exp_name: bool = True,
    emit_selected: bool = True,
    stdin=None,
    stdout=None,
    stderr=None,
    rerun_hint: str = _RUN_SELECTOR_HINT,
) -> DiscoveredRun | None:
    """Resolve a public CLI run selector to a discovered run.

    Supported selectors:
    - ``--run-id`` via ``run_id``
    - ``dataset[/subfolder]/run_dir_name`` via ``selector``
    - bare ``run_dir_name`` via ``selector``
    - bare partial semantic experiment name when no exact run directory matches
    """
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    normalized_selector = None if selector is None else selector.strip()
    if run_id is not None and normalized_selector:
        print("Use either a run selector or --run-id, not both.", file=err)
        return None

    if run_id is None and not normalized_selector:
        print(
            "Missing run selector. Use --run-id <run_id>, dataset[/subfolder]/run_dir_name, "
            "run_dir_name, or partial exp_name.",
            file=err,
        )
        return None

    # Scan once so invalid metadata warnings stay consistent with the selected run list.
    resolved_scan = scan_runs() if scan_result is None else scan_result

    if run_id is not None:
        matches = filter_runs(
            resolved_scan.runs,
            run_id=run_id,
            dataset=dataset,
            model_subfolder=model_subfolder,
        )
        selector_description = f"run_id={run_id!r}"
    else:
        assert normalized_selector is not None
        matches, selector_description = _select_runs_by_public_selector(
            resolved_scan.runs,
            normalized_selector,
            dataset=dataset,
            model_subfolder=model_subfolder,
            allow_partial_exp_name=allow_partial_exp_name,
            stderr=err,
        )
        if matches is None:
            return None

    if not matches:
        print(f"No matching run found for {selector_description}.", file=err)
        print("Use 'lisai runs list' to inspect available runs.", file=err)
        write_invalid_run_warnings(resolved_scan.invalid, stderr=err)
        return None

    if len(matches) == 1:
        selected = matches[0]
    else:
        selected = resolve_ambiguous_run_matches(
            matches,
            stdin=sys.stdin if stdin is None else stdin,
            stdout=out,
            stderr=err,
            rerun_hint=rerun_hint,
        )
        if selected is None:
            write_invalid_run_warnings(resolved_scan.invalid, stderr=err)
            return None

    if emit_selected:
        print("Selected run:", file=out)
        print(render_runs_table([selected]), file=out)
    write_invalid_run_warnings(resolved_scan.invalid, stderr=err)
    return selected


def _select_runs_by_public_selector(
    runs: Sequence[DiscoveredRun],
    selector: str,
    *,
    dataset: str | None,
    model_subfolder: str | None,
    allow_partial_exp_name: bool,
    stderr,
) -> tuple[list[DiscoveredRun] | None, str]:
    normalized = selector.replace("\\", "/")

    has_path_separator = "/" in normalized
    if has_path_separator:
        if dataset is not None or model_subfolder is not None:
            print(
                "--dataset/--subfolder cannot be combined with dataset[/subfolder]/run_dir_name selectors.",
                file=stderr,
            )
            return None, f"run={selector!r}"

        parts = [part for part in normalized.split("/") if part]
        if len(parts) < 2:
            print(
                "Run selector must be dataset[/subfolder]/run_dir_name.",
                file=stderr,
            )
            return None, f"run={selector!r}"

        selector_dataset = parts[0]
        selector_model_subfolder = "/".join(parts[1:-1])
        run_dir_name = parts[-1]

        matches = filter_runs(
            runs,
            run_dir_name=run_dir_name,
            dataset=selector_dataset,
            model_subfolder=selector_model_subfolder,
        )
        return matches, f"run={selector!r}"

    # Prefer an exact folder name before falling back to partial experiment names.
    exact_matches = filter_runs(
        runs,
        run_dir_name=selector,
        dataset=dataset,
        model_subfolder=model_subfolder,
    )
    if exact_matches or not allow_partial_exp_name:
        return exact_matches, f"run={selector!r}"

    scoped_runs = filter_runs(
        runs,
        dataset=dataset,
        model_subfolder=model_subfolder,
    )
    return [
        run
        for run in scoped_runs
        if matches_exp_name(run, selector)
    ], f"exp_name~={selector!r}"



__all__ = [
    "resolve_ambiguous_run_matches",
    "resolve_discovered_run_selector",
]
