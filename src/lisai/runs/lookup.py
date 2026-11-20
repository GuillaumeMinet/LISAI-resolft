"""Non-interactive, workspace-scoped lookup of saved training runs."""

from lisai.infra.paths import Paths
from lisai.model_sources import RunRef
from lisai.runs import scanner
from lisai.runs.scanner import DiscoveredRun


def resolve_run_ref(ref: RunRef, *, paths: Paths) -> DiscoveredRun:
    """Resolve a run exactly once, never using interactive CLI selection."""
    scanned = scanner.scan_runs(paths=paths)
    matches = [run for run in scanned.runs if run.metadata.run_id == ref.run_id]
    if not matches:
        raise FileNotFoundError(
            f"Training run {ref.run_id!r} was not found in workspace {paths.workspace_root()}. "
            "Use 'lisai runs list' to inspect saved runs."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Training run ID {ref.run_id!r} is ambiguous ({len(matches)} matches) "
            "in the selected workspace."
        )
    selected = matches[0]
    if not selected.path_consistent:
        raise ValueError(
            f"Training run {ref.run_id!r} has inconsistent metadata/path: "
            + "; ".join(selected.consistency_issues)
        )
    return selected


__all__ = ["resolve_run_ref"]
