from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil, floor
from statistics import median
import sys
from typing import Literal

from lisai.config import settings

from .scanner import DiscoveredRun, InvalidRunMetadata
from .schema import format_timestamp_local, utc_now

_STALE_TIMEOUT_MULTIPLIER = 1.05
_STALE_TIMEOUT_FLOOR_SECONDS = 60.0
_STALE_CRASH_RATIO = 5.0


@dataclass(frozen=True)
class _RunHeartbeatClassification:
    kind: Literal["not_running", "active", "stale", "crash"]
    stale_multiplier: int | None = None
    heartbeat_delay_s: float | None = None
    threshold_s: float | None = None


def active_heartbeat_timeout() -> timedelta:
    minutes = settings.project_cfg.run_tracking.active_heartbeat_timeout_minutes
    return timedelta(minutes=minutes)


def _normalized_reference_time(now: datetime | None) -> datetime:
    reference = utc_now() if now is None else now
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("Reference time must be timezone-aware.")
    return reference.astimezone(timezone.utc)


def _dynamic_heartbeat_timeout_seconds(run: DiscoveredRun) -> float | None:
    stats = run.metadata.live_runtime_stats
    if stats is None:
        return None

    recent = [float(value) for value in stats.recent_epoch_durations_s]
    if not recent and stats.last_epoch_duration_s is not None:
        recent = [float(stats.last_epoch_duration_s)]
    if not recent:
        return None

    return max(_STALE_TIMEOUT_FLOOR_SECONDS, float(median(recent)) * _STALE_TIMEOUT_MULTIPLIER)


def _heartbeat_timeout_seconds(run: DiscoveredRun) -> float:
    dynamic_seconds = _dynamic_heartbeat_timeout_seconds(run)
    if dynamic_seconds is not None:
        return dynamic_seconds
    return active_heartbeat_timeout().total_seconds()


def _classify_running_heartbeat(
    run: DiscoveredRun,
    *,
    now: datetime | None = None,
) -> _RunHeartbeatClassification:
    if run.metadata.status != "running" or run.metadata.closed_cleanly:
        return _RunHeartbeatClassification(kind="not_running")

    reference = _normalized_reference_time(now)
    threshold_s = _heartbeat_timeout_seconds(run)
    if run.metadata.last_heartbeat_at >= reference:
        return _RunHeartbeatClassification(
            kind="active",
            stale_multiplier=None,
            heartbeat_delay_s=0.0,
            threshold_s=threshold_s,
        )

    heartbeat_delay_s = (reference - run.metadata.last_heartbeat_at).total_seconds()
    ratio = heartbeat_delay_s / threshold_s
    if ratio < 1.0:
        return _RunHeartbeatClassification(
            kind="active",
            stale_multiplier=None,
            heartbeat_delay_s=heartbeat_delay_s,
            threshold_s=threshold_s,
        )
    if ratio > _STALE_CRASH_RATIO:
        return _RunHeartbeatClassification(
            kind="crash",
            stale_multiplier=None,
            heartbeat_delay_s=heartbeat_delay_s,
            threshold_s=threshold_s,
        )
    return _RunHeartbeatClassification(
        kind="stale",
        stale_multiplier=max(1, int(floor(ratio))),
        heartbeat_delay_s=heartbeat_delay_s,
        threshold_s=threshold_s,
    )


def filter_runs(
    runs: Iterable[DiscoveredRun],
    *,
    run_id: str | None = None,
    run_dir_name: str | None = None,
    exp_name: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
    kept: bool | None = None,
) -> list[DiscoveredRun]:
    return [
        run
        for run in runs
        if (run_id is None or run.metadata.run_id == run_id)
        and (run_dir_name is None or run.run_dir.name == run_dir_name)
        and matches_exp_name(run, exp_name)
        and (dataset is None or run.dataset == dataset)
        and (model_subfolder is None or run.model_subfolder == model_subfolder)
        and (status is None or run.metadata.status == status)
        and (kept is None or run.metadata.kept is kept)
    ]


def matches_exp_name(run: DiscoveredRun, query: str | None) -> bool:
    if query is None:
        return True
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return True
    return normalized_query in run.metadata.run_name.casefold()


def is_run_heartbeat_fresh(run: DiscoveredRun, *, now: datetime | None = None) -> bool:
    reference = _normalized_reference_time(now)
    if run.metadata.last_heartbeat_at >= reference:
        return True
    return (reference - run.metadata.last_heartbeat_at).total_seconds() < _heartbeat_timeout_seconds(run)


def is_run_likely_active(run: DiscoveredRun, *, now: datetime | None = None) -> bool:
    return _classify_running_heartbeat(run, now=now).kind == "active"


def is_run_likely_stale(run: DiscoveredRun, *, now: datetime | None = None) -> bool:
    return _classify_running_heartbeat(run, now=now).kind in {"stale", "crash"}


def display_run_status(run: DiscoveredRun, *, now: datetime | None = None) -> str:
    classification = _classify_running_heartbeat(run, now=now)
    if classification.kind == "stale":
        return f"stale (x{classification.stale_multiplier})"
    if classification.kind == "crash":
        return "crash"
    return run.metadata.status


def render_runs_table(
    runs: Sequence[DiscoveredRun],
    *,
    now: datetime | None = None,
    full: bool = False,
    include_selection_index: bool = False,
) -> str:
    if not runs:
        return ""

    reference = utc_now() if now is None else now
    headers = [
        "dataset",
        "model_subfolder",
        "run_dir",
        "kept",
        "status",
        "epoch",
        "eta_left",
        "start_time"
    ]
    if full:
        headers.extend(
            [
                "last_seen",
                "run_id",
                "failure",
                "path_consistent",
                "closed_cleanly",
            ]
        )
    selection_width = max(2, len(str(len(runs))))
    rows: list[list[str]] = []
    for selection_idx, run in enumerate(runs, start=1):
        row = [
            run.dataset,
            run.model_subfolder,
            run.run_dir.name,
            "*" if run.metadata.kept else "",
            display_run_status(run, now=reference),
            _format_epoch(run),
            _format_eta_left(run),
            format_timestamp_local(run.metadata.created_at),
        ]
        if full:
            row.extend(
                [
                    format_timestamp_local(run.last_seen),
                    run.metadata.run_id,
                    _format_failure_summary(run),
                    str(run.path_consistent).lower(),
                    str(run.metadata.closed_cleanly).lower(),
                ]
            )
        if include_selection_index:
            row = [f"{selection_idx:0{selection_width}d}"] + row
        rows.append(row)

    if include_selection_index:
        headers = ["#"] + headers

    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def write_invalid_run_warnings(
    invalid_runs: Iterable[InvalidRunMetadata],
    *,
    stderr=None,
) -> None:
    err = sys.stderr if stderr is None else stderr
    for invalid in invalid_runs:
        print(
            f"warning: skipped invalid run metadata {invalid.metadata_path} "
            f"({invalid.kind}: {invalid.message})",
            file=err,
        )


def has_path_inconsistencies(runs: Iterable[DiscoveredRun]) -> bool:
    return any(not run.path_consistent for run in runs)


def _format_epoch(run: DiscoveredRun) -> str:
    if run.metadata.last_epoch is None and run.metadata.max_epoch is None:
        return "-"
    # Internally, last_epoch is zero-based (checkpoint index). For user-facing
    # CLI output, display one-based epoch numbers.
    last_epoch = "-" if run.metadata.last_epoch is None else str(run.metadata.last_epoch + 1)
    max_epoch = "-" if run.metadata.max_epoch is None else str(run.metadata.max_epoch)
    return f"{last_epoch}/{max_epoch}"


def _format_eta_left(run: DiscoveredRun) -> str:
    seconds_left = _estimate_seconds_left(run)
    if seconds_left is None:
        return "-"

    total_minutes = max(1, int(ceil(seconds_left / 60.0)))
    days, rem_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    return f"{days}d{hours}h{minutes}m"


def _estimate_seconds_left(run: DiscoveredRun) -> float | None:
    metadata = run.metadata
    if metadata.status != "running" or metadata.closed_cleanly:
        return None
    if metadata.max_epoch is None:
        return None

    completed_epochs = 0 if metadata.last_epoch is None else metadata.last_epoch + 1
    remaining_epochs = metadata.max_epoch - completed_epochs
    if remaining_epochs <= 0:
        return None

    mean_epoch_duration_s = _mean_epoch_duration_seconds(run)
    if mean_epoch_duration_s is None or mean_epoch_duration_s <= 0:
        return None
    return float(remaining_epochs) * mean_epoch_duration_s


def _mean_epoch_duration_seconds(run: DiscoveredRun) -> float | None:
    live_stats = run.metadata.live_runtime_stats
    if live_stats is not None:
        recent = [float(value) for value in live_stats.recent_epoch_durations_s]
        if recent:
            return float(sum(recent) / len(recent))
        if live_stats.last_epoch_duration_s is not None:
            return float(live_stats.last_epoch_duration_s)
        if live_stats.median_epoch_duration_s is not None:
            return float(live_stats.median_epoch_duration_s)

    runtime_stats = run.metadata.runtime_stats
    if runtime_stats is not None and runtime_stats.training_time_per_epoch_sec is not None:
        return float(runtime_stats.training_time_per_epoch_sec)
    return None


def _format_failure_summary(run: DiscoveredRun) -> str:
    reason = (run.metadata.failure_reason or "").strip()
    if not reason:
        return "-"
    return reason if len(reason) <= 60 else f"{reason[:57]}..."


__all__ = [
    "active_heartbeat_timeout",
    "display_run_status",
    "filter_runs",
    "has_path_inconsistencies",
    "is_run_heartbeat_fresh",
    "is_run_likely_active",
    "is_run_likely_stale",
    "matches_exp_name",
    "render_runs_table",
    "write_invalid_run_warnings",
]
