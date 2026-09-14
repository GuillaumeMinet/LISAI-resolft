from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import pytest

from lisai.config import settings
from lisai.runs.listing import (
    display_run_status,
    is_run_likely_active,
    is_run_likely_stale,
    render_runs_table,
)
from lisai.runs.scanner import DiscoveredRun
from lisai.runs.schema import RunMetadata


def _running_run(*, run_id: str, heartbeat: datetime, live_runtime_stats=None) -> DiscoveredRun:
    metadata = RunMetadata.model_validate(
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_name": "demo",
            "run_index": 0,
            "dataset": "Gag",
            "model_subfolder": "HDN",
            "kept": True,
            "status": "running",
            "closed_cleanly": False,
            "created_at": "2026-03-20T10:00:00Z",
            "updated_at": "2026-03-20T10:00:00Z",
            "ended_at": None,
            "last_heartbeat_at": heartbeat,
            "last_epoch": 1,
            "max_epoch": 10,
            "best_val_loss": 0.1,
            "path": "datasets/Gag/models/HDN/demo_00",
            "group_path": None,
            "live_runtime_stats": live_runtime_stats,
        }
    )
    run_dir = Path("/tmp/Gag/demo_00")
    return DiscoveredRun(
        metadata=metadata,
        metadata_path=run_dir / ".lisai_run_meta.json",
        run_dir=run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/demo_00",
        path_consistent=True,
        consistency_issues=(),
    )


def _table_value(table: str, column: str) -> str:
    lines = table.splitlines()
    headers = [part for part in re.split(r"\s{2,}", lines[0].strip()) if part]
    values = [part for part in re.split(r"\s{2,}", lines[2].strip()) if part]
    return values[headers.index(column)]


def test_listing_fallback_uses_config_timeout_when_live_stats_missing(monkeypatch):
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    run = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
        heartbeat=now - timedelta(minutes=10),
        live_runtime_stats=None,
    )
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)

    assert is_run_likely_active(run, now=now) is False
    assert is_run_likely_stale(run, now=now) is True
    assert display_run_status(run, now=now) == "stale (x1)"


def test_listing_dynamic_timeout_uses_median_of_recent_window_with_floor():
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    run_two_epochs = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
        heartbeat=now - timedelta(seconds=165),
        live_runtime_stats={
            "last_epoch_duration_s": 200.0,
            "recent_epoch_durations_s": [100.0, 200.0],
        },
    )
    run_floor = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
        heartbeat=now - timedelta(seconds=61),
        live_runtime_stats={
            "last_epoch_duration_s": 20.0,
            "recent_epoch_durations_s": [20.0],
        },
    )

    # median([100, 200]) = 150 -> threshold = 165 -> stale at boundary.
    assert display_run_status(run_two_epochs, now=now) == "stale (x1)"
    # threshold floor applies (max(60, 20 * 1.1) = 60) -> stale after 61s.
    assert display_run_status(run_floor, now=now) == "stale (x1)"


@pytest.mark.parametrize(
    ("delay_seconds", "expected"),
    [
        (600, "stale (x1)"),
        (3000, "stale (x5)"),
        (3001, "crash"),
    ],
)
def test_listing_stale_and_crash_labels_follow_multiplier_rule(monkeypatch, delay_seconds: int, expected: str):
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    run = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAD",
        heartbeat=now - timedelta(seconds=delay_seconds),
        live_runtime_stats=None,
    )
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)

    assert display_run_status(run, now=now) == expected


def test_render_runs_table_displays_eta_left_from_mean_recent_epoch_durations():
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    run = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAE",
        heartbeat=now - timedelta(seconds=5),
        live_runtime_stats={
            "last_epoch_duration_s": 200.0,
            "recent_epoch_durations_s": [100.0, 200.0],
        },
    )
    # completed=2 (zero-based last_epoch=1), max=10 => remaining=8.
    # mean recent epoch duration=150s => eta=1200s=20m.
    table = render_runs_table([run], now=now)

    assert "eta_left" in table.splitlines()[0]
    assert _table_value(table, "eta_left") == "0d0h20m"


def test_render_runs_table_displays_dash_eta_left_when_not_running_or_stats_missing():
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    running_missing_stats = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAF",
        heartbeat=now - timedelta(seconds=5),
        live_runtime_stats=None,
    )
    table_running = render_runs_table([running_missing_stats], now=now)
    assert _table_value(table_running, "eta_left") == "-"

    completed_payload = running_missing_stats.metadata.model_dump()
    completed_payload.update(
        {
            "status": "completed",
            "closed_cleanly": True,
            "ended_at": now,
        }
    )
    completed_run = DiscoveredRun(
        metadata=RunMetadata.model_validate(completed_payload),
        metadata_path=running_missing_stats.metadata_path,
        run_dir=running_missing_stats.run_dir,
        dataset=running_missing_stats.dataset,
        model_subfolder=running_missing_stats.model_subfolder,
        group_path=running_missing_stats.group_path,
        path=running_missing_stats.path,
        path_consistent=running_missing_stats.path_consistent,
        consistency_issues=running_missing_stats.consistency_issues,
    )
    table_completed = render_runs_table([completed_run], now=now)
    assert _table_value(table_completed, "eta_left") == "-"


def test_display_run_status_preserves_paused_state_label():
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    run = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB2",
        heartbeat=now - timedelta(seconds=5),
        live_runtime_stats=None,
    )
    paused = run.metadata.model_copy(update={"status": "paused"})
    paused_run = DiscoveredRun(
        metadata=paused,
        metadata_path=run.metadata_path,
        run_dir=run.run_dir,
        dataset=run.dataset,
        model_subfolder=run.model_subfolder,
        group_path=run.group_path,
        path=run.path,
        path_consistent=run.path_consistent,
        consistency_issues=run.consistency_issues,
    )
    assert display_run_status(paused_run, now=now) == "paused"


def test_render_runs_table_full_includes_start_time_last_seen_and_run_id(monkeypatch):
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    run = _running_run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
        heartbeat=now - timedelta(seconds=5),
        live_runtime_stats={
            "last_epoch_duration_s": 120.0,
            "recent_epoch_durations_s": [120.0],
        },
    )

    def _fake_local_time(value: datetime) -> str:
        if value == run.metadata.created_at:
            return "START_TS"
        return "LAST_TS"

    monkeypatch.setattr("lisai.runs.listing.format_timestamp_local", _fake_local_time)

    table = render_runs_table([run], now=now, full=True)
    header = table.splitlines()[0].strip()
    assert header.endswith(
        "failure  path_consistent  closed_cleanly  start_time  last_seen  run_id"
    )
    assert _table_value(table, "failure") == "-"
    assert _table_value(table, "start_time") == "START_TS"
    assert _table_value(table, "last_seen") == "LAST_TS"
    assert _table_value(table, "run_id") == run.metadata.run_id
