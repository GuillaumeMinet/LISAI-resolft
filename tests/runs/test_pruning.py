from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lisai.runs.pruning import archive_run_directory, build_prune_plan
from lisai.runs.scanner import DiscoveredRun
from lisai.runs.schema import RunMetadata


def _discovered_run(
    tmp_path: Path,
    *,
    name: str,
    status: str = "completed",
    kept: bool = False,
    run_id: str = "01ARZ3NDEKTSV4RRFFQ69G5FAV",
) -> DiscoveredRun:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    terminal = status in {"completed", "stopped", "failed"}
    metadata = RunMetadata.model_validate(
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_name": name.rsplit("_", 1)[0],
            "run_index": int(name.rsplit("_", 1)[1]),
            "dataset": "Gag",
            "model_subfolder": "HDN",
            "kept": kept,
            "status": status,
            "closed_cleanly": terminal,
            "created_at": "2026-03-20T10:14:00Z",
            "updated_at": "2026-03-20T10:15:00Z",
            "ended_at": "2026-03-20T10:20:00Z" if terminal else None,
            "last_heartbeat_at": "2026-03-20T10:15:00Z",
            "path": f"datasets/Gag/models/HDN/{name}",
            "group_path": None,
        }
    )
    return DiscoveredRun(
        metadata=metadata,
        metadata_path=run_dir / ".lisai_run_meta.json",
        run_dir=run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path=metadata.path,
    )


def test_build_prune_plan_protects_kept_and_non_terminal_runs(tmp_path):
    candidate = _discovered_run(tmp_path, name="beta_00")
    kept = _discovered_run(
        tmp_path,
        name="beta_01",
        kept=True,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
    )
    running = _discovered_run(
        tmp_path,
        name="beta_02",
        status="running",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
    )

    plan = build_prune_plan([candidate, kept, running])

    assert plan.candidates == (candidate,)
    assert plan.kept == (kept,)
    assert plan.non_terminal == (running,)


def test_archive_run_directory_uses_local_timestamped_archive_and_avoids_collision(tmp_path):
    run_dir = tmp_path / "HDN" / "beta_02"
    run_dir.mkdir(parents=True)
    (run_dir / "payload.txt").write_text("first", encoding="utf-8")
    archived_at = datetime(2026, 9, 14, 10, 52, 30, tzinfo=timezone.utc)

    first = archive_run_directory(run_dir, archived_at=archived_at)

    assert first == tmp_path / "HDN" / "_archive" / "beta_02_archived_20260914-105230Z"
    assert (first / "payload.txt").read_text(encoding="utf-8") == "first"
    assert not run_dir.exists()

    run_dir.mkdir()
    (run_dir / "payload.txt").write_text("second", encoding="utf-8")
    second = archive_run_directory(run_dir, archived_at=archived_at)

    assert second.name == "beta_02_archived_20260914-105230Z_2"
    assert (second / "payload.txt").read_text(encoding="utf-8") == "second"
