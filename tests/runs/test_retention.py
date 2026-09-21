from __future__ import annotations

from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.retention import set_run_kept
from lisai.runs.schema import RunMetadata


def _write_metadata(run_dir):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            {
                "schema_version": 2,
                "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "run_name": run_name,
                "run_index": run_index,
                "dataset": "Gag",
                "model_subfolder": "HDN",
                "status": "completed",
                "closed_cleanly": True,
                "created_at": "2026-03-20T10:14:00Z",
                "updated_at": "2026-03-20T10:15:00Z",
                "ended_at": "2026-03-20T10:20:00Z",
                "last_heartbeat_at": "2026-03-20T10:15:00Z",
                "path": f"datasets/Gag/models/HDN/{run_dir.name}",
                "group_path": None,
            }
        ),
    )


def test_set_run_kept_only_changes_retention_metadata(tmp_path):
    run_dir = tmp_path / "beta_00"
    _write_metadata(run_dir)
    before = read_run_metadata(run_dir)

    updated = set_run_kept(run_dir, kept=True)
    persisted = read_run_metadata(run_dir)

    assert updated.kept is True
    assert persisted.kept is True
    assert persisted.updated_at >= before.updated_at
    assert persisted.last_heartbeat_at == before.last_heartbeat_at
    assert persisted.status == before.status
    assert persisted.run_id == before.run_id
