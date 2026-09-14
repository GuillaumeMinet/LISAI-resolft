from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.runs.cli as runs_cli
import lisai.runs.selection as selection_mod
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now


def _write_metadata(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    model_subfolder: str,
    status: str = "completed",
    last_heartbeat_at=None,
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    now = utc_now()
    heartbeat = last_heartbeat_at if last_heartbeat_at is not None else now
    created_at = heartbeat - timedelta(minutes=5)
    updated_at = heartbeat
    ended_at = None if status == "running" else heartbeat

    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": status != "running",
        "created_at": created_at,
        "updated_at": updated_at,
        "ended_at": ended_at,
        "last_heartbeat_at": heartbeat,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_runs_open_accepts_run_dir_selector(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G9AAA",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    opened = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))

    def fake_open(path: Path) -> bool:
        opened["path"] = path
        return True

    monkeypatch.setattr(runs_cli, "_try_open_path", fake_open)

    exit_code = root_cli.main(["runs", "open", "resume_me_00", "--dataset", "Gag"])

    assert exit_code == 0
    assert opened["path"] == run_dir.resolve()


def test_runs_open_accepts_run_id_selector(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "resume_me_00"
    run_id = "01ARZ3NDEKTSV4RRFFQ69G9AAB"
    _write_metadata(
        run_dir,
        run_id=run_id,
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    opened = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))

    def fake_open(path: Path) -> bool:
        opened["path"] = path
        return True

    monkeypatch.setattr(runs_cli, "_try_open_path", fake_open)

    exit_code = root_cli.main(["runs", "open", "--run-id", run_id])

    assert exit_code == 0
    assert opened["path"] == run_dir.resolve()


def test_runs_open_requires_disambiguation_when_selector_is_ambiguous(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    _write_metadata(
        datasets_root / "Actin" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G9AAC",
        dataset="Actin",
        model_subfolder="HDN",
    )
    _write_metadata(
        datasets_root / "Gag" / "runs" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G9AAD",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    called = {"value": False}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))

    def fake_open(_path: Path) -> bool:
        called["value"] = True
        return True

    monkeypatch.setattr(runs_cli, "_try_open_path", fake_open)

    exit_code = root_cli.main(["runs", "open", "duplicate_00"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert called["value"] is False
    assert "Multiple matching runs found:" in captured.out
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in captured.err


def test_runs_open_prints_run_folder_when_explorer_launch_fails(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G9AAZ",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_cli, "_try_open_path", lambda _path: False)

    exit_code = root_cli.main(["runs", "open", "Gag/Upsamp/resume_me_00"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip().splitlines()[-1] == str(run_dir.resolve())


def test_top_level_open_command_is_not_registered_anymore():
    with pytest.raises(SystemExit):
        root_cli.main(["open", "resume_me", "0"])


def test_runs_open_rejects_split_run_name_and_index_selector():
    with pytest.raises(SystemExit) as exc_info:
        root_cli.main(["runs", "open", "resume_me", "0"])
    assert exc_info.value.code == 2
