from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import pytest

import lisai.cli as root_cli
import lisai.runs.selection as selection_mod
import lisai.training.continue_cli as continue_cli
from lisai.config import settings
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def _write_metadata(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    model_subfolder: str,
    status: str = "running",
    closed_cleanly: bool | None = None,
    last_heartbeat_at=None,
    ended_at=None,
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    now = utc_now()
    heartbeat = last_heartbeat_at if last_heartbeat_at is not None else now
    created_at = heartbeat - timedelta(minutes=5)
    updated_at = heartbeat

    if closed_cleanly is None:
        closed_cleanly = status != "running"
    if ended_at is None and status != "running":
        ended_at = heartbeat

    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": closed_cleanly,
        "created_at": created_at,
        "updated_at": updated_at,
        "ended_at": ended_at,
        "last_heartbeat_at": heartbeat,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/runs/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_root_cli_continue_dispatches_unique_match_and_builds_continue_config(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag_timelapses" / "runs" / "Upsamp" / "CL1_Upsamp2_Mltpl05_lightweight_01"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
        dataset="Gag_timelapses",
        model_subfolder="Upsamp",
        status="completed",
    )

    captured = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))

    exit_code = root_cli.main([
        "continue",
        "CL1_Upsamp2_Mltpl05_lightweight_01",
        "--dataset",
        "Gag_timelapses",
        "--yes",
    ])

    assert exit_code == 0
    assert captured["cfg"] == {
        "experiment": {"mode": "continue_training"},
        "load_model": {
            "canonical_load": False,
            "model_full_path": str(run_dir.resolve()),
            "load_method": "state_dict",
            "best_or_last": "last",
        },
    }


def test_root_cli_continue_passes_progress_bar_override(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FBB",
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )

    captured = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(
        continue_cli,
        "run_training_from_config_dict",
        lambda cfg, **kwargs: captured.update({"cfg": cfg, "kwargs": kwargs}),
    )

    exit_code = root_cli.main([
        "continue",
        "resume_me_00",
        "--dataset",
        "Gag",
        "--yes",
        "--no-progress-bar",
    ])

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["model_full_path"] == str(run_dir.resolve())
    assert captured["kwargs"] == {"progress_bar": False}


def test_continue_reports_multiple_matches_and_requests_disambiguation(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "duplicate_00"
    _write_metadata(
        datasets_root / "Gag" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )
    _write_metadata(
        datasets_root / "Actin" / "runs" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
        dataset="Actin",
        model_subfolder="Upsamp",
        status="completed",
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = continue_cli.continue_run(
        selector=selector,
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert "Multiple matching runs found" in stdout.getvalue()
    assert selector in stdout.getvalue()
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in stderr.getvalue()


def test_continue_ambiguous_matches_allow_interactive_line_selection(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "duplicate_00"
    now = utc_now()
    _write_metadata(
        datasets_root / "Actin" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAF",
        dataset="Actin",
        model_subfolder="HDN",
        status="completed",
        last_heartbeat_at=now,
    )
    selected_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "duplicate_00"
    _write_metadata(
        selected_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAG",
        dataset="Gag",
        model_subfolder="Upsamp",
        status="completed",
        last_heartbeat_at=now - timedelta(minutes=1),
    )

    captured = {}
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))

    exit_code = continue_cli.continue_run(
        selector=selector,
        assume_yes=True,
        stdin=InteractiveInput("02\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["model_full_path"] == str(selected_dir.resolve())
    assert "Multiple matching runs found:" in stdout.getvalue()
    assert "Select run number from '#'" in stdout.getvalue()


def test_continue_allows_run_id_override(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "resume_me_00"
    run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAD"
    _write_metadata(
        run_dir,
        run_id=run_id,
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )

    captured = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))

    exit_code = continue_cli.continue_run(
        run_id=run_id,
        assume_yes=True,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["model_full_path"] == str(run_dir.resolve())


def test_continue_requires_yes_when_confirmation_is_non_interactive(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "resume_me_00"
    _write_metadata(
        datasets_root / "Gag" / "runs" / "HDN" / "resume_me_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAE",
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )

    captured = {"called": False}
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))

    exit_code = continue_cli.continue_run(
        selector=selector,
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert captured["called"] is False
    assert "Confirmation required. Rerun with --yes to continue non-interactively." in stderr.getvalue()


def test_continue_blocks_recently_active_runs_without_force(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "still_running_00"
    now = utc_now()
    _write_metadata(
        datasets_root / "Gag" / "runs" / "HDN" / "still_running_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAF",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at=now - timedelta(minutes=2),
        ended_at=None,
    )

    captured = {"called": False}
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = continue_cli.continue_run(
        selector=selector,
        assume_yes=True,
        force=False,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
        now=now,
    )

    assert exit_code == 1
    assert captured["called"] is False
    assert "Rerun with --force" in stderr.getvalue()


def test_continue_force_allows_recently_active_runs(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "force_resume_00"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "force_resume_00"
    now = utc_now()
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at=now - timedelta(minutes=2),
        ended_at=None,
    )

    captured = {}
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = continue_cli.continue_run(
        selector=selector,
        assume_yes=True,
        force=True,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
        now=now,
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["model_full_path"] == str(run_dir.resolve())
    assert "warning: forcing continuation" in stderr.getvalue()


def test_continue_requires_force_for_non_interactive_path_inconsistency(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "renamed_00"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "renamed_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB1",
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )

    # Force a path inconsistency by editing stored metadata path.
    metadata = read_run_metadata(run_dir)
    broken = metadata.model_copy(update={"path": "datasets/Gag/models/HDN/some_other_name"})
    write_run_metadata_atomic(run_dir, broken)

    captured = {"called": False}
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))

    exit_code = continue_cli.continue_run(
        selector=selector,
        assume_yes=True,
        force=False,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert captured["called"] is False
    assert "inconsistent path metadata" in stderr.getvalue()


def test_continue_allows_non_interactive_path_inconsistency_with_yes_and_force(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "renamed_00"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "renamed_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB3",
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )

    metadata = read_run_metadata(run_dir)
    broken = metadata.model_copy(update={"path": "datasets/Gag/models/HDN/some_other_name"})
    write_run_metadata_atomic(run_dir, broken)

    captured = {"called": False}
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))

    exit_code = continue_cli.continue_run(
        selector=selector,
        assume_yes=True,
        force=True,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 0
    assert captured["called"] is True
    assert "continuing with --yes --force" in stderr.getvalue()


def test_continue_allows_stale_running_runs_after_confirmation(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "stale_run_00"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "stale_run_00"
    now = utc_now()
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB2",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at=now - timedelta(minutes=30),
        ended_at=None,
    )

    captured = {}
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = continue_cli.continue_run(
        selector=selector,
        stdin=InteractiveInput("y\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        now=now,
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["model_full_path"] == str(run_dir.resolve())
    assert "appears stale" in stderr.getvalue()


def test_continue_failed_run_uses_generic_confirmation_prompt(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "failed_once_00"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "failed_once_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB4",
        dataset="Gag",
        model_subfolder="HDN",
        status="failed",
    )

    captured = {}
    stdout = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))

    exit_code = continue_cli.continue_run(
        selector=selector,
        stdin=InteractiveInput("y\n"),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["model_full_path"] == str(run_dir.resolve())
    assert "Continue training this run in place?" in stdout.getvalue()


def test_continue_failed_run_non_interactive_requires_yes(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    selector = "failed_once_00"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "failed_once_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB5",
        dataset="Gag",
        model_subfolder="HDN",
        status="failed",
    )

    captured = {"called": False}
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))

    exit_code = continue_cli.continue_run(
        selector=selector,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert exit_code == 1
    assert captured["called"] is False
    assert "Confirmation required. Rerun with --yes to continue non-interactively." in stderr.getvalue()


def test_continue_rejects_split_run_name_and_index_selector():
    with pytest.raises(SystemExit) as exc_info:
        root_cli.main(["continue", "resume_me", "0"])
    assert exc_info.value.code == 2
