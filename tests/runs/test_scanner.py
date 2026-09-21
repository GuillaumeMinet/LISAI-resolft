from __future__ import annotations

import json

from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata


def _write_metadata(
    run_dir,
    *,
    dataset,
    model_subfolder,
    group_path,
    path,
    status="running",
    run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": status != "running",
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None if status == "running" else "2026-03-20T10:20:00Z",
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": path,
        "group_path": group_path,
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_scan_runs_discovers_nested_run_directories(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "runs" / "HDN" / "HDN_Gag_KL07_01"
    run_b = datasets_root / "Gag" / "runs" / "HDN" / "ablationA" / "HDN_Gag_KL07_02"
    run_c = datasets_root / "Gag" / "runs" / "Upsamp" / "2026_03" / "test1" / "Upsamp_Gag_CL5"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/runs/HDN/HDN_Gag_KL07_01",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="HDN/ablationA",
        group_path="ablationA",
        path="datasets/Gag/runs/HDN/ablationA/HDN_Gag_KL07_02",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
    )
    _write_metadata(
        run_c,
        dataset="Gag",
        model_subfolder="Upsamp/2026_03/test1",
        group_path="2026_03/test1",
        path="datasets/Gag/runs/Upsamp/2026_03/test1/Upsamp_Gag_CL5",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
    )

    results = scan_runs(datasets_root)
    by_id = {run.metadata.run_id: run for run in results.runs}

    assert len(results.runs) == 3
    assert results.invalid == ()
    assert by_id["01ARZ3NDEKTSV4RRFFQ69G5FAV"].model_subfolder == "HDN"
    assert by_id["01ARZ3NDEKTSV4RRFFQ69G5FAA"].group_path == "ablationA"
    assert by_id["01ARZ3NDEKTSV4RRFFQ69G5FAB"].model_subfolder == "Upsamp/2026_03/test1"
    assert all(run.path_consistent for run in results.runs)


def test_scan_runs_skips_invalid_metadata_files(tmp_path):
    datasets_root = tmp_path / "datasets"
    valid_run = datasets_root / "Gag" / "runs" / "HDN" / "valid_run"
    invalid_json_run = datasets_root / "Gag" / "runs" / "HDN" / "broken_json"
    invalid_schema_run = datasets_root / "Gag" / "runs" / "HDN" / "broken_schema"

    _write_metadata(
        valid_run,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/runs/HDN/valid_run",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FA0",
    )
    (invalid_json_run / ".lisai_run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (invalid_json_run / ".lisai_run_meta.json").write_text("{bad json", encoding="utf-8")
    (invalid_schema_run / ".lisai_run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (invalid_schema_run / ".lisai_run_meta.json").write_text(
        json.dumps({"schema_version": 2, "status": "wrong"}),
        encoding="utf-8",
    )

    results = scan_runs(datasets_root)
    kinds = {item.kind for item in results.invalid}

    assert len(results.runs) == 1
    assert "json_parse_error" in kinds
    assert "schema_validation_error" in kinds


def test_scan_runs_marks_path_mismatches_in_discovered_runs(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "wrong_path"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/not_the_real_run_dir",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FA1",
    )

    results = scan_runs(datasets_root)

    assert len(results.runs) == 1
    assert results.invalid == ()
    assert results.runs[0].path_consistent is False
    assert any(issue.startswith("path_mismatch:") for issue in results.runs[0].consistency_issues)


def test_scan_runs_accepts_run_id_not_matching_folder_name(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "folder_name_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/runs/HDN/folder_name_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FA2",
    )

    results = scan_runs(datasets_root)

    assert len(results.runs) == 1
    assert results.invalid == ()
    assert results.runs[0].metadata.run_id == "01ARZ3NDEKTSV4RRFFQ69G5FA2"
    assert results.runs[0].path_consistent is True


def test_scan_runs_ignores_runs_below_local_archive_folder(tmp_path):
    datasets_root = tmp_path / "datasets"
    active_run = datasets_root / "Gag" / "runs" / "HDN" / "beta_00"
    archived_run = (
        datasets_root
        / "Gag"
        / "runs"
        / "HDN"
        / "_archive"
        / "beta_01_archived_20260914-105230Z"
    )

    _write_metadata(
        active_run,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/runs/HDN/beta_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
    )
    _write_metadata(
        archived_run,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/runs/HDN/beta_01",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
    )

    results = scan_runs(datasets_root)

    assert [run.run_dir.name for run in results.runs] == ["beta_00"]
    assert results.invalid == ()


def test_default_run_scan_uses_only_configured_training_dataset_root(tmp_path, monkeypatch):
    import lisai.runs.scanner as scanner_mod

    datasets_root = tmp_path / "datasets"
    training_root = datasets_root / "train_bucket"
    evaluation_root = datasets_root / "eval_bucket"
    training_run = training_root / "Gag" / "runs" / "HDN" / "train_00"
    evaluation_run = evaluation_root / "EvalSet" / "runs" / "HDN" / "eval_00"

    _write_metadata(
        training_run,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/train_bucket/Gag/runs/HDN/train_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FA3",
    )
    _write_metadata(
        evaluation_run,
        dataset="EvalSet",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/eval_bucket/EvalSet/runs/HDN/eval_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FA4",
    )

    monkeypatch.setattr(scanner_mod._PATHS, "data_root", lambda: tmp_path)
    monkeypatch.setattr(scanner_mod._PATHS, "datasets_root", lambda: datasets_root)
    monkeypatch.setattr(scanner_mod._PATHS, "training_datasets_root", lambda: training_root)

    results = scanner_mod.scan_runs()

    assert [run.dataset for run in results.runs] == ["Gag"]
    assert results.runs[0].path == "datasets/train_bucket/Gag/runs/HDN/train_00"
    assert results.runs[0].path_consistent is True
