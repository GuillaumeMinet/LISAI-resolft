from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now
from lisai.runs.selection import resolve_discovered_run_selector


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
    last_heartbeat_offset_minutes: int = 0,
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    now = utc_now()
    last_heartbeat_at = now - timedelta(minutes=last_heartbeat_offset_minutes)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": "completed",
        "closed_cleanly": True,
        "created_at": now - timedelta(minutes=30),
        "updated_at": last_heartbeat_at,
        "ended_at": last_heartbeat_at,
        "last_heartbeat_at": last_heartbeat_at,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_resolve_discovered_run_selector_accepts_run_id(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "resume_me_00"
    run_id = "01ARZ3NDEKTSV4RRFFQ69GC100"
    _write_metadata(run_dir, run_id=run_id, dataset="Gag", model_subfolder="HDN")

    selected = resolve_discovered_run_selector(
        run_id=run_id,
        scan_result=scan_runs(datasets_root),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected is not None
    assert selected.run_dir == run_dir.resolve()


def test_resolve_discovered_run_selector_accepts_dataset_subfolder_run_dir(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp_base" / "SubA" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC101",
        dataset="Gag",
        model_subfolder="Upsamp_base/SubA",
    )

    selected = resolve_discovered_run_selector(
        selector="Gag/Upsamp_base/SubA/resume_me_00",
        scan_result=scan_runs(datasets_root),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected is not None
    assert selected.run_dir == run_dir.resolve()


def test_resolve_discovered_run_selector_accepts_bare_run_dir_with_dataset_filter(tmp_path):
    datasets_root = tmp_path / "datasets"
    selected_dir = datasets_root / "Gag" / "runs" / "HDN" / "duplicate_00"
    other_dir = datasets_root / "Actin" / "runs" / "HDN" / "duplicate_00"
    _write_metadata(
        selected_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC102",
        dataset="Gag",
        model_subfolder="HDN",
    )
    _write_metadata(
        other_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC103",
        dataset="Actin",
        model_subfolder="HDN",
    )

    selected = resolve_discovered_run_selector(
        selector="duplicate_00",
        dataset="Gag",
        scan_result=scan_runs(datasets_root),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected is not None
    assert selected.run_dir == selected_dir.resolve()


def test_resolve_discovered_run_selector_reports_bare_run_dir_ambiguity(tmp_path):
    datasets_root = tmp_path / "datasets"
    _write_metadata(
        datasets_root / "Gag" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69GC104",
        dataset="Gag",
        model_subfolder="HDN",
    )
    _write_metadata(
        datasets_root / "Actin" / "runs" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69GC105",
        dataset="Actin",
        model_subfolder="Upsamp",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    selected = resolve_discovered_run_selector(
        selector="duplicate_00",
        scan_result=scan_runs(datasets_root),
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert selected is None
    assert "Multiple matching runs found:" in stdout.getvalue()
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in stderr.getvalue()


def test_resolve_discovered_run_selector_uses_partial_exp_name_fallback(tmp_path):
    datasets_root = tmp_path / "datasets"
    first_dir = datasets_root / "Gag" / "runs" / "HDN" / "reducedUpsamp_beta_00"
    second_dir = datasets_root / "Actin" / "runs" / "Upsamp" / "reducedUpsamp_debug_00"
    _write_metadata(
        first_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC106",
        dataset="Gag",
        model_subfolder="HDN",
        last_heartbeat_offset_minutes=0,
    )
    _write_metadata(
        second_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC107",
        dataset="Actin",
        model_subfolder="Upsamp",
        last_heartbeat_offset_minutes=1,
    )

    selected = resolve_discovered_run_selector(
        selector="reduced",
        scan_result=scan_runs(datasets_root),
        stdin=InteractiveInput("02\n"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected is not None
    assert selected.run_dir == second_dir.resolve()


def test_resolve_discovered_run_selector_prefers_exact_bare_run_dir_over_partial_exp_name(tmp_path):
    datasets_root = tmp_path / "datasets"
    exact_dir = datasets_root / "Gag" / "runs" / "HDN" / "reduced"
    partial_dir = datasets_root / "Gag" / "runs" / "HDN" / "reducedUpsamp_beta_00"
    _write_metadata(
        exact_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC108",
        dataset="Gag",
        model_subfolder="HDN",
    )
    _write_metadata(
        partial_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC109",
        dataset="Gag",
        model_subfolder="HDN",
    )

    selected = resolve_discovered_run_selector(
        selector="reduced",
        scan_result=scan_runs(datasets_root),
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected is not None
    assert selected.run_dir == exact_dir.resolve()


def test_resolve_discovered_run_selector_rejects_selector_with_run_id(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_id = "01ARZ3NDEKTSV4RRFFQ69GC10A"
    _write_metadata(
        datasets_root / "Gag" / "runs" / "HDN" / "resume_me_00",
        run_id=run_id,
        dataset="Gag",
        model_subfolder="HDN",
    )
    stderr = io.StringIO()

    selected = resolve_discovered_run_selector(
        selector="resume_me_00",
        run_id=run_id,
        scan_result=scan_runs(datasets_root),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert selected is None
    assert "Use either a run selector or --run-id, not both." in stderr.getvalue()


def test_resolve_discovered_run_selector_accepts_partial_dataset_filter(tmp_path):
    datasets_root = tmp_path / "datasets"
    selected_dir = datasets_root / "actin_fixed_multi_snr" / "runs" / "HDN" / "duplicate_00"
    other_dir = datasets_root / "vimentin_live" / "runs" / "HDN" / "duplicate_00"
    _write_metadata(
        selected_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC10B",
        dataset="actin_fixed_multi_snr",
        model_subfolder="HDN",
    )
    _write_metadata(
        other_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GC10C",
        dataset="vimentin_live",
        model_subfolder="HDN",
    )

    selected = resolve_discovered_run_selector(
        selector="duplicate_00",
        dataset="actin_fixed",
        scan_result=scan_runs(datasets_root),
        stdin=InteractiveInput("y\n"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert selected is not None
    assert selected.run_dir == selected_dir.resolve()
