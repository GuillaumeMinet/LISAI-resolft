from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import re

from lisai.infra.fs.run_naming import parse_run_dir_name
import lisai.runs.cli as runs_cli
import lisai.runs.listing as runs_listing
from lisai.cli import main as root_main
from lisai.config import settings
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata


def _write_metadata(
    run_dir, *, dataset, model_subfolder, group_path, path, status="running",
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


def _table_header(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("dataset  "):
            return line
    raise AssertionError("Could not find table header line in output.")


def _header_columns(output: str) -> list[str]:
    header = _table_header(output)
    return [part for part in re.split(r"\s{2,}", header.strip()) if part]


def test_runs_list_uses_filters_and_warns_on_invalid_files(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"
    run_b = datasets_root / "Actin" / "models" / "Upsamp" / "run_b"
    invalid_run = datasets_root / "Gag" / "models" / "HDN" / "broken"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Actin",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Actin/models/Upsamp/run_b",
        status="completed",
    )
    (invalid_run / ".lisai_run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (invalid_run / ".lisai_run_meta.json").write_text("{bad json", encoding="utf-8")

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--status", "running"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "LISAI runs listing - Dataset: 'Gag' | Status: 'running'" in captured.out
    assert "dataset" in captured.out
    assert "eta_left" in captured.out
    assert "path_consistent" not in captured.out
    assert "closed_cleanly" not in captured.out
    assert "start_time" not in captured.out
    assert "last_seen" not in captured.out
    assert "run_id" not in captured.out
    assert "run_a" in captured.out
    assert "4/10" in captured.out
    assert "run_b" not in captured.out
    assert "warning: skipped invalid run metadata" in captured.err


def test_runs_list_model_subfolder_filter_works(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"
    run_b = datasets_root / "Gag" / "models" / "Upsamp" / "run_b"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Gag/models/Upsamp/run_b",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--model-subfolder", "Upsamp"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_b" in captured.out
    assert "run_a" not in captured.out


def test_runs_list_subfolder_alias_works(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"
    run_b = datasets_root / "Gag" / "models" / "Upsamp" / "run_b"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Gag/models/Upsamp/run_b",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--subfolder", "HDN"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_a" in captured.out
    assert "run_b" not in captured.out


def test_runs_list_run_dir_filter_works(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"
    run_b = datasets_root / "Gag" / "models" / "HDN" / "run_b_00"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a_00",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_b_00",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--run_dir", "run_b_00"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_dir='run_b_00'" in captured.out
    assert "run_b_00" in captured.out
    assert "run_a_00" not in captured.out


def test_runs_list_exp_name_partial_filter_works(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    reduced_a = datasets_root / "Gag" / "models" / "HDN" / "reducedUpsamp_Upsamp2_beta_00"
    reduced_b = datasets_root / "Actin" / "models" / "Upsamp" / "my_REDUCED_debug_01"
    other = datasets_root / "Gag" / "models" / "HDN" / "fullDataset_Upsamp2_00"

    _write_metadata(
        reduced_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/reducedUpsamp_Upsamp2_beta_00",
        status="running",
    )
    _write_metadata(
        reduced_b,
        dataset="Actin",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Actin/models/Upsamp/my_REDUCED_debug_01",
        status="completed",
    )
    _write_metadata(
        other,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/fullDataset_Upsamp2_00",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--exp-name", "reduced"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "exp_name~='reduced'" in captured.out
    assert "reducedUpsamp_Upsamp2_beta_00" in captured.out
    assert "my_REDUCED_debug_01" in captured.out
    assert "fullDataset_Upsamp2_00" not in captured.out


def test_runs_list_exp_name_partial_filter_combines_with_other_filters(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "reducedUpsamp_Upsamp2_beta_00"
    run_b = datasets_root / "Actin" / "models" / "HDN" / "reducedUpsamp_Upsamp2_beta_01"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/reducedUpsamp_Upsamp2_beta_00",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Actin",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Actin/models/HDN/reducedUpsamp_Upsamp2_beta_01",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--exp-name", "upsamp2", "--dataset", "Gag"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Dataset: 'Gag'" in captured.out
    assert "exp_name~='upsamp2'" in captured.out
    assert "reducedUpsamp_Upsamp2_beta_00" in captured.out
    assert "reducedUpsamp_Upsamp2_beta_01" not in captured.out


def test_runs_list_uses_local_timestamp_formatter(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_listing, "format_timestamp_local", lambda _value: "LOCAL_TS")

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "LOCAL_TS" in captured.out


def test_runs_list_marks_old_running_heartbeats_as_stale(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_listing, "utc_now", lambda: datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(settings.project_cfg.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--status", "running"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_a" in captured.out
    assert "stale (x4)" in captured.out


def test_runs_list_footer_notes_path_inconsistency_without_stderr_warning(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/not_the_real_folder",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--dataset", "Gag"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "path_consistent" not in captured.out
    assert "inconsistent path metadata" in captured.out
    assert "path_mismatch" not in captured.err


def test_runs_list_full_appends_extended_columns(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/not_the_real_folder",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "LISAI runs listing - Dataset: 'Gag'" in captured.out
    columns = _header_columns(captured.out)
    assert columns[:6] == ["dataset", "model_subfolder", "run_dir", "status", "epoch", "eta_left"]
    assert columns[-6:] == [
        "failure",
        "path_consistent",
        "closed_cleanly",
        "start_time",
        "last_seen",
        "run_id",
    ]
    assert "false" in captured.out
    assert "01ARZ3NDEKTSV4RRFFQ69G5FAV" in captured.out


def test_runs_list_live_renders_in_place_when_interactive(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a_00",
        status="running",
    )

    class _InteractiveBuffer(StringIO):
        def isatty(self):
            return True

    out = _InteractiveBuffer()
    err = StringIO()
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    sleep_values: list[float] = []

    def _interrupt_sleep(seconds):
        sleep_values.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(runs_cli.time, "sleep", _interrupt_sleep)

    exit_code = runs_cli.list_runs(dataset="Gag", live=True, interval_seconds=0.1, stdout=out, stderr=err)

    assert exit_code == 0
    assert sleep_values == [1.0]
    assert "\x1b[H\x1b[J" in out.getvalue()
    assert "warning: --interval 0.1s is below the minimum 1s; using 1s." in out.getvalue()
    assert "LISAI runs listing - Dataset: 'Gag'" in out.getvalue()
    assert "LIVE MODE (1s refresh) - Ctrl+C to stop live" in out.getvalue()
    assert "run_a_00" in out.getvalue()
    assert err.getvalue() == ""


def test_runs_list_live_falls_back_to_single_snapshot_without_tty(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a_00",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--live", "--interval", "0.25"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines()[0] == "warning: --interval 0.25s is below the minimum 1s; using 1s."
    assert captured.out.splitlines()[1] == "LISAI runs listing - Dataset: 'Gag'"
    assert "run_a_00" in captured.out
    assert "--live requires interactive terminal output" in captured.err


def test_runs_list_clamps_zero_interval_to_one_second(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a_00",
        status="running",
    )

    class _InteractiveBuffer(StringIO):
        def isatty(self):
            return True

    out = _InteractiveBuffer()
    err = StringIO()
    sleep_values: list[float] = []
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    def _interrupt_sleep(seconds):
        sleep_values.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(runs_cli.time, "sleep", _interrupt_sleep)

    exit_code = runs_cli.list_runs(dataset="Gag", live=True, interval_seconds=0.0, stdout=out, stderr=err)

    assert exit_code == 0
    assert sleep_values == [1.0]
    assert "warning: --interval 0s is below the minimum 1s; using 1s." in out.getvalue()
    assert "LISAI runs listing - Dataset: 'Gag'" in out.getvalue()
    assert "LIVE MODE (1s refresh) - Ctrl+C to stop live" in out.getvalue()
    assert err.getvalue() == ""


def test_runs_list_title_without_filters(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a_00",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines()[0] == "LISAI runs listing"


def test_runs_list_namespace_remains_available_with_top_level_list(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_namespace_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_namespace_00",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--status", "running"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "LISAI runs listing - Status: 'running'" in captured.out
    assert "run_namespace" in captured.out


def test_runs_list_promoted_filters_by_source_run_id(monkeypatch, tmp_path, capsys):
    import lisai.promoted_models.registry as promoted_registry

    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"
    run_b = datasets_root / "Gag" / "models" / "HDN" / "run_b_00"
    promoted_id = "01ARZ3NDEKTSV4RRFFQ69G5FAA"
    other_id = "01ARZ3NDEKTSV4RRFFQ69G5FAB"
    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a_00",
        status="completed",
        run_id=promoted_id,
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_b_00",
        status="completed",
        run_id=other_id,
    )
    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(promoted_registry, "promoted_source_run_ids", lambda: {promoted_id})

    exit_code = root_main(["runs", "list", "--promoted"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Promoted only" in captured.out
    assert "run_a_00" in captured.out
    assert "run_b_00" not in captured.out
