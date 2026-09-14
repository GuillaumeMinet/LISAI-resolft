from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import lisai.cli as root_cli
import lisai.runs.cli as runs_cli
import lisai.runs.selection as selection_mod
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def _write_metadata(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    model_subfolder: str,
    architecture: str | None = None,
):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    now = utc_now()
    created_at = now - timedelta(minutes=5)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": "completed",
        "closed_cleanly": True,
        "created_at": created_at,
        "updated_at": now,
        "ended_at": now,
        "last_heartbeat_at": now,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    if architecture is not None:
        payload["training_signature"] = {"architecture": architecture}
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_runs_plot_delegates_to_shared_plotting(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB100",
        dataset="Gag",
        model_subfolder="Upsamp",
        architecture="upsamp",
    )
    (run_dir / "loss.txt").write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.2 1.5\n",
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))

    def _fake_show(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runs_cli, "show_loss_plot_for_run", _fake_show)

    exit_code = root_cli.main(["runs", "plot", "resume_me_00", "--dataset", "Gag"])

    assert exit_code == 0
    assert captured["run_dir"] == run_dir.resolve()
    assert captured["dataset"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["architecture"] == "upsamp"
    assert captured["open_saved_plot"] is runs_cli._try_open_path


def test_runs_plot_accepts_dataset_subfolder_runref_selector(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp_base" / "SubA" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB105",
        dataset="Gag",
        model_subfolder="Upsamp_base/SubA",
        architecture="upsamp",
    )
    (run_dir / "loss.txt").write_text(
        "Epoch Train_loss Val_loss\n"
        "0 1.2 1.5\n",
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(
        runs_cli,
        "show_loss_plot_for_run",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    exit_code = root_cli.main(["runs", "plot", "Gag/Upsamp_base/SubA/resume_me_00"])

    assert exit_code == 0
    assert captured["run_dir"] == run_dir.resolve()
    assert captured["model_subfolder"] == "Upsamp_base/SubA"


def test_runs_plot_returns_nonzero_when_plotting_fails(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69GB109",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_cli, "show_loss_plot_for_run", lambda **_kwargs: 1)

    exit_code = root_cli.main(["runs", "plot", "resume_me_00", "--dataset", "Gag"])

    assert exit_code == 1


def test_runs_plot_ambiguous_selector_requires_disambiguation_when_non_interactive(
    monkeypatch,
    tmp_path,
    capsys,
):
    datasets_root = tmp_path / "datasets"
    _write_metadata(
        datasets_root / "Actin" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69GB102",
        dataset="Actin",
        model_subfolder="HDN",
    )
    _write_metadata(
        datasets_root / "Gag" / "runs" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69GB103",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    called = {"value": False}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_cli.sys, "stdin", NonInteractiveInput(""))

    def _fake_show(**_kwargs):
        called["value"] = True
        return 0

    monkeypatch.setattr(runs_cli, "show_loss_plot_for_run", _fake_show)

    exit_code = root_cli.main(["runs", "plot", "duplicate_00"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert called["value"] is False
    assert "Multiple matching runs found:" in captured.out
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in captured.err


def test_runs_plot_rejects_split_run_name_and_index_selector():
    try:
        root_cli.main(["runs", "plot", "resume_me", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected argparse to reject split run_name/run_index selector.")
