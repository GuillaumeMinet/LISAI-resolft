from __future__ import annotations

import argparse
import io
from datetime import timedelta
from pathlib import Path

import pytest

import lisai.evaluation.cli as evaluation_cli
import lisai.runs.selection as selection_mod
from lisai.cli import build_parser
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.runs.io import write_run_metadata_atomic
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


def test_apply_cli_parses_run_ref_config_and_overrides(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACA",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    def fake_run_apply_model(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", fake_run_apply_model)

    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "Gag/Upsamp/my_model_00",
            "/data/images",
            "--config",
            "fast_upsamp",
            "--tiling-size",
            "512",
            "--crop-size",
            "200",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["model_dataset"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "my_model_00"
    assert captured["data_path"] == Path("/data/images")
    assert captured["config"] == "fast_upsamp"
    assert captured["tiling_size"] == 512
    assert captured["crop_size"] == 200


def test_apply_cli_passes_limit_n_imgs(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACH",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(
        ["apply", "my_model_00", "/data/images", "--limit-n-imgs", "7"]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["limit_n_imgs"] == 7


def test_cli_parses_tiling_policy_values():
    assert evaluation_cli._parse_tiling_size("auto") == "auto"
    assert evaluation_cli._parse_tiling_size("off") == "off"
    assert evaluation_cli._parse_tiling_size("2000") == 2000


def test_apply_cli_accepts_no_tiling_alias(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACF",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(["apply", "my_model_00", "/data/images", "--no-tiling"])
    result = args.handler(args)

    assert result == 0
    assert captured["tiling_size"] == "off"


def test_apply_cli_accepts_best_or_last_both(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACB",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    def fake_run_apply_model(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", fake_run_apply_model)

    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "my_model_00",
            "/data/images",
            "--best-or-last",
            "both",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["best_or_last"] == "both"
    assert captured["model_name"] == "my_model_00"


def test_apply_cli_run_id_without_run_positional_parses_data_path():
    run_id = "01ARZ3NDEKTSV4RRFFQ69G7ACC"
    parser = build_parser()

    args = parser.parse_args(["apply", "--run-id", run_id, "/data/images"])

    assert args.run is None
    assert args.data_path == "/data/images"
    assert args.run_id == run_id


def test_apply_cli_accepts_run_id_selector_without_run_positional(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "HDN" / "resume_me_00"
    run_id = "01ARZ3NDEKTSV4RRFFQ69G7ACD"
    _write_metadata(
        run_dir,
        run_id=run_id,
        dataset="Gag",
        model_subfolder="HDN",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(["apply", "--run-id", run_id, "/data/images"])
    result = args.handler(args)

    assert result == 0
    assert captured["model_dataset"] == "Gag"
    assert captured["model_subfolder"] == "HDN"
    assert captured["model_name"] == "resume_me_00"
    assert captured["data_path"] == Path("/data/images")


def test_apply_cli_missing_run_selector_returns_nonzero(monkeypatch):
    called = {"value": False}
    stderr = io.StringIO()
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **_kwargs: called.update({"value": True}))
    monkeypatch.setattr(evaluation_cli.sys, "stderr", stderr)

    parser = build_parser()
    args = parser.parse_args(["apply", "/data/images"])
    result = args.handler(args)

    assert result == 1
    assert called["value"] is False
    assert "Missing run selector." in stderr.getvalue()


def test_evaluate_cli_parses_metrics_and_split(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ABA",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    def fake_run_evaluate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", fake_run_evaluate)

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "Gag/Upsamp/my_model_00",
            "--config",
            "benchmark",
            "--split",
            "val",
            "--metrics",
            "psnr,ssim",
            "--tiling-size",
            "auto",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "my_model_00"
    assert captured["config"] == "benchmark"
    assert captured["split"] == "val"
    assert captured["metrics_list"] == ["psnr", "ssim"]
    assert captured["tiling_size"] == "auto"


def test_evaluate_cli_accepts_best_or_last_both(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ABB",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    def fake_run_evaluate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", fake_run_evaluate)

    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "Gag/Upsamp/my_model_00",
            "--best-or-last",
            "both",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["best_or_last"] == "both"


def test_evaluate_cli_accepts_run_dir_selector(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "resume_me_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAA",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(["evaluate", "resume_me_00", "--config", "benchmark"])
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "resume_me_00"
    assert captured["config"] == "benchmark"


def test_evaluate_cli_ambiguous_selector_allows_interactive_line_selection(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    now = utc_now()
    _write_metadata(
        datasets_root / "Actin" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAB",
        dataset="Actin",
        model_subfolder="HDN",
        last_heartbeat_at=now,
    )
    _write_metadata(
        datasets_root / "Gag" / "runs" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAC",
        dataset="Gag",
        model_subfolder="Upsamp",
        last_heartbeat_at=now - timedelta(minutes=1),
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(evaluation_cli.sys, "stdin", InteractiveInput("02\n"))
    monkeypatch.setattr(evaluation_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluation_cli.sys, "stderr", stderr)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "duplicate_00"])
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "duplicate_00"
    assert "Multiple matching runs found:" in stdout.getvalue()
    assert "Select run number from '#'" in stdout.getvalue()


def test_evaluate_cli_ambiguous_selector_requires_extra_filters_when_non_interactive(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    _write_metadata(
        datasets_root / "Actin" / "runs" / "HDN" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAD",
        dataset="Actin",
        model_subfolder="HDN",
    )
    _write_metadata(
        datasets_root / "Gag" / "runs" / "Upsamp" / "duplicate_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAE",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    called = {"value": False}
    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: called.update({"value": True}))
    monkeypatch.setattr(evaluation_cli.sys, "stdin", NonInteractiveInput(""))
    monkeypatch.setattr(evaluation_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluation_cli.sys, "stderr", stderr)

    parser = build_parser()
    args = parser.parse_args(["evaluate", "duplicate_00"])
    result = args.handler(args)

    assert result == 1
    assert called["value"] is False
    assert "Multiple matching runs found:" in stdout.getvalue()
    assert "Rerun with --dataset/--subfolder or with --run-id to disambiguate." in stderr.getvalue()


def test_evaluate_cli_rejects_split_run_name_and_index_selector():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["evaluate", "resume_me", "0"])
    assert exc_info.value.code == 2


def test_apply_cli_accepts_promoted_model_without_run_selector(monkeypatch):
    captured = {}
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(["apply", "--model", "hdn-vimentin", "/data/images"])
    result = args.handler(args)

    assert result == 0
    assert args.run is None
    assert captured["promoted_model_name"] == "hdn-vimentin"
    assert captured["model_name"] == "hdn-vimentin"
    assert captured["model_subfolder"] == "promoted"
    assert captured["data_path"] == Path("/data/images")


def test_evaluate_cli_passes_independent_evaluation_dataset(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7AAF",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_evaluate", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(
        ["evaluate", "Gag/Upsamp/my_model_00", "--on", "gag_independent"]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["evaluation_dataset_name"] == "gag_independent"
    assert captured["split"] is evaluation_cli.UNSET


def test_evaluate_cli_rejects_on_with_split():
    parser = build_parser()
    args = parser.parse_args(
        ["evaluate", "some_run", "--on", "gag_independent", "--split", "test"]
    )

    with pytest.raises(SystemExit) as exc_info:
        args.handler(args)

    assert exc_info.value.code == 2


def test_apply_cli_accepts_output_mode(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACG",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(
        ["apply", "my_model_00", "/data/images", "--output-mode", "folder_inside"]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["output_mode"] == "folder_inside"


def test_apply_cli_output_arguments_are_mutually_exclusive():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "apply",
                "run_00",
                "/data/images",
                "--save-folder",
                "/tmp/predictions",
                "--in-place",
            ]
        )


def test_apply_cli_output_mode_is_mutually_exclusive_with_legacy_output_flags():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "apply",
                "run_00",
                "/data/images",
                "--output-mode",
                "folder_outside",
                "--in-place",
            ]
        )


def test_apply_cli_save_input_override(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACG",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(
        ["apply", "my_model_00", "/data/images", "--no-save-input"]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["save_input"] is False


def test_apply_help_has_output_location_group():
    parser = build_parser()
    apply_parser = next(
        action.choices["apply"]
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    help_text = apply_parser.format_help()

    assert "Model selection" in help_text
    assert "Configuration" in help_text
    assert "Checkpoint" in help_text
    assert "Input selection" in help_text
    assert "Inference" in help_text
    assert "Output location" in help_text
    assert "Output contents" in help_text
    assert "--save-folder" in help_text
    assert "--output-mode" in help_text
    assert "--in-place" in help_text
    assert "--no-in-place" in help_text
    assert "--save-input" in help_text
    assert "--no-save-input" in help_text
    assert "--limit-n-imgs" in help_text
    assert "Advanced and model-specific inference settings" in help_text


def test_evaluate_help_uses_workflow_groups():
    parser = build_parser()
    evaluate_parser = next(
        action.choices["evaluate"]
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    help_text = evaluate_parser.format_help()

    assert "Model selection" in help_text
    assert "Configuration" in help_text
    assert "Checkpoint" in help_text
    assert "Evaluation data" in help_text
    assert "Inference" in help_text
    assert "Metrics" in help_text
    assert "Output" in help_text
    assert "Advanced and model-specific inference settings" in help_text


@pytest.mark.parametrize(
    "technical_option",
    [
        "--downsamp",
        "--fill-factor",
        "--stack-selection-idx",
        "--dark-frame-context-length",
        "--denormalize-output",
        "--color-code-option",
        "--keep-original-shape",
    ],
)
def test_apply_help_hides_config_only_options(technical_option):
    parser = build_parser()
    apply_parser = next(
        action.choices["apply"]
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert technical_option not in apply_parser.format_help()


@pytest.mark.parametrize("technical_option", ["--ch-out", "--data-option"])
def test_evaluate_help_hides_config_only_options(technical_option):
    parser = build_parser()
    evaluate_parser = next(
        action.choices["evaluate"]
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert technical_option not in evaluate_parser.format_help()
