from __future__ import annotations

import argparse
import io
from datetime import timedelta
from pathlib import Path

import pytest

import lisai.evaluation.cli as evaluation_cli
import lisai.runs.selection as selection_mod
from lisai.cli import build_parser
from lisai.config.models.inference import ApplyDefaults, EvaluateDefaults
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

    def fake_resolve_apply_config(*, model_config, config, overrides):
        captured["model_config"] = model_config
        captured["config"] = config
        captured["overrides"] = overrides
        return ApplyDefaults()

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "resolve_apply_config", fake_resolve_apply_config)
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
            "--no-progress-bar",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["model_dataset"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "my_model_00"
    assert captured["data_path"] == Path("/data/images")
    assert captured["model_config"] is None
    assert captured["config"] == "fast_upsamp"
    assert captured["overrides"].inference.tiling_size == 512
    assert captured["overrides"].inference.crop_size == 200
    assert captured["progress_bar"] is False


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
    assert captured["cfg"].input.limit_n_imgs == 7


def test_cli_parses_tiling_policy_values():
    assert evaluation_cli._parse_tiling_size("auto") == "auto"
    assert evaluation_cli._parse_tiling_size("off") == "off"
    assert evaluation_cli._parse_tiling_size("2000") == 2000


def test_build_apply_overrides_is_sparse():
    parser = build_parser()
    args = parser.parse_args(["apply", "run_00", "/data/images"])

    overrides = evaluation_cli._build_apply_overrides(args)

    assert overrides.model_dump(exclude_unset=True) == {}


def test_build_apply_overrides_translates_cli_aliases_to_nested_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "run_00",
            "/data/images",
            "--no-save-input",
            "--apply-color-code",
            "--output-mode",
            "folder_inside",
        ]
    )

    overrides = evaluation_cli._build_apply_overrides(args)

    assert overrides.saving.save_input_mode == "never"
    assert overrides.saving.mode == "folder_inside"
    assert overrides.postprocess.color_code.enabled is True


def test_build_evaluate_overrides_is_sparse():
    parser = build_parser()
    args = parser.parse_args(["evaluate", "run_00"])

    overrides = evaluation_cli._build_evaluate_overrides(args)

    assert overrides.model_dump(exclude_unset=True) == {}


def test_build_evaluate_overrides_maps_cli_fields_to_nested_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "run_00",
            "--split",
            "val",
            "--metrics",
            "psnr,ssim",
            "--tiling-size",
            "512",
            "--save-folder",
            "/tmp/eval",
            "--overwrite",
        ]
    )

    overrides = evaluation_cli._build_evaluate_overrides(args)

    assert overrides.data.split == "val"
    assert overrides.metrics == ["psnr", "ssim"]
    assert overrides.inference.tiling_size == 512
    assert overrides.saving.save_folder == "/tmp/eval"
    assert overrides.saving.overwrite is True


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
    assert captured["cfg"].inference.tiling_size == "off"


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
    assert captured["cfg"].checkpoint.best_or_last == "both"
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

    def fake_resolve_evaluate_config(*, config, overrides):
        captured["config"] = config
        captured["overrides"] = overrides
        return EvaluateDefaults()

    def fake_run_evaluate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "resolve_evaluate_config", fake_resolve_evaluate_config)
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
            "--progress-bar",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["dataset_name"] == "Gag"
    assert captured["model_subfolder"] == "Upsamp"
    assert captured["model_name"] == "my_model_00"
    assert captured["config"] == "benchmark"
    assert captured["overrides"].data.split == "val"
    assert captured["overrides"].metrics == ["psnr", "ssim"]
    assert captured["overrides"].inference.tiling_size == "auto"
    assert isinstance(captured["cfg"], EvaluateDefaults)
    assert captured["progress_bar"] is True


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
    assert captured["cfg"].checkpoint.best_or_last == "both"


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

    def fake_resolve_evaluate_config(*, config, overrides):
        captured["config"] = config
        captured["overrides"] = overrides
        return EvaluateDefaults()

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "resolve_evaluate_config", fake_resolve_evaluate_config)
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
    from types import SimpleNamespace

    captured = {}
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        evaluation_cli,
        "load_promoted_model_registry",
        lambda: SimpleNamespace(models={"hdn-vimentin": object()}),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "load_promoted_model",
        lambda name: SimpleNamespace(inference_config_path=None),
    )

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
    assert captured["cfg"].data.split == "test"


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
    assert captured["cfg"].saving.mode == "folder_inside"


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


def test_apply_cli_passes_save_folder_and_overwrite(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACJ",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "my_model_00",
            "/data/images",
            "--save-folder",
            "/tmp/predictions",
            "--overwrite",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["cfg"].saving.save_folder == "/tmp/predictions"
    assert captured["overwrite"] is True


def test_apply_cli_passes_reuse_folder_and_skip_existing(monkeypatch, tmp_path):
    captured = {}
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "runs" / "Upsamp" / "my_model_00"
    _write_metadata(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G7ACK",
        dataset="Gag",
        model_subfolder="Upsamp",
    )

    monkeypatch.setattr(selection_mod, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))

    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "my_model_00",
            "/data/images",
            "--save-folder",
            "/tmp/predictions",
            "--reuse-folder",
            "--skip-existing",
        ]
    )
    result = args.handler(args)

    assert result == 0
    assert captured["cfg"].saving.save_folder == "/tmp/predictions"
    assert captured["reuse_folder"] is True
    assert captured["skip_existing"] is True


def test_apply_cli_rejects_overwrite_with_in_place_output():
    parser = build_parser()
    args = parser.parse_args(
        ["apply", "run_00", "/data/images", "--in-place", "--overwrite"]
    )

    with pytest.raises(SystemExit) as exc_info:
        args.handler(args)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "flags",
    [
        ["--overwrite", "--reuse-folder"],
        ["--overwrite", "--skip-existing"],
        ["--in-place", "--reuse-folder"],
        ["--in-place", "--skip-existing"],
    ],
)
def test_apply_cli_rejects_conflicting_reuse_flags(flags):
    parser = build_parser()
    args = parser.parse_args(["apply", "run_00", "/data/images", *flags])

    with pytest.raises(SystemExit) as exc_info:
        args.handler(args)

    assert exc_info.value.code == 2


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
    assert captured["cfg"].saving.save_input_mode == "never"


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
    assert "--overwrite" in help_text
    assert "--reuse-folder" in help_text
    assert "--skip-existing" in help_text
    assert "--output-mode" in help_text
    assert "--in-place" in help_text
    assert "--no-in-place" in help_text
    assert "--save-input" in help_text
    assert "--no-save-input" in help_text
    assert "--limit-n-imgs" in help_text
    assert "For more advanced or model-specific inference settings" in help_text


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
    assert "For more advanced or model-specific inference settings" in help_text


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


def test_apply_cli_resolves_confirmed_partial_promoted_model(monkeypatch):
    from types import SimpleNamespace

    captured = {}
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        evaluation_cli,
        "load_promoted_model_registry",
        lambda: SimpleNamespace(
            models={"hdn-vimentin-5frames": object(), "rcan-actin": object()}
        ),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "load_promoted_model",
        lambda name: SimpleNamespace(inference_config_path=None),
    )
    monkeypatch.setattr(evaluation_cli.sys, "stdin", InteractiveInput("y\n"))
    monkeypatch.setattr(evaluation_cli.sys, "stdout", stdout)
    monkeypatch.setattr(evaluation_cli.sys, "stderr", stderr)

    parser = build_parser()
    args = parser.parse_args(["apply", "--model", "hdn-vim", "/data/images"])
    result = args.handler(args)

    assert result == 0
    assert captured["promoted_model_name"] == "hdn-vimentin-5frames"
    assert captured["model_name"] == "hdn-vimentin-5frames"
    assert "Did you mean 'hdn-vimentin-5frames'? [y/N]" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_apply_cli_layers_promoted_model_inference_config(monkeypatch, tmp_path):
    from types import SimpleNamespace

    captured = {}
    model_config = tmp_path / "config_inference.yaml"
    model_config.write_text(
        "apply:\n  inference:\n    lvae_num_samples: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_cli, "run_apply_model", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        evaluation_cli,
        "load_promoted_model_registry",
        lambda: SimpleNamespace(models={"hdn-vimentin": object()}),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "load_promoted_model",
        lambda name: SimpleNamespace(inference_config_path=model_config),
    )

    parser = build_parser()
    args = parser.parse_args(["apply", "--model", "hdn-vimentin", "/data/images"])
    result = args.handler(args)

    assert result == 0
    assert captured["cfg"].inference.lvae_num_samples == 10
