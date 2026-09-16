from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lisai.config.models.inference import ApplyDefaults
from lisai.infra.fs import OutputFolderResolution

apply_mod = importlib.import_module("lisai.evaluation.run_apply_model")


def test_run_apply_model_has_typed_config_runtime_boundary():
    signature = inspect.signature(apply_mod.run_apply_model)

    assert list(signature.parameters)[0] == "cfg"
    assert signature.parameters["cfg"].annotation is ApplyDefaults
    flat_runtime_fields = {
        "tiling_size",
        "crop_size",
        "lvae_num_samples",
        "downsamp",
        "fill_factor",
        "filters",
        "limit_n_imgs",
        "save_input",
        "output_mode",
    }
    assert flat_runtime_fields.isdisjoint(signature.parameters)


def _base_apply_config(
    *,
    downsamp: int | None = 2,
    fill_factor: float | None = None,
    limit_n_imgs: int | None = None,
) -> ApplyDefaults:
    return ApplyDefaults.model_validate(
        {
            "checkpoint": {"epoch_number": None, "best_or_last": "best"},
            "input": {
                "filters": ["tif", "tiff"],
                "skip_if_contain": None,
                "stack_selection_idx": None,
                "limit_n_imgs": limit_n_imgs,
                "timelapse_max": None,
            },
            "inference": {
                "crop_size": None,
                "keep_original_shape": True,
                "tiling_size": 64,
                "lvae_num_samples": 20,
                "downsamp": downsamp,
                "fill_factor": fill_factor,
                "dark_frame_context_length": False,
            },
            "postprocess": {
                "denormalize": False,
                "color_code": {"enabled": False},
            },
            "saving": {
                "lvae_save_samples": True,
                "mode": "default",
                "save_input_mode": "never",
            },
        }
    )


def _output_folder_resolution(
    path: Path,
    *,
    if_exists_policy: str = "numbered",
) -> OutputFolderResolution:
    path = Path(path)
    requested_existed = path.exists()
    if requested_existed and if_exists_policy == "reuse":
        action = "reused"
    elif requested_existed and if_exists_policy == "overwrite":
        action = "overwritten"
    else:
        action = "created"

    return OutputFolderResolution(
        requested=path,
        path=path,
        action=action,
        requested_existed=requested_existed,
    )


def _patch_prepare_output_folder(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict,
) -> None:
    def _fake_prepare_output_folder(path, *, if_exists_policy, parent_policy="create"):
        captured["save_folder"] = Path(path)
        captured["if_exists_policy"] = if_exists_policy
        captured["parent_policy"] = parent_policy
        return _output_folder_resolution(Path(path), if_exists_policy=if_exists_policy)

    monkeypatch.setattr(apply_mod, "prepare_output_folder", _fake_prepare_output_folder)


def _patch_common_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    input_image: np.ndarray | None = None,
) -> None:
    if input_image is None:
        input_image = np.ones((8, 8), dtype=np.float32)

    monkeypatch.setattr(apply_mod, "resolve_run_dir", lambda **_: tmp_path / "run")
    monkeypatch.setattr(
        apply_mod,
        "load_saved_run",
        lambda _: SimpleNamespace(
            is_lvae=False,
            data_norm_prm=None,
            model_norm_prm=None,
            upsampling_factor=1,
            context_length=None,
        ),
    )
    monkeypatch.setattr(
        apply_mod,
        "initialize_runtime",
        lambda **_: SimpleNamespace(model=object(), device="cpu", tiling_size=32),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (tmp_path, ["input.tif"], None),
    )
    monkeypatch.setattr(apply_mod, "imread", lambda *_: input_image.copy())
    monkeypatch.setattr(
        apply_mod,
        "prepare_output_folder",
        lambda path, **kwargs: _output_folder_resolution(
            Path(path),
            if_exists_policy=kwargs.get("if_exists_policy", "numbered"),
        ),
    )
    monkeypatch.setattr(apply_mod, "save_outputs", lambda *_args, **_kwargs: None)


def test_create_apply_save_folder_reports_numbered_existing_destination(tmp_path: Path, capsys):
    requested = tmp_path / "predictions"
    requested.mkdir()

    resolved = apply_mod._create_apply_save_folder(
        requested,
        overwrite=False,
        progress=apply_mod.InferenceProgress(enabled=False),
    )

    assert resolved == tmp_path / "predictions_01"
    assert resolved.is_dir()
    captured = capsys.readouterr()
    assert f"Folder {requested} already exists" in captured.out
    assert f"saving to {resolved} instead" in captured.out
    assert "Use --overwrite to replace the existing folder." in captured.out


def test_create_apply_save_folder_reports_overwrite_existing_destination(tmp_path: Path, capsys):
    requested = tmp_path / "predictions"
    requested.mkdir()
    old_file = requested / "old_prediction.tif"
    old_file.write_text("old")

    resolved = apply_mod._create_apply_save_folder(
        requested,
        overwrite=True,
        progress=apply_mod.InferenceProgress(enabled=False),
    )

    assert resolved == requested
    assert resolved.is_dir()
    assert not old_file.exists()
    captured = capsys.readouterr()
    assert f"Folder {requested} already exists" in captured.out
    assert "--overwrite enabled, replacing it." in captured.out


def test_create_apply_save_folder_reuses_existing_destination_without_deleting(tmp_path: Path, capsys):
    requested = tmp_path / "predictions"
    requested.mkdir()
    old_file = requested / "old_prediction.tif"
    old_file.write_text("old")

    resolved = apply_mod._create_apply_save_folder(
        requested,
        overwrite=False,
        reuse_folder=True,
        progress=apply_mod.InferenceProgress(enabled=False),
    )

    assert resolved == requested
    assert old_file.exists()
    captured = capsys.readouterr()
    assert f"Reusing output folder: {requested}" in captured.out


def test_create_apply_save_folder_reuse_existing_file_delegates_directory_check(tmp_path: Path):
    requested = tmp_path / "predictions"
    requested.write_text("not a directory")

    with pytest.raises(NotADirectoryError, match="not a directory"):
        apply_mod._create_apply_save_folder(
            requested,
            overwrite=False,
            reuse_folder=True,
            progress=apply_mod.InferenceProgress(enabled=False),
        )


def test_reuse_folder_refuses_existing_apply_outputs(tmp_path: Path):
    save_folder = tmp_path / "predictions"
    save_folder.mkdir()
    (save_folder / "first_pred.tif").touch()

    with pytest.raises(FileExistsError, match="Use --skip-existing to continue"):
        apply_mod._resolve_apply_files_for_output(
            ["first.tif", "second.tif"],
            save_folder=save_folder,
            name_file=None,
            limit_n_imgs=None,
            reuse_folder=True,
            skip_existing=False,
            progress=apply_mod.InferenceProgress(enabled=False),
        )


def test_skip_existing_filters_completed_predictions_and_limits_remaining(tmp_path: Path, capsys):
    save_folder = tmp_path / "predictions"
    save_folder.mkdir()
    (save_folder / "first_pred.tif").touch()

    selected = apply_mod._resolve_apply_files_for_output(
        ["first.tif", "second.tif", "third.tif"],
        save_folder=save_folder,
        name_file=None,
        limit_n_imgs=1,
        reuse_folder=True,
        skip_existing=True,
        progress=apply_mod.InferenceProgress(enabled=False),
    )

    assert selected == ["second.tif"]
    captured = capsys.readouterr()
    assert "Found #3 candidate files." in captured.out
    assert "Skipping 1 file(s) with existing predictions." in captured.out
    assert "Processing 1 remaining file(s)." in captured.out


def test_skip_existing_rejects_partial_apply_outputs(tmp_path: Path):
    save_folder = tmp_path / "predictions"
    save_folder.mkdir()
    (save_folder / "first_samples.tif").touch()

    with pytest.raises(FileExistsError, match="Partial existing apply outputs"):
        apply_mod._resolve_apply_files_for_output(
            ["first.tif", "second.tif"],
            save_folder=save_folder,
            name_file=None,
            limit_n_imgs=None,
            reuse_folder=True,
            skip_existing=True,
            progress=apply_mod.InferenceProgress(enabled=False),
        )


def test_run_apply_model_rejects_overwrite_with_in_place_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="in_place", save_folder=None),
    )

    with pytest.raises(ValueError, match="in-place apply output"):
        apply_mod.run_apply_model(
            cfg=cfg,
            model_dataset="dataset",
            model_subfolder="Upsamp",
            model_name="model",
            data_path=tmp_path,
            overwrite=True,
        )


def test_run_apply_model_rejects_reuse_folder_with_in_place_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="in_place", save_folder=None),
    )

    with pytest.raises(ValueError, match="in-place apply output"):
        apply_mod.run_apply_model(
            cfg=cfg,
            model_dataset="dataset",
            model_subfolder="Upsamp",
            model_name="model",
            data_path=tmp_path,
            reuse_folder=True,
        )


def test_run_apply_model_skip_existing_reuses_folder_and_only_processes_remaining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_folder = tmp_path / "predictions"
    save_folder.mkdir()
    (save_folder / "first_pred.tif").touch()
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (source_dir, ["first.tif", "second.tif"], None),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="folder", save_folder=save_folder),
    )
    read_files = []
    saved_names = []
    monkeypatch.setattr(
        apply_mod,
        "imread",
        lambda path: read_files.append(Path(path).name)
        or np.ones((8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (
            np.zeros((1, 1, 8, 8), dtype=np.float32),
            None,
        ),
    )
    monkeypatch.setattr(
        apply_mod,
        "save_outputs",
        lambda _tosave, _save_folder, img_name: saved_names.append(img_name),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="model",
        data_path=source_dir,
        skip_existing=True,
    )

    assert read_files == ["second.tif"]
    assert saved_names == ["second"]
    captured = capsys.readouterr()
    assert f"Reusing output folder: {save_folder}" in captured.out
    assert "Skipping 1 file(s) with existing predictions." in captured.out


def test_run_apply_model_keeps_legacy_stride_downsampling_when_fill_factor_is_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=2, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)

    captured = {}

    def _fake_predict(*args, **_kwargs):
        img = args[1]
        captured["img_shape"] = img.shape
        captured["ch_out"] = _kwargs.get("ch_out")
        return np.zeros_like(img), None

    monkeypatch.setattr(apply_mod, "predict_4d_stack", _fake_predict)
    monkeypatch.setattr(
        apply_mod,
        "generate_downsamp_inp",
        lambda *_args, **_kwargs: pytest.fail("generate_downsamp_inp should not be called"),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="model",
        data_path=tmp_path,
    )

    assert captured["img_shape"] == (1, 1, 4, 4)
    assert captured["ch_out"] is None


def test_run_apply_model_limits_number_of_input_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=None, fill_factor=None, limit_n_imgs=2)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (
            tmp_path,
            ["first.tif", "second.tif", "third.tif"],
            None,
        ),
    )

    read_files = []
    monkeypatch.setattr(
        apply_mod,
        "imread",
        lambda path: read_files.append(Path(path).name)
        or np.ones((8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (
            np.zeros((1, 1, 8, 8), dtype=np.float32),
            None,
        ),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="model",
        data_path=tmp_path,
    )

    assert read_files == ["first.tif", "second.tif"]


def test_run_apply_model_uses_deterministic_multiple_downsampling_when_fill_factor_is_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=2, fill_factor=0.5)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)

    captured = {}

    def _fake_generate_downsamp_inp(img, downsampling_prm):
        captured["downsampling_prm"] = downsampling_prm
        captured["source_shape"] = img.shape
        return np.ones((img.shape[0], 2, img.shape[-2] // 2, img.shape[-1] // 2), dtype=np.float32), None

    def _fake_predict(*args, **_kwargs):
        img = args[1]
        captured["img_shape"] = img.shape
        captured["ch_out"] = _kwargs.get("ch_out")
        return np.zeros_like(img), None

    monkeypatch.setattr(apply_mod, "generate_downsamp_inp", _fake_generate_downsamp_inp)
    monkeypatch.setattr(apply_mod, "predict_4d_stack", _fake_predict)

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="model",
        data_path=tmp_path,
    )

    assert captured["source_shape"] == (1, 1, 8, 8)
    assert captured["img_shape"] == (1, 2, 4, 4)
    assert captured["ch_out"] == 1
    assert captured["downsampling_prm"] == {
        "downsamp_factor": 2,
        "downsamp_method": "multiple",
        "multiple_prm": {"fill_factor": 0.5, "random": False},
    }


def test_run_apply_model_rejects_fill_factor_without_downsamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=None, fill_factor=0.5)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(apply_mod, "predict_4d_stack", lambda *_args, **_kwargs: (None, None))

    with pytest.raises(ValueError, match="requires `apply.downsamp`"):
        apply_mod.run_apply_model(
            cfg=cfg,
            model_dataset="dataset",
            model_subfolder="Upsamp",
            model_name="model",
            data_path=tmp_path,
        )


def test_run_apply_model_rejects_unsupported_deterministic_multiple_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=4, fill_factor=0.75)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "generate_downsamp_inp",
        lambda *_args, **_kwargs: pytest.fail("generate_downsamp_inp should not be called"),
    )
    monkeypatch.setattr(apply_mod, "predict_4d_stack", lambda *_args, **_kwargs: (None, None))

    with pytest.raises(ValueError, match="not implemented"):
        apply_mod.run_apply_model(
            cfg=cfg,
            model_dataset="dataset",
            model_subfolder="Upsamp",
            model_name="model",
            data_path=tmp_path,
        )


def test_run_apply_model_default_output_uses_source_and_model_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)

    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="default", save_folder=None),
    )
    captured = {}

    class _FakePaths:
        def inference_output_dir(self, *, source_name, model_name):
            captured["source_name"] = source_name
            captured["model_name"] = model_name
            return tmp_path / "inference" / source_name / model_name

    monkeypatch.setattr(apply_mod, "Paths", _FakePaths)
    _patch_prepare_output_folder(monkeypatch, captured)
    monkeypatch.setattr(
        apply_mod,
        "save_outputs",
        lambda _tosave, save_folder, _img_name: captured.setdefault(
            "writer_save_folder", Path(save_folder)
        ),
    )
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 8, 8), dtype=np.float32), None),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="mito_model_03",
        data_path=tmp_path,
    )

    assert captured["source_name"] == tmp_path.name
    assert captured["model_name"] == "mito_model_03"
    assert captured["save_folder"] == (
        tmp_path / "inference" / tmp_path.name / "mito_model_03"
    )


def test_run_apply_model_default_file_source_name_includes_parent_and_stem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source_dir = tmp_path / "mito_fast"
    source_dir.mkdir()
    source_file = source_dir / "c01.tiff"
    source_file.touch()
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (source_file, [""], source_file.name),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="default", save_folder=None),
    )
    captured = {}

    class _FakePaths:
        def inference_output_dir(self, *, source_name, model_name):
            captured["source_name"] = source_name
            return tmp_path / "inference" / source_name / model_name

    monkeypatch.setattr(apply_mod, "Paths", _FakePaths)
    _patch_prepare_output_folder(monkeypatch, captured)
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 8, 8), dtype=np.float32), None),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="mito_model_03",
        data_path=source_file,
    )

    assert captured["source_name"] == "mito_fast_c01"
    assert captured["save_folder"] == (
        tmp_path / "inference" / "mito_fast_c01" / "mito_model_03"
    )


def test_run_apply_model_folder_inside_directory_uses_model_only_folder_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (source_dir, ["input.tif"], None),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="folder_inside", save_folder=None),
    )
    captured = {}
    _patch_prepare_output_folder(monkeypatch, captured)
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 8, 8), dtype=np.float32), None),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="mito_model_03",
        data_path=source_dir,
    )

    assert captured["save_folder"] == source_dir / "Predict_Upsamp_mito_model_03"


def test_run_apply_model_folder_outside_directory_includes_source_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (source_dir, ["input.tif"], None),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="folder_outside", save_folder=None),
    )
    captured = {}
    _patch_prepare_output_folder(monkeypatch, captured)
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 8, 8), dtype=np.float32), None),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="mito_model_03",
        data_path=source_dir,
    )

    assert captured["save_folder"] == (
        tmp_path / "Predict_source_Upsamp_mito_model_03"
    )


def test_run_apply_model_folder_inside_file_includes_parent_and_file_source_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source_dir = tmp_path / "mito_fast"
    source_dir.mkdir()
    source_file = source_dir / "c01.tiff"
    source_file.touch()
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (source_file, [""], source_file.name),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="folder_inside", save_folder=None),
    )
    captured = {}
    _patch_prepare_output_folder(monkeypatch, captured)
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 8, 8), dtype=np.float32), None),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="mito_model_03",
        data_path=source_file,
    )

    assert captured["save_folder"] == (
        source_dir / "Predict_mito_fast_c01_Upsamp_mito_model_03"
    )


def test_run_apply_model_folder_outside_file_moves_to_parent_of_source_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source_dir = tmp_path / "mito_fast"
    source_dir.mkdir()
    source_file = source_dir / "c01.tiff"
    source_file.touch()
    cfg = _base_apply_config(downsamp=None, fill_factor=None)
    _patch_common_runtime(monkeypatch, tmp_path=tmp_path)
    monkeypatch.setattr(
        apply_mod,
        "resolve_prediction_inputs",
        lambda *_args, **_kwargs: (source_file, [""], source_file.name),
    )
    monkeypatch.setattr(
        apply_mod,
        "resolve_apply_output_policy",
        lambda _cfg: SimpleNamespace(mode="folder_outside", save_folder=None),
    )
    captured = {}
    _patch_prepare_output_folder(monkeypatch, captured)
    monkeypatch.setattr(
        apply_mod,
        "predict_4d_stack",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 8, 8), dtype=np.float32), None),
    )

    apply_mod.run_apply_model(
        cfg=cfg,
        model_dataset="dataset",
        model_subfolder="Upsamp",
        model_name="mito_model_03",
        data_path=source_file,
    )

    assert captured["save_folder"] == (
        tmp_path / "Predict_mito_fast_c01_Upsamp_mito_model_03"
    )
