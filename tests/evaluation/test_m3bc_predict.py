"""M3-B/C contract tests using a tiny real Torch model, no published weights."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch

from lisai import (
    predict, ApplyOverrides, ApplyInputOverrides, ApplyInferenceOverrides,
    InputNormalization, PredictionInfo, PredictionResult,
)
from lisai.evaluation.inference.normalization import normalize_apply_input
from lisai.evaluation.inference.stack import _build_temporal_input


class CenterFrame(torch.nn.Module):
    def forward(self, x):
        middle = x.shape[1] // 2
        return x[:, middle:middle+1]


@pytest.fixture
def fake_model(monkeypatch):
    from lisai.promoted_models import package
    import lisai.promoted_models as promoted_models
    from lisai.evaluation import runtime
    import importlib
    apply_mod = importlib.import_module("lisai.evaluation.run_apply_model")

    state = {"context_length": None, "clip": False, "mean": None, "std": None}

    def saved_run():
        return SimpleNamespace(
            context_length=state["context_length"],
            dataset_name="fake_dataset",
            is_lvae=False, data_norm_prm={"clip": state["clip"]},
            model_norm_prm=(
                {"data_mean": state["mean"], "data_std": state["std"]}
                if state["mean"] is not None else None
            ),
            upsampling_factor=1,
        )

    fake_loader = lambda name, *, paths: SimpleNamespace(
        saved_run=saved_run(), inference_config_path=None,
        weights_path=Path("fake-weights.pt"), noise_model_path=None,
        noise_model_norm_prm_path=None, manifest=SimpleNamespace(name=name),
    )
    monkeypatch.setattr(package, "load_promoted_model", fake_loader)
    monkeypatch.setattr(promoted_models, "load_promoted_model", fake_loader)
    fake_runtime = lambda **kwargs: SimpleNamespace(
        model=CenterFrame(), device="cpu", tiling_size=None,
    )
    monkeypatch.setattr(runtime, "initialize_runtime", fake_runtime)
    monkeypatch.setattr(apply_mod, "initialize_runtime", fake_runtime)
    return state


def _opts(mode="drop", norm="training"):
    return ApplyOverrides(
        input=ApplyInputOverrides(normalization=InputNormalization(mode=norm)),
        inference=ApplyInferenceOverrides(temporal_edge_mode=mode, tiling_size="off"),
    )


def test_public_exports_and_strict_dispatch(settings, fake_model):
    from lisai import predict as entrypoint
    assert entrypoint is predict
    with pytest.raises(TypeError, match="data must"):
        predict("sample", [[1, 2]], settings=settings)
    with pytest.raises(TypeError, match="settings"):
        predict("sample", np.ones((8, 8)), settings=None)
    with pytest.raises(ValueError, match="output_dir and overwrite"):
        predict("sample", np.ones((8, 8)), settings=settings, overwrite=True)
    with pytest.raises(ValueError, match="output-routing"):
        predict("sample", np.ones((8, 8)), settings=settings,
                overrides=ApplyOverrides(saving={"mode": "default"}))
    with pytest.raises(ValueError, match="YX, TYX or TZYX"):
        predict("sample", np.ones((1, 1, 1, 8, 8)), settings=settings)


def test_integer_array_not_mutated_and_alignment(settings, fake_model):
    fake_model["clip"] = 5
    image = np.arange(16*16, dtype=np.uint16).reshape(16, 16)
    original = image.copy()
    prediction, info = predict("sample", image, settings=settings, return_info=True)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_allclose(prediction, np.maximum(image, 5).astype(np.float32))
    assert prediction.dtype == np.float32
    assert info == PredictionInfo((0,), ())


def test_training_normalization_does_not_mutate_negative_float_array(settings, fake_model):
    fake_model["clip"] = 2
    image = np.array([[-9., 0.], [2., 8.]], dtype=np.float32).repeat(8, 0).repeat(8, 1)
    original = image.copy()
    result = predict("sample", image, settings=settings)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(result, np.maximum(original, 2))


@pytest.mark.parametrize("mode,n_out,indices,black", [
    ("drop", 4, (2, 3, 4, 5), ()),
    ("black_output", 8, tuple(range(8)), (0, 1, 6, 7)),
    ("black_input", 8, tuple(range(8)), ()),
])
def test_temporal_edge_modes(settings, fake_model, mode, n_out, indices, black):
    fake_model["context_length"] = 5
    fake_model["mean"], fake_model["std"] = 10., 2.
    stack = np.broadcast_to(np.arange(1, 9, dtype=np.float32)[:, None, None], (8, 16, 16)).copy()
    result, info = predict("sample", stack, settings=settings,
                           overrides=_opts(mode), return_info=True)
    assert result.shape == (n_out, 16, 16)
    assert info == PredictionInfo(indices, black)
    if mode == "drop":
        np.testing.assert_allclose(result[:, 0, 0], np.arange(3, 7))
    elif mode == "black_output":
        np.testing.assert_array_equal(result[list(black)], 0)
        np.testing.assert_allclose(result[2:6, 0, 0], np.arange(3, 7))
    else:
        np.testing.assert_allclose(result[:, 0, 0], np.arange(1, 9))


@pytest.mark.parametrize("mode", ["drop", "black_output", "black_input"])
def test_short_stack_always_raises(settings, fake_model, mode):
    fake_model["context_length"] = 5
    with pytest.raises(ValueError, match="requires at least 5 input frames, but received 3"):
        predict("sample", np.ones((3, 16, 16), dtype=np.float32), settings=settings,
                overrides=_opts(mode))


def test_black_input_preserves_valid_neighbors():
    data = np.broadcast_to(np.arange(1, 9, dtype=np.float32)[None, :, None, None],
                           (1, 8, 8, 8))
    result = _build_temporal_input(data, z=0, t=0, context_length=5,
                                   temporal_edge_mode="black_input", normalized_black_value=-7)
    np.testing.assert_array_equal(result[0, :, 0, 0], [-7, -7, 1, 2, 3])


def test_black_padding_uses_image_reference_stats(settings, fake_model, monkeypatch):
    fake_model["context_length"] = 5
    fake_model["mean"], fake_model["std"] = 50., 10.
    data = np.broadcast_to(np.arange(1, 9, dtype=np.float32)[:, None, None],
                           (8, 16, 16)).copy()
    # Predict should not compute statistics on padded zeros. The transformed
    # raw zero is not generally zero in normalized model space.
    input_ref = np.zeros((1, 1), dtype=np.float32)
    expected = normalize_apply_input(
        input_ref, False, {"clip": False}, {"data_mean": 50., "data_std": 10.},
        InputNormalization(mode="image"), reference_image=data,
    )[0, 0]
    import importlib
    run_module = importlib.import_module("lisai.evaluation.run_apply_model")
    captured = []
    original = run_module.predict_4d_stack
    def recording(*args, **kwargs):
        captured.append(kwargs["normalized_black_value"])
        return original(*args, **kwargs)
    monkeypatch.setattr(run_module, "predict_4d_stack", recording)
    predict("sample", data, settings=settings, overrides=_opts("black_input", "image"))
    np.testing.assert_allclose(captured, [expected])


def test_tzyx_roundtrip_and_time_selection(settings, fake_model):
    # Input is TZYX; the public output must not be internal ZTYX.
    data = np.arange(8*3*16*16, dtype=np.float32).reshape(8, 3, 16, 16)
    result, info = predict("sample", data, settings=settings, return_info=True,
                           overrides=ApplyOverrides(input={"timelapse_max": 5}))
    np.testing.assert_allclose(result, data[:5])
    assert result.shape == (5, 3, 16, 16)
    assert info.source_frame_indices == tuple(range(5))


@pytest.mark.parametrize("norm", ["training", "image", "manual"])
@pytest.mark.parametrize("edge", ["drop", "black_output", "black_input"])
def test_tiff_matches_array_with_same_cfg(tmp_path, settings, fake_model, norm, edge):
    fake_model["context_length"] = 5
    fake_model["mean"], fake_model["std"] = 25., 4.
    data = np.arange(8*16*16, dtype=np.float32).reshape(8, 16, 16)
    src = tmp_path / "sample.tif"
    tifffile.imwrite(src, data)
    overrides = _opts(edge, norm)
    if norm == "manual":
        overrides = ApplyOverrides(input=ApplyInputOverrides(
            normalization=InputNormalization(mode="manual", gain=2., offset=-9.)),
            inference=ApplyInferenceOverrides(temporal_edge_mode=edge, tiling_size="off"))
    expected, expected_info = predict("sample", data, settings=settings,
                                     overrides=overrides, return_info=True)
    written = predict("sample", src, settings=settings, overrides=overrides,
                      output_dir=tmp_path / "out")
    assert isinstance(written, PredictionResult)
    assert written.output_dir == tmp_path / "out"
    assert len(written.prediction_paths) == 1
    assert written.info_by_path[written.prediction_paths[0]] == expected_info
    np.testing.assert_allclose(tifffile.imread(written.prediction_paths[0]), expected, rtol=1e-5)


def test_file_paths_numbering_and_dir_collision(tmp_path, settings, fake_model):
    source = tmp_path / "source"
    source.mkdir()
    tifffile.imwrite(source / "cell.1.tif", np.ones((16, 16), dtype=np.float32))
    tifffile.imwrite(source / "cell.2.tiff", np.full((16, 16), 2., dtype=np.float32))
    (source / "subfolder").mkdir()
    tifffile.imwrite(source / "subfolder" / "nested.tif", np.ones((16, 16)))
    result = predict("sample", source, settings=settings, output_dir=tmp_path / "pred")
    assert sorted(p.name for p in result.prediction_paths) == ["cell.1_pred.tif", "cell.2_pred.tif"]
    assert all(p.is_file() for p in result.prediction_paths)
    again = predict("sample", source, settings=settings, output_dir=tmp_path / "pred")
    assert again.output_dir != result.output_dir
    assert all(p.parent == again.output_dir for p in again.prediction_paths)
    tifffile.imwrite(source / "cell.1.tiff", np.ones((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="same prediction filename"):
        predict("sample", source, settings=settings, output_dir=tmp_path / "newpred")


def test_output_conflicts_and_return_info_on_paths(tmp_path, settings, fake_model):
    src = tmp_path / "input.tif"
    tifffile.imwrite(src, np.ones((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="Conflicting explicit output_dir"):
        predict("sample", src, settings=settings, output_dir=tmp_path / "a",
                overrides=ApplyOverrides(saving={"save_folder": str(tmp_path / "b")}))
    with pytest.raises(ValueError, match="return_info"):
        predict("sample", src, settings=settings, return_info=True)


def test_volumetric_tiff_and_array_have_same_axes(tmp_path, settings, fake_model):
    data = np.arange(5 * 2 * 16 * 16, dtype=np.float32).reshape(5, 2, 16, 16)
    src = tmp_path / "volume.tif"
    tifffile.imwrite(src, data, imagej=True, metadata={"axes": "TZYX"})
    expected, info = predict("sample", data, settings=settings, return_info=True)
    written = predict("sample", src, settings=settings, output_dir=tmp_path / "volumes")
    assert written.info_by_path[written.prediction_paths[0]] == info
    observed = tifffile.imread(written.prediction_paths[0])
    assert observed.shape == data.shape
    np.testing.assert_allclose(observed, expected)


def test_single_frame_legacy_edge_setting_and_conflicts(tmp_path, settings, fake_model):
    from lisai.config.models.inference import ApplyDefaults
    assert ApplyDefaults.model_validate(
        {"inference": {"dark_frame_context_length": True}}
    ).inference.temporal_edge_mode == "black_input"
    assert ApplyDefaults.model_validate(
        {"inference": {"dark_frame_context_length": False}}
    ).inference.temporal_edge_mode == "drop"
    with pytest.raises(ValueError, match="not both"):
        ApplyDefaults.model_validate({"inference": {
            "dark_frame_context_length": True,
            "temporal_edge_mode": "drop",
        }})
    with pytest.raises(ValueError, match="TIFF"):
        source = tmp_path / "image.jpeg"
        source.touch()
        predict("sample", source, settings=settings)


def test_short_stack_exact_context_length_is_accepted(settings, fake_model):
    fake_model["context_length"] = 5
    src = np.ones((5, 16, 16), dtype=np.float32)
    pred, info = predict("sample", src, settings=settings,
                         overrides=_opts("drop"), return_info=True)
    assert pred.shape == (1, 16, 16)
    assert info.source_frame_indices == (2,)
