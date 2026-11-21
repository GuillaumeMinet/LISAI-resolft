"""P2: public HDN performance choices, shared planning, and typed OOM handling."""
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
from pydantic import ValidationError

from lisai import (
    ApplyInferenceOverrides, ApplyOverrides, HDNOutOfMemoryError,
    HDNPerformanceOptions, RunRef, plan_inference, predict,
)
from lisai.config.models.inference import ApplyDefaults, ApplyInferenceDefaults
from lisai.evaluation.inference import planning
import importlib
apply_module = importlib.import_module("lisai.evaluation.run_apply_model")
from lisai.evaluation.run_apply_model import compute_prediction_array
from tests.evaluation.test_l3_runref_predict import _run_fixture


def test_public_options_are_strict_and_sparse():
    assert HDNPerformanceOptions().model_dump(exclude_unset=True) == {}
    assert HDNPerformanceOptions(sample_batch_size=3).model_dump(exclude_unset=True) == {
        "sample_batch_size": 3,
    }
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValidationError):
            HDNPerformanceOptions(sample_batch_size=invalid)
    with pytest.raises(ValidationError):
        HDNPerformanceOptions(precision="bf16")


def test_planning_shape_matches_actual_padded_tiles():
    from lisai.data.utils.tiling import adjust_for_tiling
    shape = (950, 950)
    tile = 512
    x = torch.zeros((1, 1, *shape))
    _, tile_shape, overlap, _ = adjust_for_tiling(x, tile, 16)
    assert planning.effective_inference_shape(shape, tiling_size=tile) == (
        tile_shape[0] + overlap[0], tile_shape[1] + overlap[1]
    )
    assert planning.effective_inference_shape((951, 951), tiling_size=None) == (960, 960)


def test_planner_cpu_controls_and_gpu_memory_heuristic(monkeypatch):
    class SmallModel:
        n_filters = 8
        n_layers = 1
        blocks_per_layer = 1
        def get_padded_size(self, shape):
            return tuple(shape)
        def prepare_inference_features(self, x):
            return ()

    model = SmallModel()
    cpu = planning.resolve_hdn_plan(
        model=model, device="cpu", image_shape=(950, 950), tiling_size=None,
        requested_samples=10,
    )
    assert cpu.precision == "fp32" and cpu.sample_batch_size == 1
    assert cpu.source == "auto"
    with pytest.raises(ValueError, match="compatible CUDA"):
        planning.resolve_hdn_plan(
            model=model, device="cpu", image_shape=(200, 200),
            tiling_size=None, requested_samples=5,
            performance=HDNPerformanceOptions(precision="fp16"),
        )
    with pytest.raises(ValueError, match="CPU"):
        planning.resolve_hdn_plan(
            model=model, device="cpu", image_shape=(200, 200),
            tiling_size=None, requested_samples=5,
            performance=HDNPerformanceOptions(sample_batch_size=2),
        )
    monkeypatch.setattr(planning, "_cuda_fp16_compatible", lambda device: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 * 1024**3, 16 * 1024**3))
    small = planning.resolve_hdn_plan(
        model=model, device="cuda:0", image_shape=(950, 950), tiling_size=None,
        requested_samples=20, performance=HDNPerformanceOptions(
            precision="auto", sample_batch_size="auto"),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
    large = planning.resolve_hdn_plan(
        model=model, device="cuda:0", image_shape=(950, 950), tiling_size=None,
        requested_samples=20, performance=HDNPerformanceOptions(
            precision="auto", sample_batch_size="auto"),
    )
    assert small.precision == large.precision == "fp16"
    assert small.sample_batch_size == large.sample_batch_size == 1
    assert large.free_gpu_memory_bytes == 16 * 1024**3
    assert large.effective_input_shape == (960, 960)
    manual = planning.resolve_hdn_plan(
        model=model, device="cuda:0", image_shape=(950, 950), tiling_size=512,
        requested_samples=3,
        performance=HDNPerformanceOptions(precision="fp32", sample_batch_size=12),
    )
    assert manual.sample_batch_size == 3 and manual.precision == "fp32"
    assert manual.source == "manual" and manual.tiling_size == 512


def test_public_preview_matches_executed_plan_and_file_metadata(settings, tmp_path):
    run_id, _, _ = _run_fixture(settings)
    ref = RunRef(run_id)
    raw = np.random.default_rng(4).uniform(1, 20, (32, 32)).astype(np.float32)
    overrides = ApplyOverrides(
        inference=ApplyInferenceOverrides(tiling_size="off", lvae_num_samples=3),
        saving={"lvae_save_samples": False},
    )
    preview = plan_inference(ref, raw, settings=settings, overrides=overrides)
    torch.manual_seed(11)
    restored, info = predict(ref, raw, settings=settings, overrides=overrides,
                             return_info=True, measure_gpu_memory=True)
    assert info.gpu_memory is None  # CPU test; optional instrumentation is harmless
    assert info.inference_plan == preview
    assert info.inference_plan.requested_samples == 3
    assert restored.shape == raw.shape and restored.dtype == np.float32
    tiff = tmp_path / "input.tif"
    tifffile.imwrite(tiff, raw)
    file_preview = plan_inference(ref, tiff, settings=settings, overrides=overrides)
    assert file_preview == preview
    torch.manual_seed(11)
    result = predict(ref, tiff, settings=settings, output_dir=tmp_path / "results",
                     overrides=overrides, measure_gpu_memory=True)
    assert result.inference_plan == preview
    assert len(result.prediction_paths) == 1
    assert result.info_by_path[result.prediction_paths[0]].inference_plan == preview
    assert result.info_by_path[result.prediction_paths[0]].gpu_memory is None
    np.testing.assert_allclose(tifffile.imread(result.prediction_paths[0]), restored, rtol=1e-5, atol=1e-5)


def test_per_field_keyword_priority_over_typed_overrides(settings):
    from lisai.api import _resolve_prediction_setup
    run_id, _, _ = _run_fixture(settings)
    overrides = ApplyOverrides(inference=ApplyInferenceOverrides(
        performance=HDNPerformanceOptions(precision="fp32", sample_batch_size=2),
    ))
    _, _, _, _, cfg = _resolve_prediction_setup(
        RunRef(run_id), settings=settings, overrides=overrides,
        config=None, device="auto",
        performance=HDNPerformanceOptions(precision="auto", sample_batch_size=5),
    )
    assert cfg.inference.performance.precision == "auto"
    assert cfg.inference.performance.sample_batch_size == 5
    _, _, _, _, cfg2 = _resolve_prediction_setup(
        RunRef(run_id), settings=settings, overrides=overrides,
        config=None, device="auto",
        performance=HDNPerformanceOptions(sample_batch_size=3),
    )
    assert cfg2.inference.performance.precision == "fp32"
    assert cfg2.inference.performance.sample_batch_size == 3


def test_cuda_oom_has_one_suggestion_and_no_retry(monkeypatch):
    seen = []
    def fail_once(*args, **kwargs):
        seen.append(kwargs)
        raise torch.cuda.OutOfMemoryError("simulated CUDA allocation failure")

    monkeypatch.setattr(apply_module, "predict_4d_stack", fail_once)
    saved = SimpleNamespace(
        is_lvae=True, context_length=None, data_norm_prm={"clip": 0},
        model_norm_prm={"data_mean": 10., "data_std": 4.}, upsampling_factor=1,
    )
    runtime = SimpleNamespace(model=object(), device=torch.device("cuda:0"), tiling_size=None)
    cfg = ApplyDefaults(inference=ApplyInferenceDefaults(
        lvae_num_samples=12, performance=HDNPerformanceOptions(
            precision="fp32", sample_batch_size=4,
        ),
    ))
    with pytest.raises(HDNOutOfMemoryError) as raised:
        compute_prediction_array(np.ones((16, 16), dtype=np.float32),
                                 saved_run=saved, runtime=runtime, cfg=cfg)
    exc = raised.value
    assert len(seen) == 1
    assert exc.requested_plan.requested_samples == 12
    assert exc.requested_plan.sample_batch_size == 4
    assert exc.suggested_performance.sample_batch_size == 1
    assert exc.suggested_performance.precision == "fp32"
    assert exc.requested_plan.tiling_size is None
    assert "simulated CUDA allocation failure" in str(exc)
    cfg_one = ApplyDefaults(inference=ApplyInferenceDefaults(
        performance=HDNPerformanceOptions(precision="fp32", sample_batch_size=1)
    ))
    with pytest.raises(HDNOutOfMemoryError) as single:
        compute_prediction_array(np.ones((16, 16), dtype=np.float32),
                                 saved_run=saved, runtime=runtime, cfg=cfg_one)
    assert single.value.suggested_performance is None
    assert single.value.suggested_tiling_size == 512
    assert "tiling" in single.value.suggestion_reason

    def unrelated(*args, **kwargs):
        raise RuntimeError("Not an allocation error")
    monkeypatch.setattr(apply_module, "predict_4d_stack", unrelated)
    with pytest.raises(RuntimeError, match="Not an allocation error"):
        compute_prediction_array(np.ones((16, 16), dtype=np.float32),
                                 saved_run=saved, runtime=runtime, cfg=cfg)


def test_non_hdn_explicit_performance_rejected():
    saved = SimpleNamespace(is_lvae=False)
    runtime = SimpleNamespace(model=object(), device=torch.device("cpu"), tiling_size=None)
    cfg = ApplyDefaults(inference=ApplyInferenceDefaults(
        performance=HDNPerformanceOptions(sample_batch_size=2),
    ))
    from lisai.evaluation.run_apply_model import _resolved_prediction_plan
    with pytest.raises(ValueError, match="non-HDN"):
        _resolved_prediction_plan(saved_run=saved, runtime=runtime,
                                  cfg=cfg, normalized=np.zeros((1, 1, 16, 16)))


def test_nonfinite_hdn_prediction_fails_instead_of_retrying():
    from lisai.evaluation.inference.engine import lvae_predict

    class NonfiniteModel(torch.nn.Module):
        mode_pred = True
        def prepare_inference_features(self, x):
            return (x,)
        def forward(self, x, cached_bottomup=None):
            return {"out_mean": torch.full_like(x, float("nan"))}

    model = NonfiniteModel()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        lvae_predict(torch.zeros((1, 1, 16, 16)), 3, model, None, "cpu",
                     sample_batch_size=2)
