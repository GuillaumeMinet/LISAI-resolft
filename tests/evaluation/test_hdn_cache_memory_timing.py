"""Regression checks for CUDA cache-aware planning and truthful inference timing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lisai import HDNPerformanceOptions
from lisai.config.models.inference import ApplyDefaults, ApplyInferenceDefaults
from lisai.evaluation.inference import planning
from lisai.evaluation.inference.results import PredictionInfo
from lisai.evaluation.run_apply_model import compute_prediction_array

GIB = 1024 ** 3


class ReferenceHDN:
    n_filters = 64
    n_layers = 5
    blocks_per_layer = 6

    def prepare_inference_features(self, x):
        return ()

    def get_padded_size(self, shape):
        return tuple(shape)


def _plan(monkeypatch, *, driver_free=16, reserved=0, allocated=0,
          inactive_split=0, image_shape=(950, 950),
          precision="auto", samples=20, model=None):
    monkeypatch.setattr(planning, "_cuda_fp16_compatible", lambda device: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (int(driver_free * GIB), 16 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: int(reserved * GIB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: int(allocated * GIB))
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {
        "inactive_split_bytes.all.current": int(inactive_split * GIB),
    })
    return planning.resolve_hdn_plan(
        model=model or ReferenceHDN(), device="cuda:0", image_shape=image_shape,
        tiling_size=None, requested_samples=samples,
        performance=HDNPerformanceOptions(precision=precision),
    )


def test_auto_batch_always_one_even_when_cuda_cache_or_vram_changes(monkeypatch):
    initial = _plan(monkeypatch)
    after_large_batch = _plan(monkeypatch, driver_free=10, reserved=6,
                              allocated=0.25)
    pressured = _plan(monkeypatch, driver_free=3, reserved=8, allocated=7)
    assert initial.sample_batch_size == after_large_batch.sample_batch_size == pressured.sample_batch_size == 1
    assert initial.precision == "fp16"
    assert after_large_batch.free_gpu_memory_bytes == 10 * GIB
    assert after_large_batch.reusable_cuda_cache_bytes == int(5.75 * GIB)
    assert after_large_batch.estimated_usable_gpu_memory_bytes == int(15.75 * GIB)


def test_auto_batch_independent_of_samples_shape_precision_and_missing_stats(monkeypatch):
    assert _plan(monkeypatch, samples=3).sample_batch_size == 1
    assert _plan(monkeypatch, samples=5).sample_batch_size == 1
    assert _plan(monkeypatch, precision='fp32').sample_batch_size == 1
    assert _plan(monkeypatch, image_shape=(1900, 1900)).sample_batch_size == 1
    monkeypatch.setattr(planning, '_cuda_available_including_cache', lambda device: (None, None, None))
    assert planning.resolve_hdn_plan(
        model=ReferenceHDN(), device='cuda:0', image_shape=(950, 950),
        tiling_size=None, requested_samples=20,
        performance=HDNPerformanceOptions(sample_batch_size='auto'),
    ).sample_batch_size == 1
    assert planning.resolve_hdn_plan(
        model=ReferenceHDN(), device='cuda:0', image_shape=(950, 950),
        tiling_size=512, requested_samples=3,
        performance=HDNPerformanceOptions(precision='fp32', sample_batch_size=8),
    ).sample_batch_size == 3


def test_cpu_inference_time_is_optional_backward_compatible(monkeypatch):
    import importlib
    apply_mod = importlib.import_module('lisai.evaluation.run_apply_model')
    def fake_stack(model, normalized, **kwargs):
        return np.zeros((1, 1, 16, 16), dtype=np.float32), None
    monkeypatch.setattr(apply_mod, 'predict_4d_stack', fake_stack)
    saved = SimpleNamespace(
        is_lvae=False, context_length=None,
        data_norm_prm={'clip': 0}, model_norm_prm={'data_mean': 10., 'data_std': 4.},
        upsampling_factor=1,
    )
    runtime = SimpleNamespace(model=object(), device=torch.device('cpu'), tiling_size=None)
    cfg = ApplyDefaults(inference=ApplyInferenceDefaults())
    output, _, _, info, _ = compute_prediction_array(
        np.ones((16, 16), dtype=np.float32), saved_run=saved, runtime=runtime, cfg=cfg,
    )
    assert output.shape == (16, 16)
    assert info.inference_seconds is not None and info.inference_seconds >= 0
    assert PredictionInfo((0,)) == PredictionInfo((0,), (), None, None)


def test_cuda_timer_synchronizes_before_and_after_stack(monkeypatch):
    import importlib
    module = importlib.import_module('lisai.evaluation.run_apply_model')
    events = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))
    readings = iter((11.0, 15.25))
    monkeypatch.setattr(module, 'perf_counter', lambda: next(readings))
    def stack(*args, **kwargs):
        events.append('stack')
        return np.ones((1, 1, 16, 16), dtype=np.float32), None
    monkeypatch.setattr(module, 'predict_4d_stack', stack)
    saved = SimpleNamespace(
        is_lvae=False, context_length=None,
        data_norm_prm={'clip': 0}, model_norm_prm={'data_mean': 10., 'data_std': 4.},
        upsampling_factor=1,
    )
    runtime = SimpleNamespace(model=object(), device=torch.device('cuda:0'), tiling_size=None)
    _, _, _, info, _ = compute_prediction_array(
        np.arange(256, dtype=np.float32).reshape(16, 16),
        saved_run=saved, runtime=runtime, cfg=ApplyDefaults(),
    )
    assert events == ['sync', 'stack', 'sync']
    assert info.inference_seconds == 4.25
    assert info == PredictionInfo((0,))
