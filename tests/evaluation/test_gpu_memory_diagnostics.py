"""S1: runtime CUDA allocator measurements and explicit batch-1 policy."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lisai import GPUMemoryInfo, HDNPerformanceOptions, HDNOutOfMemoryError, InferencePlan
from lisai.evaluation.inference.memory import (
    start_gpu_memory_measurement, finish_gpu_memory_measurement,
)
from lisai.evaluation.inference.planning import resolve_hdn_plan
from lisai.evaluation.inference.results import PredictionInfo
from lisai.config.models.inference import ApplyDefaults, ApplyInferenceDefaults
from lisai.evaluation.run_apply_model import compute_prediction_array

GIB = 1024**3


def test_memory_diagnostics_cpu_is_none():
    assert start_gpu_memory_measurement("cpu") is None
    assert finish_gpu_memory_measurement(None) is None
    assert PredictionInfo((0,), inference_seconds=1.0) == PredictionInfo((0,), inference_seconds=9.0)


def test_allocator_measurements_cuda_mock(monkeypatch):
    calls = []
    memory = SimpleNamespace(allocated=2 * GIB, reserved=3 * GIB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (
        (10 * GIB, 16 * GIB) if not calls or calls.count("sync") == 1
        else (8 * GIB, 16 * GIB)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: memory.allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: memory.reserved)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 5 * GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 7 * GIB)

    baseline = start_gpu_memory_measurement("cuda:0")
    assert calls == ["sync", "reset"]
    memory.allocated = 2 * GIB
    memory.reserved = 7 * GIB
    stats = finish_gpu_memory_measurement(baseline)
    assert calls == ["sync", "reset", "sync"]
    assert stats.device == "cuda:0"
    assert stats.allocated_before_bytes == 2 * GIB
    assert stats.reserved_before_bytes == 3 * GIB
    assert stats.device_total_bytes == 16 * GIB
    assert stats.peak_allocated_bytes == 5 * GIB
    assert stats.peak_reserved_bytes == 7 * GIB
    assert stats.additional_peak_allocated_bytes == 3 * GIB
    assert stats.reusable_cache_before_bytes == GIB
    assert stats.reserved_after_bytes == 7 * GIB


def test_opt_in_metadata_on_shared_inference_path(monkeypatch):
    import importlib
    mod = importlib.import_module("lisai.evaluation.run_apply_model")
    stats = GPUMemoryInfo("cuda:0", GIB, 2*GIB, 10*GIB, 16*GIB, 3*GIB, 4*GIB,
                          GIB, 4*GIB, 8*GIB)
    calls = []
    monkeypatch.setattr(mod, "start_gpu_memory_measurement", lambda device: (calls.append("start"), object())[1])
    monkeypatch.setattr(mod, "finish_gpu_memory_measurement", lambda baseline: (calls.append("end"), stats)[1] if baseline is not None else None)
    monkeypatch.setattr(mod, "predict_4d_stack", lambda *args, **kwargs: (
        calls.append("stack") or np.ones((1, 1, 16, 16), dtype=np.float32), None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    saved = SimpleNamespace(is_lvae=False, context_length=None, data_norm_prm={"clip": 0},
                            model_norm_prm={"data_mean": 10., "data_std": 4.}, upsampling_factor=1)
    runtime = SimpleNamespace(model=object(), device="cuda:0", tiling_size=None)
    cfg = ApplyDefaults(inference=ApplyInferenceDefaults(tiling_size="off"))
    output, _, _, info, _ = compute_prediction_array(
        np.ones((16, 16), dtype=np.float32), saved_run=saved, runtime=runtime,
        cfg=cfg, measure_gpu_memory=True,
    )
    assert output.shape == (16, 16)
    assert info.gpu_memory == stats
    assert calls.index("start") < calls.index("stack") < calls.index("end")
    calls.clear()
    _, _, _, default_info, _ = compute_prediction_array(
        np.ones((16, 16), dtype=np.float32), saved_run=saved, runtime=runtime,
        cfg=cfg,
    )
    assert default_info.gpu_memory is None
    assert "start" not in calls and "end" not in calls
    assert info == default_info  # diagnostics must not affect result identity


def test_defaults_and_legacy_auto_batch_policy(monkeypatch):
    from lisai.evaluation.inference import planning
    class Model:
        def prepare_inference_features(self, x): return ()
    monkeypatch.setattr(planning, "_cuda_fp16_compatible", lambda device: True)
    monkeypatch.setattr(planning, "_cuda_available_including_cache", lambda device: (GIB, 0, GIB))
    for options in (None, HDNPerformanceOptions(), HDNPerformanceOptions(sample_batch_size="auto")):
        plan = resolve_hdn_plan(model=Model(), device="cuda:0", image_shape=(950, 950),
                                tiling_size=None, requested_samples=20, performance=options)
        assert plan.precision == "fp16"
        assert plan.sample_batch_size == 1
    assert ApplyInferenceDefaults().tiling_size == "off"
    assert ApplyInferenceDefaults(tiling_size="auto").tiling_size == "auto"


def test_oom_tiling_and_experimental_batch_advice():
    base = InferencePlan(device="cuda:0", precision="fp16", reuse_encoder=True,
                         requested_samples=20, sample_batch_size=1, tiling_size=None,
                         effective_input_shape=(960, 960), source="auto", reason="test")
    for requested_tile, recommended_tile in [(None, 512), (1024, 512), (512, 256), (256, 128), (128, None)]:
        oom = HDNOutOfMemoryError(replace(base, tiling_size=requested_tile),
                                  torch.cuda.OutOfMemoryError("simulated"))
        assert oom.suggested_performance is None
        assert oom.suggested_tiling_size == recommended_tile
        assert "simulated" in str(oom)
    oom_batch = HDNOutOfMemoryError(replace(base, sample_batch_size=6),
                                    torch.cuda.OutOfMemoryError("simulated"))
    assert oom_batch.suggested_performance.sample_batch_size == 1
    assert oom_batch.suggested_tiling_size is None
