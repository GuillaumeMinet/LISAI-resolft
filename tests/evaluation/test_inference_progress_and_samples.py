"""Structured inference progress, cooperative cancellation and opt-in HDN samples."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lisai.api import predict
from lisai.config.models.inference import ApplyDefaults, ApplyInferenceDefaults
from lisai.evaluation.inference.engine import predict as engine_predict
from lisai.evaluation.inference.progress import (
    InferenceProgressEvent, InferenceProgressReporter, count_tiles,
)
from lisai.evaluation.inference.results import ArrayPrediction
from lisai.evaluation.run_apply_model import compute_prediction_array


class CountingHDN(torch.nn.Module):
    mode_pred = False

    def __init__(self):
        super().__init__()
        self.batches = []

    def prepare_inference_features(self, inp):
        return (inp,)

    def forward(self, inp, cached_bottomup=None):
        self.batches.append(inp.shape[0])
        return {"out_mean": inp[:, :1] + 2.0}


class Center(torch.nn.Module):
    def forward(self, inp):
        return inp[:, inp.shape[1] // 2: inp.shape[1] // 2 + 1]


def _reporter(events, *, throttle=0):
    return InferenceProgressReporter(events.append, throttle_seconds=throttle)


def _fake_saved(*, is_hdn, context=None):
    return SimpleNamespace(
        is_lvae=is_hdn, context_length=context, upsampling_factor=1,
        data_norm_prm={"clip": 0},
        model_norm_prm={"data_mean": 0., "data_std": 1.},
    )


def _config(*, samples=5, mode="drop", save_samples=False):
    return ApplyDefaults(
        inference=ApplyInferenceDefaults(
            tiling_size="off", lvae_num_samples=samples,
            temporal_edge_mode=mode,
        ),
        input={"normalization": {"mode": "training"}},
        saving={"lvae_save_samples": save_samples},
    )


def test_tiled_hdn_progress_counts_every_draw_including_batch_remainder():
    model = CountingHDN()
    events = []
    tiles = count_tiles((256, 256), 128)
    assert tiles == 4
    progress = _reporter(events)
    progress.begin(total=tiles * 5, frames=1, samples=5)
    output = engine_predict(
        model, torch.ones(1, 1, 256, 256), device="cpu", is_lvae=True,
        num_samples=5, tiling_size=128, sample_batch_size=2,
        progress_reporter=progress, save_samples=True,
    )
    assert output['samples'].shape == (5, 256, 256)
    assert model.batches == [2, 2, 1] * 4
    inference_events = [e for e in events if e.stage == 'inference']
    assert [e.completed for e in inference_events] == [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20]
    assert inference_events[-1].tile_index == 4
    assert inference_events[-1].sample_index == 5
    np.testing.assert_allclose(output['prediction'], 3.0)
    np.testing.assert_allclose(np.squeeze(output['prediction']), output['samples'].mean(axis=0))


def test_cancel_in_hdn_loop_restores_prediction_state():
    class Cancelled(BaseException):
        pass
    model = CountingHDN().train()
    events = []
    checks = [0]

    def checkpoint():
        checks[0] += 1
        if events and events[-1].completed >= 2:
            raise Cancelled('requested')

    reporter = InferenceProgressReporter(events.append, checkpoint, throttle_seconds=0)
    reporter.begin(total=8, frames=1, samples=8)
    with pytest.raises(Cancelled):
        engine_predict(
            model, torch.ones(1, 1, 32, 32), device='cpu', is_lvae=True,
            num_samples=8, progress_reporter=reporter,
        )
    assert len(model.batches) <= 3
    assert model.training is True
    assert model.mode_pred is False
    assert checks[0] >= 2


def test_temporal_drop_progress_counts_only_processed_timepoints():
    events = []
    reporter = _reporter(events)
    saved = _fake_saved(is_hdn=False, context=5)
    runtime = SimpleNamespace(model=Center(), device=torch.device('cpu'), tiling_size=None)
    pred, samples, _, info, _ = compute_prediction_array(
        np.ones((8, 16, 16), dtype=np.float32),
        saved_run=saved, runtime=runtime, cfg=_config(),
        progress_reporter=reporter,
    )
    assert pred.shape == (4, 16, 16)
    assert samples is None
    assert info.source_frame_indices == (2, 3, 4, 5)
    inference_events = [e for e in events if e.stage == 'inference']
    assert [(e.completed, e.total) for e in inference_events] == [(0, 4), (4, 4)] or (
        [e.completed for e in inference_events] == [0, 1, 2, 3, 4]
    )
    assert events[-1].stage == 'done'


def test_public_array_samples_opt_in_and_legacy_return_shapes(monkeypatch, settings):
    import lisai.api as api_mod
    fake_saved = _fake_saved(is_hdn=True)
    model = CountingHDN()
    cfg = _config(samples=3)
    monkeypatch.setattr(api_mod, '_resolve_prediction_setup', lambda *args, **kwargs:
                        (None, None, fake_saved, None, cfg))
    monkeypatch.setattr(api_mod, '_load_prediction_runtime', lambda **kwargs:
                        SimpleNamespace(model=model, device='cpu', tiling_size=None))
    data = np.ones((16, 16), dtype=np.float32)
    no_samples = predict('fake', data, settings=settings)
    assert isinstance(no_samples, np.ndarray)
    image, info = predict('fake', data, settings=settings, return_info=True)
    assert isinstance(image, np.ndarray)
    assert info.source_frame_indices == (0,)
    events = []
    samples_out = predict('fake', data, settings=settings, return_samples=True,
                          on_progress=events.append)
    assert isinstance(samples_out, ArrayPrediction)
    assert samples_out.samples.shape == (3, 16, 16)
    np.testing.assert_allclose(samples_out.prediction, samples_out.samples.mean(axis=0))
    assert events[0].stage == 'loading'
    assert events[-1].stage == 'done'
    assert [e.total for e in events if e.stage == 'inference'][-1] == 3


def test_public_non_hdn_rejects_samples_before_runtime(monkeypatch, settings):
    import lisai.api as api_mod
    saved = _fake_saved(is_hdn=False)
    monkeypatch.setattr(api_mod, '_resolve_prediction_setup', lambda *args, **kwargs:
                        (None, None, saved, None, _config()))
    monkeypatch.setattr(api_mod, '_load_prediction_runtime', lambda **kwargs:
                        pytest.fail('should not load non-HDN model'))
    with pytest.raises(ValueError, match='HDN/LVAE'):
        predict('fake', np.zeros((8, 8)), settings=settings, return_samples=True)

@pytest.mark.parametrize('shape', [(16, 16), (4, 16, 16), (3, 2, 16, 16)])
def test_hdn_sample_array_axes_follow_prediction_for_2d_time_and_volume(
    monkeypatch, settings, shape,
):
    import lisai.api as api_mod
    saved = _fake_saved(is_hdn=True)
    monkeypatch.setattr(api_mod, '_resolve_prediction_setup', lambda *args, **kwargs:
                        (None, None, saved, None, _config(samples=2)))
    monkeypatch.setattr(api_mod, '_load_prediction_runtime', lambda **kwargs:
                        SimpleNamespace(model=CountingHDN(), device='cpu', tiling_size=None))
    data = np.ones(shape, dtype=np.float32)
    result = predict('fake', data, settings=settings, return_samples=True)
    assert result.prediction.shape == shape
    assert result.samples.shape == (2, *shape)
    np.testing.assert_allclose(result.samples.mean(axis=0), result.prediction)


def test_tiled_progress_count_matches_real_engine_for_odd_dims():
    tiles = count_tiles((255, 257), 128)
    assert tiles >= 1
    events = []
    reporter = _reporter(events)
    reporter.begin(total=tiles, frames=1, samples=1)
    result = engine_predict(Center(), torch.ones((1, 1, 255, 257)),
                            device='cpu', tiling_size=128, progress_reporter=reporter)
    assert result['prediction'].shape[-2:] == (255, 257)
    assert events[-1].completed == tiles


def test_black_output_keeps_edge_frames_but_does_not_count_them_as_draws():
    saved = _fake_saved(is_hdn=False, context=3)
    runtime = SimpleNamespace(model=Center(), device=torch.device('cpu'), tiling_size=None)
    events = []
    reporter = _reporter(events)
    pred, _, _, info, _ = compute_prediction_array(
        np.ones((6, 16, 16), dtype=np.float32),
        saved_run=saved, runtime=runtime,
        cfg=_config(mode='black_output'),
        progress_reporter=reporter,
    )
    assert pred.shape == (6, 16, 16)
    assert info.black_output_indices == (0, 5)
    assert pred[0].max() == pred[-1].max() == 0
    steps = [ev for ev in events if ev.stage == 'inference']
    assert steps[-1].completed == steps[-1].total == 4
