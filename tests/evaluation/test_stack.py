from __future__ import annotations

import numpy as np
import pytest

import lisai.evaluation.inference.stack as stack_mod


class RecordingProgress:
    def __init__(self):
        self.calls = []
        self.items = []
        self.messages = []

    def write(self, message):
        self.messages.append(message)

    def track(self, iterable, *, total=None, desc=None, level=0, leave=None):
        self.calls.append(
            {"total": total, "desc": desc, "level": level, "leave": leave}
        )
        for item in iterable:
            self.items.append(item)
            yield item


def test_predict_4d_stack_keeps_channel_axis_for_non_timelapse(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_infer_batch(*args, **_kwargs):
        x = args[1]
        captured["shape"] = tuple(x.shape)
        return {"prediction": np.zeros((1, 1, x.shape[-2], x.shape[-1]), dtype=np.float32)}

    monkeypatch.setattr(stack_mod, "infer_batch", _fake_infer_batch)

    img = np.ones((1, 2, 32, 32), dtype=np.float32)
    pred_stack, samples_stack = stack_mod.predict_4d_stack(
        model=None,
        img=img,
        timelapse=False,
        ch_out=None,
        device="cpu",
        is_lvae=False,
        tiling_size=None,
        lvae_num_samples=None,
        lvae_save_samples=False,
        upsamp=1,
        context_length=None,
        dark_frame_context_length=False,
        verbose=False,
    )

    assert captured["shape"] == (1, 2, 32, 32)
    assert pred_stack.shape == (1, 1, 32, 32)
    assert samples_stack is None


def test_predict_4d_stack_tracks_verbose_stack_progress(monkeypatch: pytest.MonkeyPatch):
    captured = []

    def _fake_infer_batch(*args, **kwargs):
        x = args[1]
        captured.append(kwargs)
        return {"prediction": np.zeros((1, 1, x.shape[-2], x.shape[-1]), dtype=np.float32)}

    monkeypatch.setattr(stack_mod, "infer_batch", _fake_infer_batch)

    progress = RecordingProgress()
    img = np.ones((2, 3, 16, 16), dtype=np.float32)
    pred_stack, samples_stack = stack_mod.predict_4d_stack(
        model=None,
        img=img,
        timelapse=True,
        ch_out=None,
        device="cpu",
        is_lvae=False,
        tiling_size=None,
        lvae_num_samples=None,
        lvae_save_samples=False,
        upsamp=1,
        context_length=None,
        dark_frame_context_length=False,
        verbose=True,
        progress=progress,
    )

    assert progress.calls == [
        {"total": 6, "desc": "Stack inference", "level": 0, "leave": True}
    ]
    assert progress.items == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert all(call["progress"] is progress for call in captured)
    assert all(call["progress_level"] == 1 for call in captured)
    assert pred_stack.shape == (2, 3, 16, 16)
    assert samples_stack is None


def test_predict_4d_stack_skips_single_item_stack_progress(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_infer_batch(*args, **kwargs):
        x = args[1]
        captured.update(kwargs)
        return {"prediction": np.zeros((1, 1, x.shape[-2], x.shape[-1]), dtype=np.float32)}

    monkeypatch.setattr(stack_mod, "infer_batch", _fake_infer_batch)

    progress = RecordingProgress()
    img = np.ones((1, 2, 16, 16), dtype=np.float32)
    pred_stack, samples_stack = stack_mod.predict_4d_stack(
        model=None,
        img=img,
        timelapse=False,
        ch_out=None,
        device="cpu",
        is_lvae=False,
        tiling_size=None,
        lvae_num_samples=None,
        lvae_save_samples=False,
        upsamp=1,
        context_length=None,
        dark_frame_context_length=False,
        verbose=True,
        progress=progress,
    )

    assert progress.calls == []
    assert captured["progress"] is progress
    assert captured["progress_level"] == 0
    assert pred_stack.shape == (1, 1, 16, 16)
    assert samples_stack is None


def test_predict_4d_stack_writes_skipped_frames_through_progress(monkeypatch: pytest.MonkeyPatch):
    def _fake_infer_batch(*args, **_kwargs):
        x = args[1]
        return {"prediction": np.zeros((1, 1, x.shape[-2], x.shape[-1]), dtype=np.float32)}

    monkeypatch.setattr(stack_mod, "infer_batch", _fake_infer_batch)

    progress = RecordingProgress()
    img = np.ones((1, 5, 16, 16), dtype=np.float32)
    stack_mod.predict_4d_stack(
        model=None,
        img=img,
        timelapse=True,
        ch_out=1,
        device="cpu",
        is_lvae=False,
        tiling_size=None,
        lvae_num_samples=None,
        lvae_save_samples=False,
        upsamp=1,
        context_length=3,
        dark_frame_context_length=False,
        verbose=True,
        progress=progress,
    )

    assert progress.messages == [
        "Skipping frame 0 because not enough context_length",
        "Skipping frame 4 because not enough context_length",
    ]
