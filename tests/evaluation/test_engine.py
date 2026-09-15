from __future__ import annotations

import numpy as np
import torch

import lisai.evaluation.inference.engine as engine_mod


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


def test_predict_uses_injected_progress_for_tiling(monkeypatch):
    def _fake_make_prediction(_model, patch, _device, _is_lvae, _num_samples):
        return {"prediction": patch.cpu().detach().numpy()}

    monkeypatch.setattr(engine_mod, "make_prediction", _fake_make_prediction)

    progress = RecordingProgress()
    outputs = engine_mod.predict(
        model=None,
        inp=torch.ones((1, 1, 200, 200), dtype=torch.float32),
        tiling_size=100,
        progress=progress,
        progress_level=2,
    )

    assert progress.calls == [
        {"total": 4, "desc": "Tiling", "level": 2, "leave": False}
    ]
    assert progress.items == [(0, 0), (0, 100), (100, 0), (100, 100)]
    assert outputs["prediction"].shape == (1, 1, 200, 200)
