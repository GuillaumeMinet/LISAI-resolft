from __future__ import annotations

from lisai.evaluation.inference.progress import InferenceProgress


def test_inference_progress_honors_disable_tqdm_env(monkeypatch):
    monkeypatch.setenv("LISAI_DISABLE_TQDM", "1")

    progress = InferenceProgress()

    assert progress.enabled is False
    assert list(progress.track(range(3), total=3, desc="Hidden")) == [0, 1, 2]
