"""Qt thread boundary for LISAI's existing public in-memory inference API.

No TIFF browsing, model selection, caching, or alternate inference engine lives here.
One worker owns one immutable selection/input/settings snapshot and never touches
widgets. A hosting page owns worker lifetime until ``finished``.
"""
from __future__ import annotations

import numpy as np
from qtpy import QtCore

from lisai.application.inference import InferenceSelection
from lisai.config.settings import Settings


class InferenceWorker(QtCore.QThread):
    """Run a single array prediction without blocking Qt's event loop.

    ``generation`` identifies the host's request, so an obsolete result can be
    discarded even if a caller changes input/settings while a worker is active.
    """

    succeeded = QtCore.Signal(int, object, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, *, generation: int, settings: Settings,
                 selection: InferenceSelection, data: np.ndarray):
        super().__init__()  # parentless: the host retains this thread until finished
        if not isinstance(settings, Settings):
            raise TypeError("InferenceWorker requires explicit LISAI Settings")
        if not isinstance(selection, InferenceSelection):
            raise TypeError("InferenceWorker requires a validated InferenceSelection")
        if not isinstance(data, np.ndarray):
            raise TypeError("InferenceWorker requires a NumPy array")
        self.generation = generation
        self.settings = settings
        self.selection = selection
        self.data = data  # strong reference; do not modify the caller's array

    def run(self):
        try:
            from lisai import predict

            output, info = predict(
                self.selection.model, self.data, settings=self.settings,
                overrides=self.selection.overrides, device="auto", return_info=True,
            )
            self.succeeded.emit(self.generation, output, info)
        except Exception as exc:  # only report; no implicit retry or fallback
            self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")


__all__ = ["InferenceWorker"]
