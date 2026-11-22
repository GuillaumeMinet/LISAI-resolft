"""APP-3 viewer-only smoke test; requires PySide6/PyQt plus lisai[qt].

Run: python scripts/lisai_image_comparison_qt_demo.py
No LISAI workspace, ImProcess, TIFF, or trained model is required.
"""
from __future__ import annotations

import sys

import numpy as np
from qtpy import QtWidgets

from lisai.evaluation.inference.results import PredictionInfo
from lisai.qt.inference import ImageComparisonWidget


def synthetic_images():
    yy, xx = np.mgrid[:250, :340]
    rng = np.random.default_rng(7)
    raw = []
    for t in range(12):
        image = np.zeros_like(xx, dtype=np.float32)
        for x, y, width in ((100 + 5 * t, 90, 6), (220, 175 - 3 * t, 10), (260, 60, 4)):
            image += np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * width**2))
        raw.append(image + rng.normal(0, 0.13, image.shape).astype(np.float32))
    raw = np.stack(raw)
    filtered = np.stack([
        np.maximum(raw[t], 0) for t in range(2, 10)
    ])
    return raw, filtered


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    window.setWindowTitle("LISAI APP-3 — Image comparison (synthetic data)")
    layout = QtWidgets.QVBoxLayout(window)
    comparison = ImageComparisonWidget(window)
    layout.addWidget(comparison, 1)
    raw, filtered = synthetic_images()
    comparison.set_input(raw, axes="TYX")
    comparison.set_prediction(filtered, info=PredictionInfo(tuple(range(2, 10))))

    buttons = QtWidgets.QHBoxLayout()
    drop = QtWidgets.QPushButton("Drop edges")
    black = QtWidgets.QPushButton("Black output edges")
    clear = QtWidgets.QPushButton("Clear prediction")
    buttons.addWidget(drop)
    buttons.addWidget(black)
    buttons.addWidget(clear)
    layout.addLayout(buttons)
    drop.clicked.connect(lambda: comparison.set_prediction(
        filtered, info=PredictionInfo(tuple(range(2, 10)))))

    def black_mode():
        result = np.zeros_like(raw)
        result[2:10] = filtered
        comparison.set_prediction(
            result, info=PredictionInfo(tuple(range(12)), (0, 1, 10, 11)),
        )

    black.clicked.connect(black_mode)
    clear.clicked.connect(comparison.clear_prediction)
    comparison.set_frame_index(2)
    window.resize(1120, 630)
    window.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
