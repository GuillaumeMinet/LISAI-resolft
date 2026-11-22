"""Reusable grayscale YX image viewer; no model, TIFF, or host dependencies."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets


def _grayscale_yx(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 2 or not all(n > 0 for n in value.shape):
        raise ValueError(f"ImageViewer2D requires a nonempty YX image, got {value.shape}")
    if value.dtype.kind not in "uif":
        raise TypeError(f"ImageViewer2D requires real numeric pixels, got {value.dtype}")
    return value


def _pixel_spacing(pixel_size) -> tuple[float, float] | None:
    if pixel_size is None:
        return None
    if np.isscalar(pixel_size):
        values = (float(pixel_size), float(pixel_size))
    else:
        try:
            values = tuple(float(v) for v in pixel_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("pixel_size must be a positive number or (y, x) pair") from exc
    if len(values) != 2 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("pixel_size must be a positive finite number or (y, x) pair")
    return values


def _auto_levels(image: np.ndarray) -> tuple[float, float]:
    """Sample the image for fast display-only robust percentiles."""
    # np.ravel is usually a view for the row-major TIFF slices used here.
    sample = np.ravel(image)
    sample = sample[::max(1, sample.size // 150_000)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("Image contains no finite pixels to display")
    low, high = (float(v) for v in np.percentile(sample, (0.5, 99.5)))
    if low == high:
        radius = max(abs(low) * 0.01, 1e-6)
        low -= radius
        high += radius
    return low, high


class ImageViewer2D(QtWidgets.QWidget):
    """One grayscale image, independent contrast and independently panned view.

    ``set_image`` and display-level operations never alter the supplied array.
    Pyqtgraph uses explicit row-major axes (image[y, x]) and a downward Y axis
    so the top-left pixel is displayed in the top-left of the canvas.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._pixel_size: tuple[float, float] | None = None
        self._levels: tuple[float, float] | None = None
        self._update_controls = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = QtWidgets.QStackedWidget(self)
        self.empty_label = QtWidgets.QLabel("No image", self.canvas)
        self.empty_label.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.canvas.addWidget(self.empty_label)

        self.plot = pg.PlotWidget(self.canvas)
        self.plot.setBackground((18, 20, 24))
        self.plot.setMenuEnabled(False)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.plot.setMouseEnabled(x=True, y=True)
        self.view_box = self.plot.getViewBox()
        self.view_box.setAspectLocked(True)
        self.view_box.invertY(True)
        self.image_item = pg.ImageItem(axisOrder="row-major", autoDownsample=True)
        self.plot.addItem(self.image_item)
        self.canvas.addWidget(self.plot)
        layout.addWidget(self.canvas, 1)

        controls = QtWidgets.QHBoxLayout()
        self.auto_checkbox = QtWidgets.QCheckBox("Auto contrast", self)
        self.auto_checkbox.setChecked(True)
        controls.addWidget(self.auto_checkbox)
        controls.addWidget(QtWidgets.QLabel("Min", self))
        self.low_edit = QtWidgets.QLineEdit(self)
        self.low_edit.setMaximumWidth(95)
        controls.addWidget(self.low_edit)
        controls.addWidget(QtWidgets.QLabel("Max", self))
        self.high_edit = QtWidgets.QLineEdit(self)
        self.high_edit.setMaximumWidth(95)
        controls.addWidget(self.high_edit)
        self.fit_button = QtWidgets.QPushButton("Fit", self)
        self.fit_button.setToolTip("Reset zoom to fit this image")
        controls.addWidget(self.fit_button)
        layout.addLayout(controls)
        self.auto_checkbox.toggled.connect(self._auto_toggled)
        self.low_edit.editingFinished.connect(self._apply_manual_edits)
        self.high_edit.editingFinished.connect(self._apply_manual_edits)
        self.fit_button.clicked.connect(self.reset_view)
        self._sync_controls()

    @property
    def image(self) -> np.ndarray | None:
        return self._image

    @property
    def levels(self) -> tuple[float, float] | None:
        return self._levels

    @property
    def pixel_size(self) -> tuple[float, float] | None:
        return self._pixel_size

    def set_image(self, image_yx, *, pixel_size=None, label=None):
        data = _grayscale_yx(image_yx)
        spacing = _pixel_spacing(pixel_size)
        first = self._image is None
        old_shape = None if first else self._image.shape
        old_spacing = self._pixel_size
        # Validate the image before changing current display or its controls.
        computed_levels = _auto_levels(data) if self.auto_checkbox.isChecked() else self._levels
        if computed_levels is None:
            computed_levels = _auto_levels(data)
        self._image = data
        self._pixel_size = spacing
        self._levels = computed_levels
        self.image_item.setImage(data, autoLevels=False, levels=list(computed_levels))
        # Only meaningful if the host supplied real calibrated spacing.
        y_step, x_step = spacing or (1.0, 1.0)
        self.image_item.setRect(QtCore.QRectF(0, 0, data.shape[1] * x_step,
                                               data.shape[0] * y_step))
        self.canvas.setCurrentWidget(self.plot)
        self._sync_controls()
        if first or old_shape != data.shape or old_spacing != spacing:
            self.reset_view()
        if label is not None:
            self.setToolTip(str(label))

    def clear(self, message: str = "No image", *, preserve_levels: bool = False):
        self._image = None
        self._pixel_size = None
        if not preserve_levels:
            self._levels = None
            self._update_controls = True
            try:
                self.auto_checkbox.setChecked(True)
            finally:
                self._update_controls = False
        # Hide the previous array without creating a dummy frame.
        self.image_item.clear()
        self.empty_label.setText(message)
        self.canvas.setCurrentWidget(self.empty_label)
        self._sync_controls()

    def set_levels(self, low: float, high: float):
        low, high = float(low), float(high)
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError("Display levels require finite min < max")
        self._levels = (low, high)
        self._update_controls = True
        try:
            self.auto_checkbox.setChecked(False)
        finally:
            self._update_controls = False
        if self._image is not None:
            self.image_item.setLevels([low, high])
        self._sync_controls()

    def reset_view(self, *_):
        if self._image is not None:
            self.view_box.autoRange(padding=0.02)

    def _auto_toggled(self, checked):
        if self._update_controls:
            return
        if checked and self._image is not None:
            self._levels = _auto_levels(self._image)
            self.image_item.setLevels(list(self._levels))
        self._sync_controls()

    def _apply_manual_edits(self):
        if self._image is None or self.auto_checkbox.isChecked():
            return
        try:
            self.set_levels(float(self.low_edit.text()), float(self.high_edit.text()))
        except ValueError:
            # Keep the previously valid display range; do not affect image values.
            self._sync_controls()

    def _sync_controls(self):
        enabled = self._image is not None
        self.auto_checkbox.setEnabled(enabled)
        editable = enabled and not self.auto_checkbox.isChecked()
        self.low_edit.setEnabled(editable)
        self.high_edit.setEnabled(editable)
        self.fit_button.setEnabled(enabled)
        if self._levels is not None:
            self.low_edit.setText(f"{self._levels[0]:.7g}")
            self.high_edit.setText(f"{self._levels[1]:.7g}")
        else:
            self.low_edit.clear()
            self.high_edit.clear()
