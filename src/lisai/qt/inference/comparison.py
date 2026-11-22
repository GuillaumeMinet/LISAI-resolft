"""Reusable YX/TYX comparison widget with input-frame-centric alignment.

This component displays images only. It never loads TIFFs, invokes a model,
selects a workspace, or infers spatial registration from array dimensions.
"""
from __future__ import annotations

import numpy as np
from qtpy import QtCore, QtWidgets

from lisai.evaluation.inference.results import PredictionInfo
from .alignment import FrameAlignment
from .image_viewer import ImageViewer2D


def _image(data, *, axes: str, name: str) -> np.ndarray:
    result = np.asarray(data)
    if axes not in ("YX", "TYX") or result.ndim != len(axes):
        raise ValueError(f"{name} must be YX or TYX; got axes={axes!r}, shape={result.shape}")
    if any(size < 1 for size in result.shape) or result.dtype.kind not in "uif":
        raise ValueError(f"{name} must be a nonempty real numeric grayscale image")
    return result


class ImageComparisonWidget(QtWidgets.QWidget):
    """Independent image views with navigation over the **input** time axis."""

    frameChanged = QtCore.Signal(int)
    inputChanged = QtCore.Signal()
    predictionCleared = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw: np.ndarray | None = None
        self._axes: str | None = None
        self._prediction: np.ndarray | None = None
        self._alignment: FrameAlignment | None = None
        self._raw_pixel_size = None
        self._pred_pixel_size = None
        self._frame = 0
        layout = QtWidgets.QVBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.input_viewer = ImageViewer2D(self)
        self.prediction_viewer = ImageViewer2D(self)
        for name, viewer in (("Input", self.input_viewer),
                             ("Prediction", self.prediction_viewer)):
            panel = QtWidgets.QWidget(self.splitter)
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(2, 2, 2, 2)
            panel_layout.addWidget(QtWidgets.QLabel(name, panel))
            panel_layout.addWidget(viewer, 1)
            self.splitter.addWidget(panel)
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, 1)

        self.navigation = QtWidgets.QWidget(self)
        nav = QtWidgets.QHBoxLayout(self.navigation)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.addWidget(QtWidgets.QLabel("Input frame", self.navigation))
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.navigation)
        self.frame_slider.setRange(0, 0)
        nav.addWidget(self.frame_slider, 1)
        self.frame_spin = QtWidgets.QSpinBox(self.navigation)
        self.frame_spin.setRange(1, 1)
        nav.addWidget(self.frame_spin)
        self.frame_count = QtWidgets.QLabel("/ 1", self.navigation)
        nav.addWidget(self.frame_count)
        layout.addWidget(self.navigation)
        self.prediction_status = QtWidgets.QLabel("No prediction", self)
        self.prediction_status.setWordWrap(True)
        layout.addWidget(self.prediction_status)
        self.frame_slider.valueChanged.connect(self.set_frame_index)
        self.frame_spin.valueChanged.connect(lambda value: self.set_frame_index(value - 1))
        self.navigation.hide()

    @property
    def current_frame_index(self) -> int:
        return self._frame

    @property
    def axes(self) -> str | None:
        return self._axes

    @property
    def has_prediction(self) -> bool:
        return self._prediction is not None

    @property
    def alignment(self) -> FrameAlignment | None:
        return self._alignment

    def current_prediction_index(self) -> int | None:
        if self._alignment is None:
            return None
        return self._alignment.prediction_index(self._frame)

    def set_input(self, raw, *, axes: str, pixel_size=None):
        array = _image(raw, axes=axes, name="Input")
        # Validate the first frame and supplied pixel size before replacing the
        # current input. Nothing is inferred from output shape or filename.
        from .image_viewer import _auto_levels, _pixel_spacing
        _pixel_spacing(pixel_size)
        _auto_levels(array if axes == "YX" else array[0])
        self._raw = array
        self._axes = axes
        self._raw_pixel_size = pixel_size
        self._frame = 0
        self.clear_prediction()
        length = self._input_frames()
        with QtCore.QSignalBlocker(self.frame_slider), QtCore.QSignalBlocker(self.frame_spin):
            self.frame_slider.setRange(0, length - 1)
            self.frame_spin.setRange(1, length)
            self.frame_slider.setValue(0)
            self.frame_spin.setValue(1)
        self.frame_count.setText(f"/ {length}")
        self.navigation.setVisible(axes == "TYX")
        self.input_viewer.clear()
        self._render_frame()
        self.inputChanged.emit()

    def set_prediction(self, pred, *, info: PredictionInfo, spatial_scale=None):
        if self._raw is None or self._axes is None:
            raise RuntimeError("Set an input image before its prediction")
        array = _image(pred, axes=self._axes, name="Prediction")
        output_frames = array.shape[0] if self._axes == "TYX" else 1
        mapping = FrameAlignment.from_info(
            info, input_frames=self._input_frames(), output_frames=output_frames,
        )
        if self._axes == "YX" and (mapping.source_to_output != (0,) or
                                    mapping.black_output_indices):
            raise ValueError("YX prediction must align to source frame 0 without placeholders")
        from .image_viewer import _pixel_spacing, _auto_levels
        _pixel_spacing(spatial_scale)
        _auto_levels(array if self._axes == "YX" else array[0])
        self._prediction = array
        self._alignment = mapping
        self._pred_pixel_size = spatial_scale
        self.prediction_viewer.clear()
        self._render_frame()

    def clear_prediction(self, reason: str = "No prediction"):
        had_prediction = self._prediction is not None
        self._prediction = None
        self._alignment = None
        self._pred_pixel_size = None
        self.prediction_viewer.clear(message=reason)
        self.prediction_status.setText(reason)
        if had_prediction:
            self.predictionCleared.emit()

    def clear(self):
        self._raw = None
        self._axes = None
        self._frame = 0
        self._raw_pixel_size = None
        self.input_viewer.clear()
        self.clear_prediction()
        self.navigation.hide()
        self.inputChanged.emit()

    def invalidate_prediction(self, reason: str = "Prediction outdated — run inference again"):
        """Host calls this when model/options/workspace/input selection changes."""
        self.clear_prediction(reason)

    def set_frame_index(self, index: int):
        if self._raw is None:
            raise RuntimeError("Open an image before selecting a frame")
        if type(index) is not int or not (0 <= index < self._input_frames()):
            raise ValueError("Input frame index is out of bounds")
        if index == self._frame:
            return
        self._frame = index
        with QtCore.QSignalBlocker(self.frame_slider), QtCore.QSignalBlocker(self.frame_spin):
            self.frame_slider.setValue(index)
            self.frame_spin.setValue(index + 1)
        self._render_frame()
        self.frameChanged.emit(index)

    def _input_frames(self) -> int:
        if self._raw is None:
            return 0
        return self._raw.shape[0] if self._axes == "TYX" else 1

    def _render_frame(self):
        if self._raw is None:
            return
        raw_frame = self._raw[self._frame] if self._axes == "TYX" else self._raw
        self.input_viewer.set_image(raw_frame, pixel_size=self._raw_pixel_size)
        prediction_index = self.current_prediction_index()
        if prediction_index is None:
            message = ("No prediction for this input frame" if self._alignment is not None
                       else "No prediction")
            self.prediction_viewer.clear(message, preserve_levels=True)
            self.prediction_status.setText(message)
            return
        assert self._prediction is not None and self._alignment is not None
        predicted_frame = (self._prediction[prediction_index]
                           if self._axes == "TYX" else self._prediction)
        self.prediction_viewer.set_image(predicted_frame, pixel_size=self._pred_pixel_size)
        if self._alignment.is_black_placeholder(self._frame):
            self.prediction_status.setText(
                f"Input frame {self._frame + 1}: black output placeholder "
                f"(prediction position {prediction_index + 1})"
            )
        else:
            self.prediction_status.setText(
                f"Input frame {self._frame + 1} → prediction {prediction_index + 1}"
            )
