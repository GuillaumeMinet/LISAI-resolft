"""Standalone TIFF -> predict -> compare -> Save composition.

File dialogs and Run/Save actions belong to this host, not to reusable model
controls or viewers. ImProcess can reuse those widgets without importing here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets

from lisai.application.inference_io import (
    InferenceProvenance,
    TiffInputInfo,
    load_inference_tiff,
    save_prediction_tiff,
)
from lisai.config.settings import Settings
from lisai.evaluation.inference.results import PredictionInfo
from lisai.qt.inference.comparison import ImageComparisonWidget
from lisai.qt.inference.controls import InferenceControlsWidget
from lisai.qt.inference.worker import InferenceWorker
from lisai.qt.model_manager import ModelManagerWidget


class StandaloneInferencePage(QtWidgets.QWidget):
    """App-specific composition of reusable model controls and grayscale viewer."""

    busyChanged = QtCore.Signal(bool)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._input: np.ndarray | None = None
        self._input_info: TiffInputInfo | None = None
        self._prediction: np.ndarray | None = None
        self._prediction_info: PredictionInfo | None = None
        self._provenance: InferenceProvenance | None = None
        self._worker: InferenceWorker | None = None
        self._generation = 0

        layout = QtWidgets.QVBoxLayout(self)
        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack, 1)

        main = QtWidgets.QWidget(self.stack)
        main_layout = QtWidgets.QVBoxLayout(main)
        self.stack.addWidget(main)
        toolbar = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open TIFF…", main)
        self.open_button.clicked.connect(self.open_tiff)
        toolbar.addWidget(self.open_button)
        self.source_label = QtWidgets.QLabel("No image loaded", main)
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        toolbar.addWidget(self.source_label, 1)
        self.axes_label = QtWidgets.QLabel("", main)
        toolbar.addWidget(self.axes_label)
        main_layout.addLayout(toolbar)

        self.comparison = ImageComparisonWidget(main)
        main_layout.addWidget(self.comparison, 1)

        settings_group = QtWidgets.QGroupBox("Model and inference settings", main)
        settings_layout = QtWidgets.QVBoxLayout(settings_group)
        self.controls = InferenceControlsWidget(settings, settings_group)
        settings_layout.addWidget(self.controls)
        settings_scroll = QtWidgets.QScrollArea(main)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        settings_scroll.setWidget(settings_group)
        settings_scroll.setMaximumHeight(320)
        main_layout.addWidget(settings_scroll)

        bottom = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Open a TIFF image and choose a model.", main)
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.save_button = QtWidgets.QPushButton("Save prediction…", main)
        self.save_button.clicked.connect(self.save_prediction)
        bottom.addWidget(self.save_button)
        self.run_button = QtWidgets.QPushButton("Run inference", main)
        self.run_button.clicked.connect(self.run_inference)
        bottom.addWidget(self.run_button)
        main_layout.addLayout(bottom)

        manager_page = QtWidgets.QWidget(self.stack)
        manager_layout = QtWidgets.QVBoxLayout(manager_page)
        self.model_manager = ModelManagerWidget(settings, manager_page)
        manager_layout.addWidget(self.model_manager, 1)
        manager_actions = QtWidgets.QHBoxLayout()
        self.use_model_button = QtWidgets.QPushButton("Use selected installed model", manager_page)
        self.use_model_button.clicked.connect(self._use_manager_model)
        manager_actions.addStretch(1)
        manager_actions.addWidget(self.use_model_button)
        manager_layout.addLayout(manager_actions)
        self.stack.addWidget(manager_page)

        self.controls.selectionChanged.connect(self._invalidate_result)
        self.controls.optionsChanged.connect(self._invalidate_result)
        self.controls.manageModelsRequested.connect(self.show_model_manager)
        self.model_manager.backRequested.connect(self.show_inference)
        self.model_manager.modelsChanged.connect(self.controls.refresh_catalog)
        self.model_manager.modelInstalled.connect(self.controls.refresh_catalog)
        self._sync_buttons()

    def is_busy(self) -> bool:
        return (self._worker is not None or self.model_manager.is_busy())

    def _sync_buttons(self):
        running = self._worker is not None
        self.open_button.setEnabled(not running)
        self.run_button.setEnabled(not running and self._input is not None)
        self.controls.setEnabled(not running)
        self.save_button.setEnabled(not running and self._prediction is not None)
        self.use_model_button.setEnabled(not running)

    def _invalidate_result(self, *_):
        self._generation += 1
        self._prediction = None
        self._prediction_info = None
        self._provenance = None
        self.comparison.invalidate_prediction()
        self._sync_buttons()

    def load_tiff(self, path: str | Path) -> TiffInputInfo:
        """Validate before replacing the old image; never infer Z as time."""
        if self._worker is not None:
            raise RuntimeError("Cannot change the input while inference is running.")
        image, info = load_inference_tiff(path)
        self.comparison.set_input(image, axes=info.axes)
        self._input = image
        self._input_info = info
        self._generation += 1
        self._prediction = None
        self._prediction_info = None
        self._provenance = None
        self.source_label.setText(info.path.name)
        self.source_label.setToolTip(str(info.path))
        self.axes_label.setText(f"{info.axes} · {' × '.join(map(str, info.shape))}")
        self.status.setText("Image loaded. Select a model, then Run inference.")
        self._sync_buttons()
        return info

    def open_tiff(self, *_):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open LISAI input", "", "TIFF images (*.tif *.tiff)"
        )
        if not filename:
            return
        try:
            self.load_tiff(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open TIFF", str(exc))
            self.status.setText(f"Cannot open TIFF: {exc}")

    def run_inference(self, *_):
        """Freeze one model/input snapshot and launch exactly one prediction."""
        if self._worker is not None:
            return
        if self._input is None or self._input_info is None:
            self.status.setText("Open a TIFF input first.")
            return
        try:
            selection = self.controls.inference_selection()
        except Exception as exc:
            self.status.setText(f"Cannot start inference: {exc}")
            return
        self._invalidate_result()
        request_id = self._generation
        self._provenance = InferenceProvenance(
            source_path=self._input_info.path, model=selection.model,
            overrides=selection.overrides,
        )
        worker = InferenceWorker(
            generation=request_id, settings=self.settings,
            selection=selection, data=self._input,
        )
        self._worker = worker
        worker.succeeded.connect(self._prediction_succeeded)
        worker.failed.connect(self._prediction_failed)
        worker.finished.connect(lambda w=worker: self._prediction_finished(w))
        self.status.setText("Running inference… (no progress percentage available)")
        self._sync_buttons()
        self.busyChanged.emit(True)
        worker.start()

    def _prediction_succeeded(self, generation: int, prediction, info):
        if generation != self._generation or self._input_info is None:
            return  # stale request must never replace the currently selected input
        try:
            if not isinstance(info, PredictionInfo):
                raise TypeError("Inference returned invalid temporal alignment metadata")
            array = np.asarray(prediction)
            self.comparison.set_prediction(array, info=info)
        except Exception as exc:
            self._prediction = None
            self._prediction_info = None
            self._provenance = None
            self.comparison.clear_prediction("Cannot display inference result")
            self.status.setText(f"Prediction returned an unsupported result: {exc}")
            return
        self._prediction = array
        self._prediction_info = info
        self.status.setText("Inference complete. Review the prediction, then Save if needed.")

    def _prediction_failed(self, generation: int, message: str):
        if generation != self._generation:
            return
        self._prediction = None
        self._prediction_info = None
        self._provenance = None
        self.comparison.clear_prediction("Inference failed")
        self.status.setText(f"Inference failed: {message}")

    def _prediction_finished(self, worker: InferenceWorker):
        if self._worker is worker:
            self._worker = None
        self._sync_buttons()
        self.busyChanged.emit(False)
        worker.deleteLater()

    def save_prediction_to(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the current result, never run the neural network again."""
        if self._worker is not None:
            raise RuntimeError("Inference is still running.")
        if (self._prediction is None or self._prediction_info is None or
                self._provenance is None or self._input_info is None):
            raise RuntimeError("There is no current prediction to save.")
        return save_prediction_tiff(
            path, self._prediction, axes=self._input_info.axes,
            info=self._prediction_info, provenance=self._provenance,
            overwrite=overwrite,
        )

    def save_prediction(self, *_):
        if self._input_info is None or self._prediction is None:
            return
        proposed = self._input_info.path.with_name(self._input_info.path.stem + "_lisai.tif")
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save prediction", str(proposed), "TIFF images (*.tif *.tiff)"
        )
        if not filename:
            return
        try:
            saved = self.save_prediction_to(filename)
        except FileExistsError:
            if QtWidgets.QMessageBox.question(
                self, "Overwrite existing TIFF?", f"Overwrite {filename}?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            ) != QtWidgets.QMessageBox.Yes:
                return
            try:
                saved = self.save_prediction_to(filename, overwrite=True)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
                return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.status.setText(f"Saved prediction: {saved}")

    def show_model_manager(self):
        if self._worker is not None:
            return
        self.model_manager.refresh_local()
        self.stack.setCurrentWidget(self.model_manager.parentWidget())

    def show_inference(self):
        # Model changes refresh the controls via modelsChanged/modelInstalled.
        # An unconditional refresh here emits selectionChanged and would
        # incorrectly discard a valid prediction merely by browsing the catalog.
        self.stack.setCurrentIndex(0)

    def _use_manager_model(self):
        name = self.model_manager.selected_model_name()
        if not name or not self.model_manager.is_installed(name):
            self.model_manager.status.setText("Install the selected model before using it.")
            return
        self.controls.select_model(name)
        self.show_inference()

    def set_settings(self, settings: Settings):
        """Change workspace only while no inference/download is running."""
        if self.is_busy():
            raise RuntimeError("Cannot switch workspace during inference or model management.")
        if not isinstance(settings, Settings):
            raise TypeError("A LISAI Settings instance is required")
        self.settings = settings
        self._input = None
        self._input_info = None
        self._prediction = None
        self._prediction_info = None
        self._provenance = None
        self._generation += 1
        self.comparison.clear()
        self.source_label.setText("No image loaded")
        self.axes_label.clear()
        self.controls.set_settings(settings)
        self.model_manager.set_settings(settings)
        self.stack.setCurrentIndex(0)
        self.status.setText("Workspace changed. Open a TIFF and select a model.")
        self._sync_buttons()


__all__ = ["StandaloneInferencePage"]
