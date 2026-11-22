"""Reusable registered-dataset selection and deliberate quick import controls.

No training configuration is created here. A dataset candidate is never written
into a loaded YAML without an explicit review/Apply action.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets

from lisai.application.datasets import (
    TrainingRecording, apply_dataset_change, list_training_datasets,
    preview_dataset_change, training_recording_sources,
)


# A completed import outlives a panel that was hidden/closed during I/O.
_ACTIVE_IMPORTS: set[QtCore.QThread] = set()


class _ImportDataset(QtCore.QThread):
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, settings, sources, name, fmt):
        super().__init__()
        self.settings = settings
        self.sources = sources
        self.name = name
        self.fmt = fmt

    def run(self):
        try:
            from lisai.data import import_training_dataset
            result = import_training_dataset(
                self.sources, settings=self.settings,
                dataset_name=self.name, data_format=self.fmt,
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TrainingDataWidget(QtWidgets.QWidget):
    """Reusable Data section, valid for HDN, CARE and upsampling workflows."""

    datasetSelected = QtCore.Signal(str)
    datasetImported = QtCore.Signal(object)
    documentUpdated = QtCore.Signal()

    def __init__(self, settings, recordings=(), parent=None):
        super().__init__(parent)
        self.settings = settings
        self.recordings = tuple(recordings)
        self.files: list[Path] = []
        self.document = None
        self.worker = None
        main = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "Choose any registered training dataset, or import independent TIFF/array "
            "recordings. Applying a dataset to an existing YAML is a separate action.")
        explanation.setWordWrap(True)
        main.addWidget(explanation)
        self.pages = QtWidgets.QTabWidget(self)
        main.addWidget(self.pages, 1)
        self._build_existing()
        self._build_import()
        self.status = QtWidgets.QLabel(self)
        self.status.setWordWrap(True)
        main.addWidget(self.status)
        self.refresh()

    def _build_existing(self):
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        self.dataset_combo = QtWidgets.QComboBox(page)
        self.refresh_button = QtWidgets.QPushButton("Refresh", page)
        row.addWidget(self.dataset_combo, 1)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        self.data_type_combo = QtWidgets.QComboBox(page)
        form = QtWidgets.QFormLayout()
        form.addRow("Data type:", self.data_type_combo)
        layout.addLayout(form)
        self.details = QtWidgets.QLabel(page)
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.effective = QtWidgets.QLabel(page)
        self.effective.setWordWrap(True)
        layout.addWidget(self.effective)
        self.apply_button = QtWidgets.QPushButton("Apply selected dataset to configuration…", page)
        layout.addWidget(self.apply_button)
        layout.addStretch(1)
        self.pages.addTab(page, "Existing datasets")
        self.refresh_button.clicked.connect(self.refresh)
        self.dataset_combo.currentIndexChanged.connect(self._select_entry)
        self.data_type_combo.currentIndexChanged.connect(self._candidate_changed)
        self.apply_button.clicked.connect(self.apply_to_document)

    def _build_import(self):
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(QtWidgets.QLabel("One file or array is one independent recording (YX or TYX)."))
        self.recording_list = QtWidgets.QListWidget(page)
        self.recording_list.setMaximumHeight(135)
        for index, rec in enumerate(self.recordings):
            item = QtWidgets.QListWidgetItem(f"{rec.name} — {np.asarray(rec.data).shape} ({''.join(rec.axes)})")
            item.setData(QtCore.Qt.UserRole, index)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if rec.selected else QtCore.Qt.Unchecked)
            self.recording_list.addItem(item)
        layout.addWidget(self.recording_list)
        self.file_list = QtWidgets.QListWidget(page)
        self.file_list.setMaximumHeight(100)
        layout.addWidget(self.file_list)
        row = QtWidgets.QHBoxLayout()
        add_files = QtWidgets.QPushButton("Add TIFF files…", page)
        add_folder = QtWidgets.QPushButton("Add TIFF folder…", page)
        remove = QtWidgets.QPushButton("Remove selected", page)
        row.addWidget(add_files)
        row.addWidget(add_folder)
        row.addWidget(remove)
        layout.addLayout(row)
        add_files.clicked.connect(self._choose_files)
        add_folder.clicked.connect(self._choose_folder)
        remove.clicked.connect(self._remove_files)
        form = QtWidgets.QFormLayout()
        self.import_name = QtWidgets.QLineEdit("new_training_dataset", page)
        self.import_format = QtWidgets.QComboBox(page)
        self.import_format.addItem("Timelapses (TYX)", "timelapse")
        self.import_format.addItem("Images (YX)", "single")
        if self.recordings and self.recordings[0].axes == ("Y", "X"):
            self.import_format.setCurrentIndex(1)
        form.addRow("Dataset name:", self.import_name)
        form.addRow("Recording format:", self.import_format)
        layout.addLayout(form)
        self.import_button = QtWidgets.QPushButton("Import into LISAI workspace…", page)
        layout.addWidget(self.import_button)
        layout.addStretch(1)
        self.pages.addTab(page, "Quick import")
        self.import_button.clicked.connect(self.import_selected)

    @property
    def selected_dataset_name(self) -> str | None:
        return self.dataset_combo.currentData() or None

    @property
    def selected_data_type(self) -> str | None:
        return self.data_type_combo.currentData() or None

    def set_document(self, document):
        self.document = document
        effective = document.read_field(("data", "dataset_name")) if document.yaml_text.strip() else None
        if effective:
            index = self.dataset_combo.findData(str(effective))
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        self.effective.setText(
            f"Configuration dataset: {effective or '(not defined)'}"
            " — selection is not applied automatically."
        )
        self.apply_button.setEnabled(document is not None)

    def refresh(self):
        previous = self.selected_dataset_name
        self.entries = {entry.name: entry for entry in list_training_datasets(self.settings)}
        with QtCore.QSignalBlocker(self.dataset_combo):
            self.dataset_combo.clear()
            for item in self.entries.values():
                self.dataset_combo.addItem(item.name, item.name)
            index = self.dataset_combo.findData(previous)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        self._select_entry()

    def _select_entry(self, *_):
        entry = self.entries.get(self.selected_dataset_name)
        with QtCore.QSignalBlocker(self.data_type_combo):
            self.data_type_combo.clear()
            if entry is not None:
                for name in entry.data_types:
                    self.data_type_combo.addItem(name, name)
        if entry is None:
            self.details.setText("No training datasets registered.")
        else:
            self.details.setText(
                f"Format: {entry.data_format or 'unspecified'} · "
                f"Data types: {', '.join(entry.data_types) or 'unspecified'} · "
                f"Roles: {', '.join(entry.available_roles) or 'unspecified'}\n"
                f"Quick HDN: {'eligible' if entry.quick_hdn_eligible else 'not eligible / needs preparation'}\n"
                f"{entry.description or ''}")
        self._candidate_changed()

    def _candidate_changed(self, *_):
        if self.selected_dataset_name:
            self.datasetSelected.emit(self.selected_dataset_name)

    def apply_to_document(self):
        if self.document is None or not self.selected_dataset_name:
            self.status.setText("Open or create a training configuration first.")
            return
        try:
            preview = preview_dataset_change(
                self.document, self.settings, self.selected_dataset_name,
                data_type=self.selected_data_type,
            )
            changes = "\n".join(f"{'.'.join(path)}: {old!r} → {new!r}"
                                for path, old, new in preview.changes)
            warning = "\n".join(preview.warnings)
            if not preview.changes:
                self.status.setText("The configuration already uses these dataset fields.")
                return
            reply = QtWidgets.QMessageBox.question(
                self, "Apply dataset to YAML?",
                f"Only the following fields will change:\n{changes}\n\n{warning}\n\nApply?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            apply_dataset_change(self.document, preview)
            self.set_document(self.document)
            self.documentUpdated.emit()
            self.status.setText("Dataset fields updated in the unsaved configuration. Validate and save.")
        except Exception as exc:
            self.status.setText(f"Cannot apply dataset: {type(exc).__name__}: {exc}")

    def _choose_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose TIFF recordings", "", "TIFF files (*.tif *.tiff)")
        self._add_files(files)

    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose TIFF folder")
        if folder:
            self._add_files(p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in {".tif", ".tiff"})

    def _add_files(self, files):
        for item in files:
            path = Path(item).expanduser().resolve()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and path not in self.files:
                self.files.append(path)
        self._refresh_files()

    def _refresh_files(self):
        self.file_list.clear()
        self.file_list.addItems([str(path) for path in self.files])

    def _remove_files(self):
        unwanted = {item.text() for item in self.file_list.selectedItems()}
        self.files = [path for path in self.files if str(path) not in unwanted]
        self._refresh_files()

    def import_selected(self):
        if self.worker is not None:
            return
        try:
            fmt = str(self.import_format.currentData())
            chosen = [self.recordings[int(self.recording_list.item(i).data(QtCore.Qt.UserRole))]
                      for i in range(self.recording_list.count())
                      if self.recording_list.item(i).checkState() == QtCore.Qt.Checked]
            sources = training_recording_sources(chosen, self.files, fmt)
            name = self.import_name.text().strip()
            if not name:
                raise ValueError("Enter a dataset name.")
        except Exception as exc:
            self.status.setText(str(exc))
            return
        if QtWidgets.QMessageBox.question(
            self, "Import dataset",
            f"Import {len(sources)} independent recording(s) into dataset {name!r}? "
            "The source TIFFs will be copied into the workspace.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        worker = _ImportDataset(self.settings, sources, name, fmt)
        self.worker = worker
        _ACTIVE_IMPORTS.add(worker)
        self.import_button.setEnabled(False)
        self.status.setText("Importing recordings and updating the dataset registry…")
        worker.succeeded.connect(self._import_succeeded)
        worker.failed.connect(self._import_failed)
        worker.finished.connect(lambda w=worker: self._import_finished(w))
        worker.start()

    def _import_succeeded(self, result):
        self.status.setText(f"Imported {result.recording_count} recordings into {result.dataset_name!r}.")
        self.refresh()
        index = self.dataset_combo.findData(result.dataset_name)
        if index >= 0:
            self.dataset_combo.setCurrentIndex(index)
        self.pages.setCurrentIndex(0)
        self.datasetImported.emit(result)

    def _import_failed(self, message):
        self.status.setText(f"Import failed: {message}")

    def _import_finished(self, worker):
        _ACTIVE_IMPORTS.discard(worker)
        if self.worker is worker:
            self.worker = None
            self.import_button.setEnabled(True)
        worker.deleteLater()


__all__ = ["TrainingDataWidget"]
