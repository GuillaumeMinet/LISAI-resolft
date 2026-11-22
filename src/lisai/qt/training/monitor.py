"""Reusable training progress monitor; Qt-only view of TrainingSession."""
from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets

class TrainingMonitorWidget(QtWidgets.QGroupBox):
    """A view of TrainingSession; hiding/destroying it does not stop training."""

    def __init__(self, session, parent=None):
        super().__init__("Training progress", parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(2500)
        self.log.setPlaceholderText("Training output appears here after Start…")
        self.log.setMinimumHeight(100)
        layout.addWidget(self.log)
        row = QtWidgets.QHBoxLayout()
        self.stop_button = QtWidgets.QPushButton("Stop training", self)
        self.force_button = QtWidgets.QPushButton("Force stop…", self)
        self.folder_button = QtWidgets.QPushButton("Open run folder", self)
        for button in (self.stop_button, self.force_button, self.folder_button):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.stop_button.clicked.connect(self._request_stop)
        self.force_button.clicked.connect(self._force_stop)
        self.folder_button.clicked.connect(self._open_run_folder)
        session.log_appended.connect(self.append_log)
        session.changed.connect(self.refresh)
        session.started_new.connect(self.log.clear)
        if session.log_text:
            self.log.setPlainText(session.log_text)
        self.refresh()

    def append_log(self, output):
        if output:
            self.log.insertPlainText(output)
            scrollbar = self.log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def refresh(self):
        active = self.session.is_active()
        if not self.session.has_started and not self.session.log_text and self.log.toPlainText():
            self.log.clear()
        self.stop_button.setEnabled(active and not self.session.stop_requested)
        self.force_button.setEnabled(active and self.session.stop_requested)
        path = self.session.run_dir
        self.folder_button.setEnabled(path is not None and path.is_dir())

    def _request_stop(self):
        if self.session.is_active() and QtWidgets.QMessageBox.question(
            self, "Stop training",
            "Request that the training process stop? An incomplete checkpoint "
            "or run may remain. The dataset and training configuration are retained.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) == QtWidgets.QMessageBox.Yes:
            self.session.request_stop()

    def _force_stop(self):
        if self.session.is_active() and QtWidgets.QMessageBox.question(
            self, "Force-stop training",
            "Forcefully terminate the training process? The active checkpoint may be incomplete.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) == QtWidgets.QMessageBox.Yes:
            self.session.request_stop(force=True)

    def _open_run_folder(self):
        directory = self.session.run_dir
        if directory is not None and directory.is_dir():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory)))






__all__ = ["TrainingMonitorWidget"]
