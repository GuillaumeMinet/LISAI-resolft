"""Reusable training execution page, independent of any specific task."""
from __future__ import annotations

from qtpy import QtCore, QtWidgets


class TrainingExecutionWidget(QtWidgets.QWidget):
    """Visible validation/start/monitor controls; the coordinator owns preflight."""

    validateRequested = QtCore.Signal()
    startRequested = QtCore.Signal()
    resetRequested = QtCore.Signal()

    def __init__(self, session, monitor, parent=None):
        super().__init__(parent)
        main = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Launch a saved, validated LISAI YAML. A new run uses the backend's "
            "automatic numbered directory; it does not rename the configuration.", self)
        intro.setWordWrap(True)
        main.addWidget(intro)
        self.summary = QtWidgets.QLabel("No saved configuration selected.", self)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.summary.setWordWrap(True)
        self.summary.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.summary.setMargin(10)
        main.addWidget(self.summary)
        self.message = QtWidgets.QLabel("Open or create a configuration in Configuration.", self)
        self.message.setWordWrap(True)
        main.addWidget(self.message)
        self.job_state = QtWidgets.QLabel("Local job: idle", self)
        self.job_state.setWordWrap(True)
        main.addWidget(self.job_state)
        buttons = QtWidgets.QHBoxLayout()
        self.validate_button = QtWidgets.QPushButton("Validate configuration", self)
        self.start_button = QtWidgets.QPushButton("Start training", self)
        self.reset_button = QtWidgets.QPushButton("Clear finished session", self)
        self.start_button.setEnabled(False)
        for button in (self.validate_button, self.start_button, self.reset_button):
            buttons.addWidget(button)
        main.addLayout(buttons)
        self.validate_button.clicked.connect(self.validateRequested.emit)
        self.start_button.clicked.connect(self.startRequested.emit)
        self.reset_button.clicked.connect(self.resetRequested.emit)
        main.addWidget(monitor, 1)
        self.session = session

    def set_document_summary(self, *, source, experiment, dataset, task, mode, dirty):
        lines = [f"Configuration: {source or '(unsaved draft)'}",
                 f"Experiment: {experiment or '(unset)'}  ·  Dataset: {dataset or '(unset)'}",
                 f"Task: {task or '(unset)'}  ·  Mode: {mode or '(unset)'}"]
        if dirty:
            lines.append("Unsaved changes: save the YAML before launching.")
        self.summary.setText("\n".join(lines))

    def set_job_snapshot(self, snapshot):
        # Do not invent a percentage: local QProcess has no structured epoch
        # progress. A later runner backend may provide a real value.
        progress = (" · progress {:.0%}".format(snapshot.progress)
                    if snapshot.progress is not None else "")
        self.job_state.setText(f"Local job: {snapshot.status}{progress} · {snapshot.message}")


__all__ = ["TrainingExecutionWidget"]
