"""Reusable Qt training output and execution controls (all LISAI training tasks)."""
from __future__ import annotations

from qtpy import QtCore, QtWidgets


from lisai.qt.training.monitor import TrainingMonitorWidget

# R4-C: preserve the R3 import path while moving orchestration into one
# host-independent TrainingWorkspaceWidget. No second training implementation.
from lisai.qt.training.workspace import TrainingWorkspaceWidget
from lisai.application.datasets import TrainingRecording
from lisai.qt.training_session import get_training_session

TrainingPanel = TrainingWorkspaceWidget  # compatibility for R3 integrations


class TrainingDialog(QtWidgets.QDialog):
    """Modeless wrapper; hiding does not stop the shared local training session."""

    def __init__(self, settings, recordings=(), *, session=None, on_completed=None):
        super().__init__(None)
        from pathlib import Path
        self.settings_identity = (
            Path(settings.WORKSPACE_ROOT).resolve(),
            Path(settings.CONFIGS_ROOT).resolve(),
        )
        self.setWindowTitle("LISAI — Training")
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.resize(1000, 810)
        layout = QtWidgets.QVBoxLayout(self)
        self.panel = TrainingWorkspaceWidget(
            settings, recordings, self, session=session, on_completed=on_completed,
        )
        layout.addWidget(self.panel)
        hide_button = QtWidgets.QPushButton("Hide window", self)
        hide_button.clicked.connect(self.hide)
        layout.addWidget(hide_button)

    def reject(self):
        self.hide()

    def accept(self):
        self.hide()

    def closeEvent(self, event):
        self.hide()
        event.accept()


def show_training_dialog(settings, recordings=(), *, session=None, on_completed=None):
    """Reopen the same-workspace dialog; never silently attach a new workspace."""
    from pathlib import Path

    session = session or get_training_session()
    requested = (Path(settings.WORKSPACE_ROOT).resolve(), Path(settings.CONFIGS_ROOT).resolve())
    dialog = session.dialog
    if dialog is not None and dialog.settings_identity == requested and not (
        session.is_active() and session.workspace is not None and
        (Path(session.workspace.WORKSPACE_ROOT).resolve(),
         Path(session.workspace.CONFIGS_ROOT).resolve()) != requested
    ):
        session.show_window()
        return dialog
    if session.is_active():
        original = (Path(session.workspace.WORKSPACE_ROOT).resolve()
                    if session.workspace is not None else "unknown")
        raise RuntimeError(
            f"Training is already running in workspace {original}. "
            "Stop or finish that job before opening another workspace. "
            "The original training window is still available."
        )
    if dialog is not None:
        dialog.hide()
        session.dialog = None
        dialog.deleteLater()
    dialog = TrainingDialog(settings, recordings, session=session, on_completed=on_completed)
    session.dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


__all__ = [
    "TrainingMonitorWidget", "TrainingRecording", "TrainingWorkspaceWidget",
    "TrainingPanel", "TrainingDialog", "show_training_dialog",
]
