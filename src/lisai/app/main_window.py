"""Standalone two-tab LISAI shell; scientific UI remains in lisai.qt.

A workspace switch reconstructs host-owned widgets instead of mutating the
settings of an active training or inference operation.
"""
from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from lisai.application.workspace import create_workspace, validate_workspace
from lisai.config.settings import Settings
from lisai.qt.training.workspace import TrainingWorkspaceWidget
from lisai.qt.training_session import get_training_session

from .inference_page import StandaloneInferencePage


def load_app_settings(workspace: str | Path, *, config_root: str | Path | None = None) -> Settings:
    """Explicit workspace selection, independent of developer CLI selection."""
    validated = validate_workspace(workspace)
    if config_root is None:
        return validated
    return Settings.from_workspace(Path(workspace).expanduser().resolve(),
                                   config_root=Path(config_root).expanduser().resolve())


class WorkspaceSelectionDialog(QtWidgets.QDialog):
    """Explicit first-use workspace chooser; never invents a default workspace."""

    def __init__(self, parent=None, *, config_root=None):
        super().__init__(parent)
        self.setWindowTitle("Select LISAI workspace")
        self.resize(650, 160)
        self.config_root = config_root
        self.settings = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Select an initialized LISAI workspace. This does not change LISAI CLI settings.", self
        ))
        row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(self)
        self.path_edit.setPlaceholderText("Absolute path to LISAI workspace")
        row.addWidget(self.path_edit, 1)
        browse = QtWidgets.QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        layout.addLayout(row)
        self.message = QtWidgets.QLabel("", self)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        actions = QtWidgets.QHBoxLayout()
        create = QtWidgets.QPushButton("Create new workspace…", self)
        create.clicked.connect(self._create)
        actions.addWidget(create)
        actions.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        open_button = QtWidgets.QPushButton("Open workspace", self)
        open_button.setDefault(True)
        open_button.clicked.connect(self._accept_path)
        actions.addWidget(open_button)
        layout.addLayout(actions)

    def _browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose LISAI workspace", self.path_edit.text()
        )
        if path:
            self.path_edit.setText(path)

    def _accept_path(self):
        path = self.path_edit.text().strip()
        if not path:
            self.message.setText("Choose an initialized workspace directory.")
            return
        try:
            self.settings = load_app_settings(path, config_root=self.config_root)
        except Exception as exc:
            self.message.setText(str(exc))
            return
        self.accept()

    def _create(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose empty directory for new workspace", self.path_edit.text()
        )
        if not path:
            return
        if QtWidgets.QMessageBox.question(
            self, "Initialize workspace?", f"Initialize a LISAI workspace in {path}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        try:
            create_workspace(path)
            self.path_edit.setText(path)
            self._accept_path()
        except Exception as exc:
            self.message.setText(str(exc))


class LisaiMainWindow(QtWidgets.QMainWindow):
    """Only composes reusable Qt controls into two tabs; no model computation."""

    def __init__(self, settings: Settings, parent=None, *, config_root=None):
        super().__init__(parent)
        self.settings = settings
        self._explicit_config_root = config_root
        self.session = get_training_session()
        self.setWindowTitle("LISAI")
        self.resize(1180, 880)
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        header = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel("LISAI", central)
        font = name.font()
        font.setBold(True)
        font.setPointSize(max(font.pointSize(), 13))
        name.setFont(font)
        header.addWidget(name)
        header.addStretch(1)
        self.workspace_label = QtWidgets.QLabel(central)
        self.workspace_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        header.addWidget(self.workspace_label)
        self.change_button = QtWidgets.QPushButton("Change workspace…", central)
        self.change_button.clicked.connect(self.change_workspace)
        header.addWidget(self.change_button)
        layout.addLayout(header)
        self.tabs = QtWidgets.QTabWidget(central)
        layout.addWidget(self.tabs, 1)
        self.training = TrainingWorkspaceWidget(settings, parent=self.tabs, session=self.session)
        self.tabs.addTab(self.training, "Training")
        self.inference = StandaloneInferencePage(settings, self.tabs)
        self.tabs.addTab(self.inference, "Inference")
        self._refresh_header()

    def _refresh_header(self):
        self.workspace_label.setText(f"Workspace: {self.settings.WORKSPACE_ROOT}")
        self.workspace_label.setToolTip(
            f"Workspace: {self.settings.WORKSPACE_ROOT}\n"
            f"Experiment configs: {self.settings.CONFIGS_ROOT}"
        )

    def _check_transition(self, *, closing: bool) -> bool:
        if self.inference.is_busy():
            QtWidgets.QMessageBox.information(
                self, "Operation active",
                "Finish the inference or model catalog/download operation before "
                + ("closing LISAI." if closing else "changing workspaces."),
            )
            return False
        if self.session.is_active():
            if not closing:
                QtWidgets.QMessageBox.information(
                    self, "Training running",
                    "Stop the active training before changing workspaces. "
                    "Switching between Training and Inference tabs is always safe.",
                )
                return False
            if QtWidgets.QMessageBox.question(
                self, "Training still running",
                "Closing LISAI will stop the local training process. Close anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            ) != QtWidgets.QMessageBox.Yes:
                return False
        if (self.training.editor.has_unsaved_changes and
                self.training.editor.document.yaml_text.strip()):
            decision = QtWidgets.QMessageBox.question(
                self, "Unsaved training configuration",
                "Save changes to the training YAML before leaving this workspace?\n"
                "Yes = Save, No = Discard, Cancel = Stay.",
                (QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No |
                 QtWidgets.QMessageBox.Cancel),
                QtWidgets.QMessageBox.Cancel,
            )
            if decision == QtWidgets.QMessageBox.Cancel:
                return False
            if decision == QtWidgets.QMessageBox.Yes:
                self.training.editor.save(as_new=False)
                if (self.training.editor.has_unsaved_changes and
                self.training.editor.document.yaml_text.strip()):
                    return False  # save failed or was cancelled
        return True

    def change_workspace(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose LISAI workspace", str(self.settings.WORKSPACE_ROOT)
        )
        if not path or Path(path).resolve() == Path(self.settings.WORKSPACE_ROOT).resolve():
            return
        # Do not discard drafts unless the new workspace has already validated.
        try:
            selected = load_app_settings(path, config_root=self._explicit_config_root)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid LISAI workspace", str(exc))
            return
        if not self._check_transition(closing=False):
            return
        previous_index = self.tabs.currentIndex()
        old_training = self.training
        self.settings = selected
        self.training = TrainingWorkspaceWidget(selected, parent=self.tabs, session=self.session)
        self.tabs.removeTab(0)
        self.tabs.insertTab(0, self.training, "Training")
        old_training.deleteLater()
        self.inference.set_settings(selected)
        self.tabs.setCurrentIndex(previous_index)
        self._refresh_header()

    def closeEvent(self, event):
        if self._check_transition(closing=True):
            super().closeEvent(event)
        else:
            event.ignore()


__all__ = ["LisaiMainWindow", "WorkspaceSelectionDialog", "load_app_settings"]
