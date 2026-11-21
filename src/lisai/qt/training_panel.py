"""Reusable Qt training output and execution controls (all LISAI training tasks)."""
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


from pathlib import Path
import importlib.util

from lisai.application import list_usable_training_runs, plan_for_training_config, training_command
from lisai.qt.config_editor import TrainingConfigEditorWidget
from lisai.qt.quick_hdn import TrainingRecording, QuickHDNSetupWidget, _QuickHDNPreparation
from lisai.qt.training_session import get_training_session


class TrainingPanel(QtWidgets.QWidget):
    """Embeddable, host-independent training workflow.

    Quick HDN creates a YAML; the same editor and runner then handle that YAML
    or any already-authored LISAI training YAML. No ImProcess dependency.
    """

    def __init__(self, settings, recordings=(), parent=None, *, session=None, on_completed=None):
        super().__init__(parent)
        self.settings = settings
        self.session = session or get_training_session()
        self.on_completed = on_completed
        self.imported = None
        self.prepared = None
        self.worker = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(QtWidgets.QLabel(f"Workspace: {settings.WORKSPACE_ROOT}"))
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Workflow:"))
        self.workflow = QtWidgets.QComboBox(self)
        self.workflow.addItem("Quick HDN — prepare from recordings", "quick")
        self.workflow.addItem("Existing LISAI training YAML — any task", "saved")
        mode_row.addWidget(self.workflow, 1)
        outer.addLayout(mode_row)

        self.tabs = QtWidgets.QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        self.setup = QuickHDNSetupWidget(settings, recordings, self.tabs)
        self.tabs.addTab(self.setup, "1 · Prepare")

        self.editor = TrainingConfigEditorWidget(settings, self.tabs)
        self.editor.closeRequested.connect(lambda: self.tabs.setCurrentIndex(0))
        self.editor.configOpened.connect(self._config_opened)
        self.editor.configSaved.connect(self._config_saved)
        self.editor.raw.textChanged.connect(self._sync)
        self.tabs.addTab(self.editor, "2 · Configuration / Raw YAML")

        self.monitor = TrainingMonitorWidget(self.session, self.tabs)
        self.tabs.addTab(self.monitor, "3 · Training progress")
        self.report = QtWidgets.QPlainTextEdit(self)
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(90)
        outer.addWidget(self.report)
        self.status = QtWidgets.QLabel("Prepare Quick HDN or select an existing YAML.", self)
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.prepare_button = QtWidgets.QPushButton("Prepare HDN training", self)
        self.start_button = QtWidgets.QPushButton("Start training", self)
        self.new_button = QtWidgets.QPushButton("New training", self)
        for button in (self.prepare_button, self.start_button, self.new_button):
            row.addWidget(button)
        outer.addLayout(row)
        self.prepare_button.clicked.connect(self.prepare)
        self.start_button.clicked.connect(self.start_training)
        self.new_button.clicked.connect(self.new_training)
        self.workflow.currentIndexChanged.connect(self._sync)
        self.session.changed.connect(self._refresh_session)
        self.setup.sourceChanged.connect(self._sync)
        self._sync()

    def _config_opened(self, *_):
        self.prepared = None
        self._sync()

    def _config_saved(self, path):
        # Saving edits to an already prepared config does not require a second
        # Quick HDN import/creation. Start still revalidates the *saved* YAML.
        if self.prepared is not None and Path(self.prepared.config_path) != Path(path):
            self.prepared = None
        self._sync()

    def _sync(self, *_):
        busy = self.worker is not None or self.session.is_active()
        quick = self.workflow.currentData() == "quick"
        if not quick and self.tabs.currentWidget() is self.setup:
            self.tabs.setCurrentWidget(self.editor)
        imported = self.imported is not None
        locked = self.prepared is not None
        self.setup.setEnabled(quick and not busy and not locked)
        self.setup.update_availability(busy=busy, locked=locked, imported=imported)
        self.workflow.setEnabled(not busy and not locked)
        self.editor.setEnabled(not busy)
        self.prepare_button.setText("Prepare Quick HDN" if quick else "Validate existing YAML")
        self.prepare_button.setEnabled(not busy and not locked)
        self.start_button.setEnabled(not busy and locked and not self.editor.has_unsaved_changes)
        self.new_button.setEnabled(not busy)

    def _refresh_session(self):
        if self.session.has_started:
            self.status.setText(self.session.status_text)
            if self.session.is_active():
                self.tabs.setCurrentWidget(self.monitor)
        self._sync()

    def prepare(self):
        if self.worker is not None or self.prepared is not None or self.session.is_active():
            return
        if self.workflow.currentData() == "saved":
            try:
                if self.editor.has_unsaved_changes:
                    raise ValueError("Save edits in Configuration / Raw YAML before validating for launch.")
                path = self.editor.current_path
                if path is None:
                    raise ValueError("Open a training YAML in the Configuration tab first.")
                if not self.editor.validate():
                    raise ValueError(self.editor.status.text())
                self.prepared = plan_for_training_config(path, settings=self.settings)
                self.report.setPlainText(f"Training YAML: {path}\nTask: {self.prepared.mode}\n"
                                         f"Dataset: {self.prepared.dataset_name}\n"
                                         f"Experiment: {self.prepared.experiment_name}")
                self.status.setText("Saved YAML validated. Training has not started.")
            except Exception as exc:
                self.status.setText(f"Cannot validate training YAML: {exc}")
            self._sync()
            return
        try:
            arguments = self.setup._arguments()
            is_new = self.setup.dataset_source.currentData() == "new"
            sources = self.setup._selected_sources() if is_new and self.imported is None else []
            if self.imported is not None and arguments["dataset_name"] != self.imported.dataset_name:
                raise ValueError("Imported dataset is locked. Choose New training to change datasets.")
            if not is_new and not arguments["dataset_name"]:
                raise ValueError("No compatible registered dataset available.")
            if is_new and self.imported is None and self.setup.hdn.test_split.isChecked() and len(sources) < 3:
                raise ValueError("A test split needs three independent recordings.")
        except Exception as exc:
            self.status.setText(str(exc))
            return
        if is_new and self.imported is None:
            if QtWidgets.QMessageBox.question(
                self, "Create LISAI dataset",
                f"Import {len(sources)} recordings into dataset {arguments['dataset_name']!r}? "
                "This copies TIFF files into the LISAI workspace.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            ) != QtWidgets.QMessageBox.Yes:
                return
        self.worker = _QuickHDNPreparation(
            self.settings, sources, str(self.setup.data_format.currentData()), arguments,
            self.imported, is_new, self,
        )
        self.worker.imported.connect(self._imported)
        self.worker.succeeded.connect(self._prepared_hdn)
        self.worker.failed.connect(lambda message: self.status.setText(
            message + " — Imported data, if any, is retained for retry."))
        self.worker.finished.connect(self._preparation_finished)
        self.status.setText("Importing data and preparing Quick HDN training YAML…")
        self._sync()
        self.worker.start()

    def _imported(self, imported):
        self.imported = imported

    def _prepared_hdn(self, plan):
        if not self.editor.load_path(plan.config_path):
            self.status.setText("HDN YAML prepared, but cannot open it in the editor.")
            return
        self.prepared = plan
        self.report.setPlainText(
            f"Training YAML: {plan.config_path}\n"
            f"Train: {plan.split_counts['train']} files\n"
            f"Validation: {plan.split_counts['val']} files\n"
            f"Test: {plan.split_counts['test']} files")
        self.status.setText("Quick HDN YAML created. Review guided fields or Raw YAML, then Start.")
        self.tabs.setCurrentWidget(self.editor)
        self._sync()

    def _preparation_finished(self):
        job, self.worker = self.worker, None
        if job is not None:
            job.deleteLater()
        self._sync()

    def start_training(self):
        if self.worker is not None or self.session.is_active() or self.prepared is None:
            return
        try:
            if self.editor.has_unsaved_changes:
                raise ValueError("Unsaved configuration edits. Save (or Save As) before starting.")
            path = self.editor.current_path
            if path is None or path != Path(self.prepared.config_path):
                raise ValueError("Configuration changed. Validate the selected YAML again.")
            plan = plan_for_training_config(path, settings=self.settings)
            if (plan.dataset_name != self.prepared.dataset_name or
                    plan.experiment_name != self.prepared.experiment_name or
                    plan.mode != getattr(self.prepared, "mode", "train")):
                raise ValueError("Training dataset or experiment changed. Validate again first.")
            if plan.mode == "continue_training":
                if QtWidgets.QMessageBox.question(
                    self, "Continue existing run", f"Resume training in {plan.run_dir}?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                ) != QtWidgets.QMessageBox.Yes:
                    return
            for dependency in ("torch", "h5py", "tensorboard"):
                if importlib.util.find_spec(dependency) is None:
                    raise RuntimeError(f"Training needs {dependency!r}; install LISAI [full].")
            command = training_command(self.settings, plan)
            old_ids = {run["run_id"] for run in list_usable_training_runs(self.settings)}
            self.session.start(
                settings=self.settings, plan=plan, command=command,
                run_ids_before=old_ids, run_dir=plan.run_dir,
                on_completed=self.on_completed,
            )
        except Exception as exc:
            self.status.setText(f"Cannot start training: {exc}")

    def new_training(self):
        if self.worker is not None or self.session.is_active():
            return
        self.session.reset_for_new_training()
        self.imported = None
        self.prepared = None
        self.report.clear()
        self.status.setText("Choose a new experiment or saved training YAML.")
        self.tabs.setCurrentIndex(0)
        self.setup._refresh_datasets()
        self._sync()


class TrainingDialog(QtWidgets.QDialog):
    """Modeless wrapper: hiding never stops an active training process."""

    def __init__(self, settings, recordings=(), *, session=None, on_completed=None):
        super().__init__(None)  # Not destroyed with an ImProcess parameter widget.
        self.setWindowTitle("LISAI — Training")
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.resize(900, 800)
        layout = QtWidgets.QVBoxLayout(self)
        self.panel = TrainingPanel(
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
    """Open/reopen a reusable LISAI trainer without importing ImProcess."""
    session = session or get_training_session()
    if session.show_window():
        return session.dialog
    dialog = TrainingDialog(settings, recordings, session=session, on_completed=on_completed)
    session.dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


__all__ = [
    "TrainingMonitorWidget", "TrainingRecording", "TrainingPanel", "TrainingDialog",
    "show_training_dialog",
]
