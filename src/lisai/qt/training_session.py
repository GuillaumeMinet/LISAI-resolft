"""Application-owned, reusable Qt training process lifetime.

The session survives the closing/hiding of a training dialog. It knows only
LISAI run-discovery services, a command, and its caller's selected workspace.
"""
from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from lisai.application import find_new_completed_run, list_usable_training_runs


class TrainingSession(QtCore.QObject):
    changed = QtCore.Signal()
    log_appended = QtCore.Signal(str)
    completed = QtCore.Signal(str)

    def __init__(self, application):
        super().__init__(application)
        self.application = application
        self.dialog = None
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.running = False
        self.has_started = False
        self.exit_code = None
        self.status_text = "No training running."
        self.run_dir = None
        self.run_id = None
        self.workspace = None
        self.plan = None
        self.run_ids_before = set()
        self.log_text = ""
        self.stop_requested = False
        self._finalized = True
        self._completion_callback = None
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        application.aboutToQuit.connect(self._on_application_exit)

    def is_active(self) -> bool:
        return self.running or self.process.state() != QtCore.QProcess.NotRunning

    def append_log(self, message: str) -> None:
        self.log_text = (self.log_text + message)[-500_000:]
        self.log_appended.emit(message)

    def _read_output(self) -> None:
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if output:
            self.append_log(output)

    def start(self, *, settings, plan, command, run_ids_before, run_dir, on_completed=None):
        if self.is_active():
            raise RuntimeError("A training process is already running.")
        if not command:
            raise ValueError("Training command must not be empty.")
        self.workspace = settings
        self.plan = plan
        self.run_ids_before = set(run_ids_before)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.run_id = None
        self.exit_code = None
        self.stop_requested = False
        self.log_text = ""
        self.has_started = True
        self._finalized = False
        self._completion_callback = on_completed
        self.running = True  # Includes QProcess.Starting.
        self.status_text = "Launching training in an independent Python process…"
        self.process.setWorkingDirectory(str(settings.WORKSPACE_ROOT))
        self.append_log("Starting: " + " ".join(map(str, command[:4])) + " …\n")
        self.changed.emit()
        try:
            self.process.start(str(command[0]), list(map(str, command[1:])))
        except Exception as exc:
            self._finalized = True
            self.running = False
            self.status_text = f"Cannot launch training process: {exc}"
            self._completion_callback = None
            self.changed.emit()
            raise

    def _started(self):
        if not self._finalized:
            self.status_text = "Training running in the background."
            self.changed.emit()

    def reset_for_new_training(self):
        if self.is_active():
            raise RuntimeError("Cannot reset an active training session.")
        self.has_started = False
        self.plan = None
        self.run_dir = None
        self.run_id = None
        self.exit_code = None
        self.stop_requested = False
        self.log_text = ""
        self.status_text = "No training running."
        self._completion_callback = None
        self.changed.emit()

    def _finished(self, exit_code, exit_status):
        self._read_output()
        if self._finalized:
            return
        self._finalized = True
        self.running = False
        self.exit_code = int(exit_code)
        completed_id = None
        if int(exit_code) == 0 and exit_status == QtCore.QProcess.NormalExit:
            try:
                completed_id = find_new_completed_run(
                    self.workspace, dataset_name=self.plan.dataset_name,
                    experiment_name=self.plan.experiment_name,
                    old_run_ids=self.run_ids_before,
                )
                if completed_id:
                    actual = next((run for run in list_usable_training_runs(self.workspace)
                                   if run["run_id"] == completed_id), None)
                    if actual is not None:
                        self.run_dir = actual["run_dir"]
            except Exception as exc:
                self.append_log(f"Run discovery failed: {exc}\n")
                completed_id = None
            self.run_id = completed_id
            self.status_text = (
                f"Training completed successfully. Run: {self.run_dir}"
                + (f" (ID: {completed_id})" if completed_id else "")
            )
        else:
            self.status_text = (
                f"Training stopped or failed (exit code {exit_code}). "
                "Check the log and run folder."
            )
        callback, self._completion_callback = self._completion_callback, None
        self.changed.emit()
        if completed_id:
            self.completed.emit(completed_id)
            if callback is not None:
                try:
                    callback(completed_id)
                except (RuntimeError, ReferenceError):
                    pass

    def _process_error(self, error):
        if self._finalized:
            return
        self.status_text = f"Training process error ({error}): {self.process.errorString()}"
        if self.process.state() == QtCore.QProcess.NotRunning:
            self._finalized = True
            self.running = False
            self._completion_callback = None
        self.changed.emit()

    def request_stop(self, *, force=False):
        if not self.is_active():
            return
        self.stop_requested = True
        if force:
            self.process.kill()
            self.status_text = "Forcibly stopping training…"
        else:
            self.process.terminate()
            self.status_text = "Stop requested. If needed, use Force stop."
        self.changed.emit()

    def _on_application_exit(self):
        # Only survive dialog close, never Qt application destruction.
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1500)

    def show_window(self) -> bool:
        if self.dialog is None:
            return False
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        return True


_session = None


def get_training_session() -> TrainingSession:
    """Return one training process per QApplication (Qt imported only on demand)."""
    global _session
    application = QtWidgets.QApplication.instance()
    if application is None:
        raise RuntimeError("Training requires a running Qt application.")
    if _session is None or _session.application is not application:
        _session = TrainingSession(application)
    return _session


def current_training_session() -> TrainingSession | None:
    """Inspect existing session without creating a process before training is used."""
    if _session is not None and _session.application is QtWidgets.QApplication.instance():
        return _session
    return None


__all__ = ["TrainingSession", "get_training_session", "current_training_session"]
