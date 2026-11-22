"""LISAI-owned Data / Configuration / Training workspace.

The widget is embeddable in any Qt host.  It imports neither ImProcess nor any
microscope-specific types.  Configuration text belongs to TrainingConfigDocument;
process lifetime belongs to TrainingSession, never to the tab navigation.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from qtpy import QtCore, QtWidgets

from lisai.application import (
    list_hdn_noise_models, list_usable_training_runs, plan_for_training_config,
    quick_hdn_training_document, training_command,
)
from lisai.qt.config_editor import TrainingConfigEditorWidget
from lisai.qt.noise_models_panel import NoiseModelsPanel
from lisai.qt.training.execution_panel import TrainingExecutionWidget
from lisai.qt.training.monitor import TrainingMonitorWidget
from lisai.qt.training_data import TrainingDataWidget
from lisai.qt.training_session import get_training_session


def _settings_identity(settings) -> tuple[Path, Path]:
    return Path(settings.WORKSPACE_ROOT).resolve(), Path(settings.CONFIGS_ROOT).resolve()


class TrainingWorkspaceWidget(QtWidgets.QWidget):
    """Three-stage reusable training UI plus a full-width internal Tools page."""

    def __init__(self, settings, recordings=(), parent=None, *, session=None, on_completed=None):
        super().__init__(parent)
        self.settings = settings
        self.session = session or get_training_session()
        self.on_completed = on_completed
        self._validated_plan = None
        self._validated_revision = None
        self._last_document_key = None
        self._return_tab = 1

        outer = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        workspace_label = QtWidgets.QLabel(f"Workspace: {settings.WORKSPACE_ROOT}", self)
        workspace_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        header.addWidget(workspace_label, 1)
        tools = QtWidgets.QToolButton(self)
        tools.setText("Tools ▾")
        tools.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(tools)
        noise_action = menu.addAction("Noise models")
        noise_action.triggered.connect(self.open_noise_models)
        tools.setMenu(menu)
        header.addWidget(tools)
        outer.addLayout(header)

        self.stack = QtWidgets.QStackedWidget(self)
        outer.addWidget(self.stack, 1)
        self.tabs = QtWidgets.QTabWidget(self.stack)
        self.stack.addWidget(self.tabs)

        self.data_page = TrainingDataWidget(settings, recordings, self.tabs)
        self.tabs.addTab(self.data_page, "1 · Data")

        configuration = QtWidgets.QWidget(self.tabs)
        configuration_layout = QtWidgets.QVBoxLayout(configuration)
        self._build_quick_hdn_creator(configuration_layout, configuration)
        self.editor = TrainingConfigEditorWidget(settings, configuration, embedded=True)
        self.editor.dataset_selection_provider = lambda: self.data_page.selected_dataset_name
        configuration_layout.addWidget(self.editor, 1)
        self.tabs.addTab(configuration, "2 · Configuration")

        self.monitor = TrainingMonitorWidget(self.session, self.tabs)
        self.execution_page = TrainingExecutionWidget(self.session, self.monitor, self.tabs)
        self.tabs.addTab(self.execution_page, "3 · Training")

        self.noise_page = NoiseModelsPanel(settings, self.stack)
        self.stack.addWidget(self.noise_page)
        self.noise_page.backRequested.connect(self.back_from_tools)
        self.noise_page.modelSelected.connect(self._use_noise_model)
        self.editor.manageNoiseModelsRequested.connect(self.open_noise_models)

        self.data_page.set_document(self.editor.document)
        self.editor.configOpened.connect(self._config_opened)
        self.editor.draftCreated.connect(self._catalog_draft)
        self.editor.configSaved.connect(self._config_saved)
        self.editor.raw.textChanged.connect(self._document_changed)
        self.data_page.documentUpdated.connect(self._dataset_applied)
        self.data_page.datasetImported.connect(self._dataset_imported)
        self.execution_page.validateRequested.connect(self.validate_for_launch)
        self.execution_page.startRequested.connect(self.start_training)
        self.execution_page.resetRequested.connect(self.clear_finished_session)
        self.session.changed.connect(self._session_changed)
        self.tabs.currentChanged.connect(self._tab_changed)
        self._document_changed()
        self.execution_page.set_job_snapshot(self.session.snapshot())

    def _build_quick_hdn_creator(self, layout, parent):
        # Creation-only fields, not a second live scientific parameter form.
        # Once a draft exists, the shared editor owns all experiment values.
        self.quick_toggle = QtWidgets.QPushButton("Quick HDN from raw dataset…", parent)
        self.quick_toggle.setCheckable(True)
        self.quick_toggle.setChecked(False)
        layout.addWidget(self.quick_toggle)
        self.quick_group = QtWidgets.QGroupBox("Quick HDN · initial draft settings", parent)
        self.quick_group.setVisible(False)
        self.quick_toggle.toggled.connect(self.quick_group.setVisible)
        form = QtWidgets.QHBoxLayout(self.quick_group)
        self.quick_name = QtWidgets.QLineEdit("quick_hdn_run", self.quick_group)
        self.quick_name.setPlaceholderText("Experiment name")
        form.addWidget(self.quick_name, 1)
        self.quick_noise = QtWidgets.QComboBox(self.quick_group)
        self.quick_noise.setToolTip("Initial noise model only. Edit or change it in the resulting configuration.")
        self.quick_noise.addItem("Choose initial noise model…", "")
        for name in list_hdn_noise_models(self.settings):
            self.quick_noise.addItem(name, name)
        form.addWidget(self.quick_noise, 1)
        create = QtWidgets.QPushButton("Create Quick HDN draft", self.quick_group)
        create.clicked.connect(self.create_quick_hdn_draft)
        form.addWidget(create)
        layout.addWidget(self.quick_group)
        self.quick_hint = QtWidgets.QLabel(
            "For unprepared raw recordings only. After creation, edit βKL, noise model, "
            "splits and optimization in the shared configuration editor below.", parent)
        self.quick_hint.setWordWrap(True)
        self.quick_hint.setVisible(False)
        self.quick_toggle.toggled.connect(self.quick_hint.setVisible)
        layout.addWidget(self.quick_hint)

    def create_quick_hdn_draft(self):
        """Reuse scientific Quick HDN preflight, without writing a YAML or importing."""
        name = self.data_page.selected_dataset_name
        noise = self.quick_noise.currentData()
        if not name or not noise:
            self.editor.status.setText("Choose a registered raw dataset in Data and an initial HDN noise model.")
            return
        entry = self.data_page.entries.get(name)
        if entry is None or not entry.quick_hdn_eligible:
            self.editor.status.setText(
                f"Dataset {name!r} is not Quick HDN compatible. "
                "Choose a raw dataset with at least two independent recordings."
            )
            return
        try:
            document, plan = quick_hdn_training_document(
                settings=self.settings, dataset_name=name, noise_model_name=str(noise),
                experiment_name=self.quick_name.text().strip(), epochs=20, patch_size=64,
                beta_kl=0.5,
            )
            if not self.editor.set_document(document):
                return
            self.data_page.set_document(self.editor.document)
            self.editor.status.setText(
                f"Quick HDN draft: {plan.recording_count} recordings; "
                f"train {plan.split_counts['train']}, val {plan.split_counts['val']}, "
                f"test {plan.split_counts['test']}. Save As to train."
            )
            self.editor.draftCreated.emit(document)
            self.quick_toggle.setChecked(False)
        except Exception as exc:
            self.editor.status.setText(f"Quick HDN draft failed: {type(exc).__name__}: {exc}")

    def _config_opened(self, *_):
        self.quick_toggle.setChecked(False)
        self.data_page.set_document(self.editor.document)
        self._document_changed()

    def _catalog_draft(self, *_):
        self.quick_toggle.setChecked(False)
        self.tabs.setCurrentIndex(1)
        self._document_changed()

    def _dataset_imported(self, result):
        self.data_page.status.setText(
            f"Imported {result.dataset_name!r}. Choose a catalog configuration or Quick HDN draft "
            "in Configuration, then apply the dataset deliberately."
        )
        self._document_changed()

    def _dataset_applied(self):
        self.editor.refresh_document_view()
        self._document_changed()

    def _config_saved(self, *_):
        self.data_page.set_document(self.editor.document)
        self._document_changed()

    def _document_changed(self, *_):
        document = self.editor.document
        key = (id(document), document.yaml_text, document.save_path, document.saved_revision)
        if key != self._last_document_key:
            self._validated_plan = None
            self._validated_revision = None
            self._last_document_key = key
        try:
            name = document.read_field(("experiment", "exp_name"))
            mode = document.read_field(("experiment", "mode")) or "train"
            task = document.read_field(("experiment", "task", "name"))
            dataset = document.read_field(("data", "dataset_name"))
        except (ValueError, TypeError):
            name = mode = task = dataset = "(invalid YAML)"
        self.execution_page.set_document_summary(
            source=str(self.editor.current_path) if self.editor.current_path else None,
            experiment=name, dataset=dataset, task=task, mode=mode,
            dirty=self.editor.has_unsaved_changes,
        )
        self._sync()

    def _tab_changed(self, index):
        if index == 2:
            self._document_changed()

    def _session_is_other_workspace(self):
        return (self.session.is_active() and self.session.workspace is not None
                and _settings_identity(self.session.workspace) != _settings_identity(self.settings))

    def _sync(self):
        active = self.session.is_active()
        other = self._session_is_other_workspace()
        self.execution_page.validate_button.setEnabled(not active and self.editor.current_path is not None
                                                      and not self.editor.has_unsaved_changes)
        self.execution_page.start_button.setEnabled(not active and self._validated_plan is not None
                                                   and not self.editor.has_unsaved_changes)
        self.execution_page.reset_button.setEnabled(self.session.has_started and not active)
        self.editor.setEnabled(not active)
        self.data_page.setEnabled(not active)
        self.quick_group.setEnabled(not active)
        if other:
            self.execution_page.message.setText(
                "A local training process is running in a different LISAI workspace. "
                "Open the original training window to monitor it."
            )

    def validate_for_launch(self):
        self._validated_plan = None
        self._validated_revision = None
        try:
            if self.session.is_active():
                raise RuntimeError("A local LISAI training process is already active.")
            if self.editor.current_path is None or self.editor.has_unsaved_changes:
                raise ValueError("Save the current configuration (or Save As) before training.")
            document = self.editor.document
            document.validate_for_training(self.settings)
            path = self.editor.current_path
            if hashlib.sha256(path.read_bytes()).hexdigest() != document.saved_revision:
                raise RuntimeError("The training YAML changed on disk. Reload or Save As before launching.")
            plan = plan_for_training_config(path, settings=self.settings)
            self._validated_plan = plan
            self._validated_revision = document.saved_revision
            suffix = ("Existing run: " + str(plan.run_dir)
                      if plan.mode == "continue_training" else
                      "A new numbered run directory will be allocated when training begins.")
            self.execution_page.message.setText(
                f"Validated {plan.mode} · {plan.dataset_name} · {plan.experiment_name}. {suffix}"
            )
        except Exception as exc:
            self.execution_page.message.setText(f"Cannot launch: {type(exc).__name__}: {exc}")
        self._sync()

    def start_training(self):
        plan = self._validated_plan
        if plan is None or self.session.is_active():
            return
        try:
            if (self.editor.has_unsaved_changes or self.editor.current_path != plan.config_path or
                    self._validated_revision != self.editor.document.saved_revision or
                    hashlib.sha256(plan.config_path.read_bytes()).hexdigest() != self._validated_revision):
                raise ValueError("Configuration changed since validation. Validate again.")
            # Revalidate immediately before spawning, including mode/origin paths.
            latest = plan_for_training_config(plan.config_path, settings=self.settings)
            if (latest.mode, latest.dataset_name, latest.experiment_name) != (
                    plan.mode, plan.dataset_name, plan.experiment_name):
                raise ValueError("Training configuration changed. Validate again.")
            if latest.mode in {"continue_training", "retrain"}:
                prompt = (f"Continue training in {latest.run_dir}?" if latest.mode == "continue_training"
                          else "Retrain from the origin run configured in this YAML? A new run will be created.")
                if QtWidgets.QMessageBox.question(
                        self, "Confirm training mode", prompt,
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                    return
            for dependency in ("torch", "h5py", "tensorboard"):
                if importlib.util.find_spec(dependency) is None:
                    raise RuntimeError(f"Training needs {dependency!r}; install LISAI [full].")
            command = training_command(self.settings, latest)
            old_ids = {run["run_id"] for run in list_usable_training_runs(self.settings)}
            self._validated_plan = None
            self._validated_revision = None
            self.session.start(
                settings=self.settings, plan=latest, command=command,
                run_ids_before=old_ids, run_dir=latest.run_dir,
                on_completed=self.on_completed,
            )
            self.tabs.setCurrentIndex(2)
            self.execution_page.message.setText("Training submitted to local Python process.")
        except Exception as exc:
            self.execution_page.message.setText(f"Cannot start training: {type(exc).__name__}: {exc}")
        self._sync()

    def _session_changed(self):
        self.execution_page.set_job_snapshot(self.session.snapshot())
        if self.session.has_started and self.session.workspace is not None:
            same = _settings_identity(self.session.workspace) == _settings_identity(self.settings)
            if same:
                self.execution_page.message.setText(self.session.status_text)
                if self.session.is_active():
                    self.tabs.setCurrentIndex(2)
        self._sync()

    def clear_finished_session(self):
        if self.session.is_active():
            return
        self.session.reset_for_new_training()
        self._document_changed()  # Retain current YAML; it is reusable.
        self.execution_page.message.setText("Session cleared. The selected YAML is unchanged and reusable.")

    def open_noise_models(self):
        self._return_tab = self.tabs.currentIndex()
        self.noise_page.refresh()
        self.stack.setCurrentWidget(self.noise_page)

    def back_from_tools(self):
        self.stack.setCurrentWidget(self.tabs)
        self.tabs.setCurrentIndex(self._return_tab)

    def _use_noise_model(self, name):
        try:
            if self.editor.document.read_field(("experiment", "task", "name")) != "denoising_hdn":
                raise ValueError("Open or create an HDN configuration before selecting a noise model.")
            self.editor.document.update_field(("noise_model", "name"), name)
            self.editor.refresh_document_view()
            self.tabs.setCurrentIndex(1)
            self.stack.setCurrentWidget(self.tabs)
            self.editor.status.setText(f"Selected noise model {name!r}. Validate and save the configuration.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot select noise model", str(exc))


__all__ = ["TrainingWorkspaceWidget"]
