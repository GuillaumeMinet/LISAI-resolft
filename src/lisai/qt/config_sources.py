"""In-editor browser for saved configurations and packaged creation sources."""
from __future__ import annotations

from qtpy import QtCore, QtWidgets

from lisai.application.configs import create_document_from_source, list_config_sources
from lisai.application.schema_editor import TASK_LABELS


class TrainingConfigSourceWidget(QtWidgets.QGroupBox):
    """Create an unsaved configuration draft; file persistence stays in editor."""

    draftReady = QtCore.Signal(object, object)

    def __init__(self, settings, parent=None, *, dataset_provider=None):
        super().__init__("Create configuration from LISAI catalog", parent)
        self.settings = settings
        self.dataset_provider = dataset_provider
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.task = QtWidgets.QComboBox(self)
        self.task.addItem("All tasks", None)
        for task, label in TASK_LABELS.items():
            if task != "custom":
                self.task.addItem(label, task)
        self.kind = QtWidgets.QComboBox(self)
        for label, kind in (("Presets", "preset"), ("Examples", "example"),
                            ("Templates", "template")):
            self.kind.addItem(label, kind)
        row.addWidget(self.task, 1)
        row.addWidget(self.kind, 1)
        layout.addLayout(row)
        self.source = QtWidgets.QComboBox(self)
        layout.addWidget(self.source)
        self.description = QtWidgets.QLabel(self)
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        form = QtWidgets.QFormLayout()
        self.experiment = QtWidgets.QLineEdit(self)
        self.experiment.setPlaceholderText("Experiment name (required to complete presets)")
        form.addRow("Experiment name:", self.experiment)
        layout.addLayout(form)
        self.create_button = QtWidgets.QPushButton("Create editable draft", self)
        layout.addWidget(self.create_button)
        self.task.currentIndexChanged.connect(self.refresh)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.source.currentIndexChanged.connect(self._details)
        self.create_button.clicked.connect(self.create_draft)
        self.refresh()

    def refresh(self, *_):
        previous = self.source.currentData()
        kind = self.kind.currentData()
        task = self.task.currentData()
        sources = list_config_sources(self.settings, kind=kind, task=task)
        self.entries = {entry.relative_path: entry for entry in sources
                        if entry.ownership == "built-in"}
        with QtCore.QSignalBlocker(self.source):
            self.source.clear()
            for item in self.entries.values():
                self.source.addItem(item.name, item.relative_path)
            index = self.source.findData(previous)
            if index >= 0:
                self.source.setCurrentIndex(index)
        self._details()

    def _details(self, *_):
        item = self.entries.get(self.source.currentData())
        self.description.setText(item.description or "" if item else "No matching packaged configurations.")
        self.create_button.setEnabled(item is not None)
        self.experiment.setEnabled(item is not None and item.kind != "example")
        if item is not None:
            self.source.setToolTip(item.relative_path)

    def create_draft(self):
        source = self.source.currentData()
        if not source:
            return
        selection = self.dataset_provider() if self.dataset_provider else None
        try:
            document, warnings = create_document_from_source(
                self.settings, source,
                dataset_selection=selection,
                experiment_name=self.experiment.text().strip() or None,
            )
            self.draftReady.emit(document, warnings)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot create training configuration", str(exc))


__all__ = ["TrainingConfigSourceWidget"]
