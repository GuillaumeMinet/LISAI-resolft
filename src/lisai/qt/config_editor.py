"""Embeddable scientific parameter editor and dialog for LISAI training YAML.

A small curated set of controls uses the *real* Pydantic JSON Schema for hints,
field types and choices. Raw YAML remains available for the complete config.
Round-trip edits preserve comments and unrelated fields via optional ruamel.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from qtpy import QtCore, QtWidgets

from lisai.application import TrainingConfigDocument, list_user_training_configs
from lisai.application.schema_editor import TASK_LABELS, training_editor_fields, update_yaml_field


def _document(text: str) -> dict:
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError("Training YAML must be a mapping.")
    return doc


def _read_path(document: dict, path: tuple[str, ...]) -> Any:
    node: Any = document
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node



class ScientificFieldsWidget(QtWidgets.QWidget):
    """Task-aware scientific fields; independent of any host application."""

    fieldEdited = QtCore.Signal(object, object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._loading = False
        outer = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel("Scientific experiment parameters")
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        outer.addWidget(header)
        self.task_label = QtWidgets.QLabel()
        outer.addWidget(self.task_label)
        self.fields_layout = QtWidgets.QVBoxLayout()
        outer.addLayout(self.fields_layout)
        outer.addStretch(1)
        hint = QtWidgets.QLabel(
            "Only common scientific and training fields are shown. "
            "Use Raw YAML for all other parameters or to change the task. "
            "A task change may also require compatible model/data settings."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

    def _clear(self):
        while self.fields_layout.count():
            item = self.fields_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def load_text(self, text: str):
        self._loading = True
        try:
            document = _document(text)
            task = _read_path(document, ("experiment", "task", "name")) or "custom"
            self.task_label.setText(f"Task: {TASK_LABELS.get(task, task)}")
            self._clear()
            groups = training_editor_fields(self.settings, str(task))
            for group_name, fields in groups.items():
                group = QtWidgets.QGroupBox(group_name, self)
                form = QtWidgets.QFormLayout(group)
                for field in fields:
                    value = _read_path(document, field.path)
                    if field.field_type == "boolean" or isinstance(value, bool):
                        control = QtWidgets.QCheckBox(group)
                        control.setChecked(bool(value if value is not None else field.default))
                        control.toggled.connect(lambda checked, path=field.path: self._edit(path, checked))
                    elif field.choices:
                        control = QtWidgets.QComboBox(group)
                        for choice in field.choices:
                            control.addItem(str(choice), choice)
                        index = control.findData(value)
                        if index >= 0:
                            control.setCurrentIndex(index)
                        control.currentIndexChanged.connect(
                            lambda _index, c=control, path=field.path: self._edit(path, c.currentData()))
                    elif field.field_type == "integer" or (isinstance(value, int) and not isinstance(value, bool)):
                        control = QtWidgets.QSpinBox(group)
                        control.setRange(-2_147_483_647, 2_147_483_647)
                        control.setValue(int(value if value is not None else (field.default or 0)))
                        control.valueChanged.connect(lambda n, path=field.path: self._edit(path, n))
                    elif field.field_type == "number" or isinstance(value, float):
                        control = QtWidgets.QDoubleSpinBox(group)
                        control.setRange(-1e12, 1e12)
                        control.setDecimals(9)
                        control.setSingleStep(0.1 if field.path[-1] == "betaKL" else 0.0001)
                        control.setValue(float(value if value is not None else (field.default or 0)))
                        control.valueChanged.connect(lambda n, path=field.path: self._edit(path, n))
                    else:
                        control = QtWidgets.QLineEdit(group)
                        control.setText("" if value is None else str(value))
                        control.editingFinished.connect(
                            lambda c=control, path=field.path: self._edit(path, c.text()))
                    control.setToolTip(field.description or ".".join(field.path))
                    label = QtWidgets.QLabel(field.title.replace("BetaKL", "βKL"), group)
                    label.setToolTip(field.description)
                    form.addRow(label, control)
                self.fields_layout.addWidget(group)
        finally:
            self._loading = False

    def _edit(self, path, value):
        if not self._loading:
            self.fieldEdited.emit(path, value)


class TrainingConfigEditorWidget(QtWidgets.QWidget):
    """Two views over ONE application-owned configuration document.

    All text/source/dirty/revision state belongs to TrainingConfigDocument.
    QPlainTextEdit is an editable view, not a competing configuration model.
    """

    closeRequested = QtCore.Signal()
    configSaved = QtCore.Signal(object)
    configOpened = QtCore.Signal(object)

    def __init__(self, settings, parent=None, *, on_saved=None, initial=None):
        super().__init__(parent)
        self.settings = settings
        self.on_saved = on_saved
        self.document = TrainingConfigDocument.from_text("")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Edit scientific parameters first, or use Raw YAML for full control. "
            "Packaged presets are read-only; Save As creates a user configuration."
        ))
        self.selector = QtWidgets.QComboBox(self)
        layout.addWidget(self.selector)
        self.tabs = QtWidgets.QTabWidget(self)
        self.guided = ScientificFieldsWidget(settings, self)
        self.tabs.addTab(self.guided, "Experiment parameters")
        self.raw = QtWidgets.QPlainTextEdit(self)
        no_wrap = (QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
                   if hasattr(QtWidgets.QPlainTextEdit, "LineWrapMode") else QtWidgets.QPlainTextEdit.NoWrap)
        self.raw.setLineWrapMode(no_wrap)
        self.tabs.addTab(self.raw, "Raw YAML (all parameters)")
        layout.addWidget(self.tabs, 1)
        self.status = QtWidgets.QLabel("Select a configuration.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Refresh")
        open_button = QtWidgets.QPushButton("Open")
        self.reload_button = QtWidgets.QPushButton("Reload guided fields")
        validate = QtWidgets.QPushButton("Validate")
        save = QtWidgets.QPushButton("Save…")
        save_as = QtWidgets.QPushButton("Save As…")
        close = QtWidgets.QPushButton("Close")
        for button in (refresh, open_button, self.reload_button, validate, save, save_as, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        refresh.clicked.connect(self.refresh)
        open_button.clicked.connect(self.open_selected)
        self.reload_button.clicked.connect(self.reload_guided)
        self.guided.fieldEdited.connect(self._edited_field)
        self.raw.textChanged.connect(self._raw_changed)
        self.tabs.currentChanged.connect(self._tab_changed)
        validate.clicked.connect(self.validate)
        save.clicked.connect(lambda: self.save(as_new=False))
        save_as.clicked.connect(lambda: self.save(as_new=True))
        close.clicked.connect(self.closeRequested.emit)
        self.refresh(initial)
        if self.selector.count():
            self.open_selected()

    @property
    def current_path(self) -> Path | None:
        return self.document.save_path

    @property
    def clean_text(self) -> str:
        """R3-compatible read-only view; document owns the saved text."""
        return self.document.saved_text or ""

    @property
    def has_unsaved_changes(self) -> bool:
        return self.document.dirty

    def _raw_changed(self):
        self.document.replace_text(self.raw.toPlainText())

    def refresh(self, selected=None):
        previous = selected if isinstance(selected, (str, Path)) else self.selector.currentData()
        self.selector.clear()
        for name, path in list_user_training_configs(self.settings):
            self.selector.addItem(name, str(path))
        if previous:
            for index in range(self.selector.count()):
                if self.selector.itemData(index) == str(previous):
                    self.selector.setCurrentIndex(index)
                    break

    def _confirm_discard(self):
        if not self.document.dirty or (self.document.save_path is None and not self.document.yaml_text.strip()):
            return True
        return (QtWidgets.QMessageBox.question(
            self, "Discard edits?", "Discard unsaved configuration changes?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes)

    def set_document(self, document: TrainingConfigDocument, *, confirm_discard=True) -> bool:
        """Display an existing/draft document without copying its authoritative state."""
        if confirm_discard and not self._confirm_discard():
            return False
        self.document = document
        with QtCore.QSignalBlocker(self.raw):
            self.raw.setPlainText(self.document.yaml_text)
        self.refresh(self.document.save_path)
        self.reload_guided()
        self.status.setText(str(self.current_path) if self.current_path else
                            f"Unsaved {self.document.source_kind} configuration — Save As to create a file.")
        self.configOpened.emit(self.current_path)
        return True

    def load_path(self, value, *, confirm_discard=True) -> bool:
        """Open one user-owned YAML; never modify the selected file on read."""
        try:
            candidate = TrainingConfigDocument.open(self.settings, value)
            return self.set_document(candidate, confirm_discard=confirm_discard)
        except Exception as error:
            self.status.setText(f"Open failed: {error}")
            return False

    def load_source(self, source_ref: str, *, confirm_discard=True) -> bool:
        """Load a built-in preset/example/template as an unsaved draft."""
        try:
            candidate = TrainingConfigDocument.from_source(self.settings, source_ref)
            return self.set_document(candidate, confirm_discard=confirm_discard)
        except Exception as error:
            self.status.setText(f"Open failed: {error}")
            return False

    def open_selected(self):
        value = self.selector.currentData()
        if value:
            self.load_path(value)

    def reload_guided(self):
        try:
            self.guided.load_text(self.document.yaml_text)
            self.guided.setEnabled(True)
        except Exception as error:
            self.guided._clear()
            self.guided.task_label.setText("Guided fields unavailable until YAML syntax is corrected.")
            self.guided.setEnabled(False)
            self.status.setText(f"Cannot show guided fields: {error}. Raw YAML remains editable.")

    def _tab_changed(self, index):
        if index == 0:
            self.reload_guided()

    def _edited_field(self, path, value):
        try:
            self.document.update_field(tuple(path), value)
            self.raw.setPlainText(self.document.yaml_text)
            self.status.setText(f"Edited {'.'.join(path)} — validate and save to apply.")
        except Exception as error:
            self.status.setText(f"Guided edit failed: {error}. Use Raw YAML instead.")
            self.reload_guided()

    def validate(self) -> bool:
        try:
            resolved = self.document.validate_for_training(self.settings)
            self.status.setText(f"Trainable {resolved.experiment.mode} configuration: "
                                f"{resolved.experiment.exp_name}")
            return True
        except Exception as error:
            self.status.setText(f"Not trainable: {type(error).__name__}: {error}")
            return False

    def save(self, *, as_new: bool) -> None:
        try:
            warnings = self.document.validate_for_save(self.settings)
        except Exception as error:
            self.status.setText(f"Cannot save invalid YAML: {type(error).__name__}: {error}")
            return
        if as_new or self.current_path is None:
            target = self.current_path
            proposal = f"{target.stem}_copy.yml" if target is not None else "new_experiment.yml"
            value, ok = QtWidgets.QInputDialog.getText(
                self, "Save training YAML as…", "Path relative to configs/training:", text=proposal)
            if not ok or not value.strip():
                return
            destination = value.strip()
            operation = lambda: self.document.save_as(self.settings, destination)
        else:
            if self.document.dirty and QtWidgets.QMessageBox.question(
                self, "Overwrite training YAML?",
                f"Save edits to {self.current_path.name}? Save As creates a separate copy.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
            operation = lambda: self.document.save(self.settings)
        try:
            path = operation()
            # Document canonicalizes the on-disk newline after a successful save.
            if self.raw.toPlainText() != self.document.yaml_text:
                with QtCore.QSignalBlocker(self.raw):
                    self.raw.setPlainText(self.document.yaml_text)
            self.refresh(str(path))
            suffix = " (draft saved; not ready for training)" if warnings else ""
            self.status.setText(f"Saved: {path}{suffix}" +
                                (". " + "; ".join(warnings) if warnings else ""))
            self.configSaved.emit(path)
            if self.on_saved is not None:
                self.on_saved(path)
        except Exception as error:
            self.status.setText(f"Save failed: {type(error).__name__}: {error}")


class TrainingConfigEditorDialog(QtWidgets.QDialog):
    """Small host-independent wrapper around the embeddable editor widget."""

    def __init__(self, settings, parent=None, *, on_saved=None, initial=None):
        super().__init__(parent)
        self.setWindowTitle("LISAI — Training configurations")
        self.resize(900, 750)
        layout = QtWidgets.QVBoxLayout(self)
        self.editor = TrainingConfigEditorWidget(settings, self, on_saved=on_saved, initial=initial)
        self.editor.closeRequested.connect(self.accept)
        layout.addWidget(self.editor)


def show_training_config_editor(parent, settings, *, on_saved=None, initial=None):
    dialog = TrainingConfigEditorDialog(settings, parent, on_saved=on_saved, initial=initial)
    if hasattr(dialog, "exec"):
        dialog.exec()
    else:
        dialog.exec_()
    return dialog


__all__ = [
    "ScientificFieldsWidget", "TrainingConfigEditorWidget", "TrainingConfigEditorDialog",
    "show_training_config_editor", "update_yaml_field",
]
