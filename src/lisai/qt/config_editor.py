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

from lisai.application import (
    list_user_training_configs, managed_training_config_path,
    save_training_yaml_text, validate_training_yaml_text,
)
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
    """Embeddable training config browser, guided fields, raw YAML and safe saves."""

    closeRequested = QtCore.Signal()
    configSaved = QtCore.Signal(object)
    configOpened = QtCore.Signal(object)

    def __init__(self, settings, parent=None, *, on_saved=None, initial=None):
        super().__init__(parent)
        self.settings = settings
        self.on_saved = on_saved
        self.current_path: Path | None = None
        self.clean_text = ""
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
        self.tabs.currentChanged.connect(self._tab_changed)
        validate.clicked.connect(self.validate)
        save.clicked.connect(lambda: self.save(as_new=False))
        save_as.clicked.connect(lambda: self.save(as_new=True))
        close.clicked.connect(self.closeRequested.emit)
        self.refresh(initial)
        if self.selector.count():
            self.open_selected()

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

    @property
    def has_unsaved_changes(self) -> bool:
        return self.raw.toPlainText() != self.clean_text

    def _confirm_discard(self):
        if not self.has_unsaved_changes:
            return True
        return (QtWidgets.QMessageBox.question(
            self, "Discard edits?", "Discard changes not saved to disk?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes)

    def load_path(self, value, *, confirm_discard=True) -> bool:
        """Load a concrete user YAML into both editors; safe for host composition."""
        if confirm_discard and not self._confirm_discard():
            return False
        try:
            source = managed_training_config_path(self.settings, value)
            text = source.read_text(encoding="utf-8")
            self.current_path, self.clean_text = source, text
            self.raw.setPlainText(text)
            self.refresh(str(source))
            self.reload_guided()
            self.status.setText(str(source))
            self.configOpened.emit(source)
            return True
        except Exception as error:
            self.status.setText(f"Open failed: {error}")
            return False

    def open_selected(self):
        value = self.selector.currentData()
        if value:
            self.load_path(value)

    def reload_guided(self):
        try:
            self.guided.load_text(self.raw.toPlainText())
        except Exception as error:
            self.status.setText(f"Cannot show guided fields: {error}. Raw YAML remains editable.")

    def _tab_changed(self, index):
        if index == 0:
            self.reload_guided()

    def _edited_field(self, path, value):
        try:
            updated = update_yaml_field(self.raw.toPlainText(), tuple(path), value)
            self.raw.setPlainText(updated)
            self.status.setText(f"Edited {'.'.join(path)} — validate and save to apply.")
        except Exception as error:
            self.status.setText(f"Guided edit failed: {error}. Use Raw YAML instead.")
            self.reload_guided()

    def validate(self) -> bool:
        try:
            resolved = validate_training_yaml_text(self.raw.toPlainText(), self.settings)
            self.status.setText(f"Valid {resolved.experiment.mode} configuration: "
                                f"{resolved.experiment.exp_name}")
            return True
        except Exception as error:
            self.status.setText(f"Validation failed: {type(error).__name__}: {error}")
            return False

    def save(self, *, as_new: bool) -> None:
        if not self.validate():
            return
        target = self.current_path
        if as_new or target is None:
            proposal = f"{target.stem}_copy.yml" if target is not None else "new_experiment.yml"
            value, ok = QtWidgets.QInputDialog.getText(
                self, "Save training YAML as…", "Path relative to configs/training:", text=proposal)
            if not ok or not value.strip():
                return
            target = value.strip()
        elif QtWidgets.QMessageBox.question(
            self, "Overwrite training YAML?", "Overwrite this configuration? Save As is safer for a new run.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        try:
            path = save_training_yaml_text(self.raw.toPlainText(), target, settings=self.settings,
                                           overwrite=not as_new and self.current_path is not None)
            self.current_path, self.clean_text = path, self.raw.toPlainText()
            self.refresh(str(path))
            self.status.setText(f"Saved: {path}")
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
