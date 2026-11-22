"""In-window noise model resource browser, independent of ImProcess."""
from __future__ import annotations

from qtpy import QtCore, QtWidgets

from lisai.application.noise_models import list_noise_model_entries


class NoiseModelsPanel(QtWidgets.QWidget):
    """Full-width inspect/select view; discovery does not certify a calibration."""

    modelSelected = QtCore.Signal(str)
    backRequested = QtCore.Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        main = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        back = QtWidgets.QPushButton("← Back to training", self)
        back.clicked.connect(self.backRequested.emit)
        header.addWidget(back)
        heading = QtWidgets.QLabel("Noise models", self)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        header.addWidget(heading)
        header.addStretch(1)
        refresh = QtWidgets.QPushButton("Refresh", self)
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        main.addLayout(header)
        explanation = QtWidgets.QLabel(
            "Inspect workspace noise-model resources. File discovery is not a scientific "
            "validation of the calibration. Model creation and diagnostic checking "
            "will be added here in a later iteration.", self)
        explanation.setWordWrap(True)
        main.addWidget(explanation)
        columns = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget(self)
        columns.addWidget(self.list, 2)
        self.details = QtWidgets.QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select a noise model to inspect its files.")
        columns.addWidget(self.details, 3)
        main.addLayout(columns, 1)
        self.use_button = QtWidgets.QPushButton("Use selected model in HDN configuration", self)
        self.use_button.clicked.connect(self._use)
        main.addWidget(self.use_button)
        self.list.currentItemChanged.connect(self._show_details)
        self.entries = {}
        self.refresh()

    def refresh(self):
        selected = self.list.currentItem()
        previous = selected.data(QtCore.Qt.UserRole) if selected is not None else None
        self.entries = {entry.name: entry for entry in list_noise_model_entries(self.settings)}
        self.list.clear()
        for name, entry in self.entries.items():
            status = "files present — not validated" if entry.usable else "incomplete"
            item = QtWidgets.QListWidgetItem(f"{name}  ·  {status}", self.list)
            item.setData(QtCore.Qt.UserRole, name)
        if previous in self.entries:
            for i in range(self.list.count()):
                if self.list.item(i).data(QtCore.Qt.UserRole) == previous:
                    self.list.setCurrentRow(i)
                    break
        elif self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.details.setPlainText("No noise-model directories were found in this workspace.")
            self.use_button.setEnabled(False)

    def _show_details(self, *_):
        item = self.list.currentItem()
        entry = self.entries.get(item.data(QtCore.Qt.UserRole)) if item is not None else None
        if entry is None:
            self.details.clear()
            self.use_button.setEnabled(False)
            return
        self.details.setPlainText(
            f"Name: {entry.name}\nFolder: {entry.directory}\n\n"
            f"GMM model: {entry.model_path}\n  {'found' if entry.model_present else 'MISSING'}\n\n"
            f"Normalization: {entry.normalization_path}\n  "
            f"{'found' if entry.normalization_present else 'MISSING'}\n\n"
            "Calibration status: NOT VALIDATED. Presence of these files "
            "does not guarantee compatible camera/noise statistics."
        )
        self.use_button.setEnabled(entry.usable)

    def _use(self):
        item = self.list.currentItem()
        entry = self.entries.get(item.data(QtCore.Qt.UserRole)) if item is not None else None
        if entry is not None and entry.usable:
            self.modelSelected.emit(entry.name)


__all__ = ["NoiseModelsPanel"]
