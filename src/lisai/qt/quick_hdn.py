"""First task-specific training form; the generic monitor is in training_panel."""
from __future__ import annotations

from qtpy import QtCore, QtWidgets


class QuickHDNParametersWidget(QtWidgets.QWidget):
    """Basic scientific HDN controls reusable by any Qt-based LISAI host.

    Dataset selection and image import intentionally remain with the host for now.
    All other LISAI tasks can be launched from a validated training YAML.
    """

    def __init__(self, noise_models=(), parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        experiment_group = QtWidgets.QGroupBox("Experiment · Unsupervised HDN", self)
        exp = QtWidgets.QFormLayout(experiment_group)
        self.beta_kl = QtWidgets.QDoubleSpinBox(experiment_group)
        self.beta_kl.setRange(0.0, 100.0)
        self.beta_kl.setDecimals(5)
        self.beta_kl.setSingleStep(0.05)
        self.beta_kl.setValue(0.5)
        self.beta_kl.setToolTip("Weight of the LVAE KL divergence term (βKL). Lower values usually preserve more detail; tune against data.")
        exp.addRow("βKL (KL loss weight):", self.beta_kl)
        self.noise_combo = QtWidgets.QComboBox(experiment_group)
        self.noise_combo.addItems(list(noise_models))
        self.noise_combo.setToolTip("Calibrated observation noise model used by HDN.")
        exp.addRow("Calibrated noise model:", self.noise_combo)
        layout.addWidget(experiment_group)

        data_group = QtWidgets.QGroupBox("Data preparation", self)
        form = QtWidgets.QFormLayout(data_group)
        self.patch_size = QtWidgets.QSpinBox(data_group)
        self.patch_size.setRange(8, 4096)
        self.patch_size.setSingleStep(16)
        self.patch_size.setValue(64)
        form.addRow("Patch size:", self.patch_size)
        self.test_split = QtWidgets.QCheckBox("Reserve held-out test recording (60/20/20)", data_group)
        form.addRow("Split:", self.test_split)
        self.seed = QtWidgets.QSpinBox(data_group)
        self.seed.setRange(0, 2147483647)
        form.addRow("Split seed:", self.seed)
        layout.addWidget(data_group)

        training_group = QtWidgets.QGroupBox("Optimization", self)
        optim = QtWidgets.QFormLayout(training_group)
        self.epochs = QtWidgets.QSpinBox(training_group)
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(20)
        optim.addRow("Epochs:", self.epochs)
        self.learning_rate_enabled = QtWidgets.QCheckBox("Override preset", training_group)
        self.learning_rate = QtWidgets.QDoubleSpinBox(training_group)
        self.learning_rate.setDecimals(7)
        self.learning_rate.setRange(0.0000001, 1.0)
        self.learning_rate.setValue(0.0001)
        self.learning_rate.setEnabled(False)
        self.learning_rate_enabled.toggled.connect(self.learning_rate.setEnabled)
        lr_row = QtWidgets.QHBoxLayout()
        lr_row.addWidget(self.learning_rate_enabled)
        lr_row.addWidget(self.learning_rate)
        optim.addRow("Learning rate:", lr_row)
        layout.addWidget(training_group)




# The panel deliberately consumes ordinary arrays or TIFF paths. ImProcess (or
# another host) supplies these records; it does not supply UI widgets, callbacks
# for individual form controls, or a second training state machine.
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from lisai.application import list_hdn_noise_models, list_reusable_training_datasets


@dataclass(frozen=True)
class TrainingRecording:
    """A host-provided independent recording, not a frame in a timelapse."""

    name: str
    data: np.ndarray
    axes: tuple[str, ...]
    selected: bool = False


def _training_sources(recordings, files, data_format: str):
    """Convert selected raw arrays and TIFF paths to LISAI's existing L1 inputs."""
    expected = ("Y", "X") if data_format == "single" else ("T", "Y", "X")
    sources = []
    for i, recording in enumerate(recordings, 1):
        array = np.asarray(recording.data)
        if tuple(recording.axes) != expected or array.ndim != len(expected):
            raise ValueError(f"Recording {recording.name!r} does not have {''.join(expected)} axes.")
        filename = str(recording.name or f"recording_{i:04d}")
        filename = filename.replace("/", "_").replace("\\", "_")
        for char in '<>:"|?*':
            filename = filename.replace(char, "_")
        filename = filename.strip().rstrip(". ") or f"recording_{i:04d}"
        if not filename.lower().endswith((".tif", ".tiff")):
            filename += ".tif"
        sources.append((filename, array))
    unique_paths = set()
    for source in files:
        path = Path(source).expanduser().resolve()
        if path in unique_paths:
            continue
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Not a TIFF recording: {path}")
        unique_paths.add(path)
        sources.append(path)
    if len(sources) < 2:
        raise ValueError("Quick HDN needs at least two independent recordings for training/validation.")
    names = [item[0] if isinstance(item, tuple) else item.name for item in sources]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Two recordings have the same TIFF filename; rename one before importing.")
    return sources


class _QuickHDNPreparation(QtCore.QThread):
    imported = QtCore.Signal(object)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, settings, sources, data_format, arguments, prior, import_new, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.sources = sources
        self.data_format = data_format
        self.arguments = arguments
        self.prior = prior
        self.import_new = import_new

    def run(self):
        try:
            from lisai.data import import_training_dataset
            from lisai.training.quick_hdn import prepare_quick_hdn_training

            imported = self.prior
            if imported is None and self.import_new:
                imported = import_training_dataset(
                    self.sources, settings=self.settings,
                    dataset_name=self.arguments["dataset_name"], data_format=self.data_format,
                )
                self.imported.emit(imported)
            # Validate first; only write a new YAML after a successful preflight.
            prepare_quick_hdn_training(**self.arguments, write=False)
            plan = prepare_quick_hdn_training(**self.arguments, write=True)
            self.succeeded.emit(plan)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class QuickHDNSetupWidget(QtWidgets.QWidget):
    """Reusable Quick HDN dataset/import and experiment-parameter form."""

    sourceChanged = QtCore.Signal()

    def __init__(self, settings, recordings=(), parent=None):
        super().__init__(parent)
        self.settings = settings
        self.recordings = tuple(recordings)
        self.files = []
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        container = QtWidgets.QWidget(scroll)
        self.setup_layout = QtWidgets.QVBoxLayout(container)
        scroll.setWidget(container)
        self._build_quick_setup()
        self._refresh_datasets()

    def update_availability(self, *, busy: bool, locked: bool, imported: bool):
        new = self.dataset_source.currentData() == "new"
        editable = not busy and not locked
        self.dataset_source.setEnabled(editable and not imported)
        self.existing_dataset.setEnabled(editable and not new)
        for control in (self.result_list, self.file_list, self.add_files_button,
                        self.add_folder_button, self.remove_files_button,
                        self.data_format, self.dataset_name):
            control.setEnabled(editable and new and not imported)
        self.refresh_datasets.setEnabled(editable and not new)

    def _build_quick_setup(self):
        layout = self.setup_layout
        layout.addWidget(QtWidgets.QLabel(
            "Each YX image or TYX timelapse is one independent recording; "
            "at least two are needed for Quick HDN."
        ))
        self.result_list = QtWidgets.QListWidget(self)
        self.result_list.setMaximumHeight(130)
        for index, item in enumerate(self.recordings):
            array = np.asarray(item.data)
            entry = QtWidgets.QListWidgetItem(f"{item.name} — {array.shape} ({''.join(item.axes)})")
            entry.setData(QtCore.Qt.UserRole, index)
            entry.setFlags(entry.flags() | QtCore.Qt.ItemIsUserCheckable)
            entry.setCheckState(QtCore.Qt.Checked if item.selected else QtCore.Qt.Unchecked)
            self.result_list.addItem(entry)
        layout.addWidget(QtWidgets.QLabel("Available image recordings:"))
        layout.addWidget(self.result_list)
        self.file_list = QtWidgets.QListWidget(self)
        self.file_list.setMaximumHeight(100)
        layout.addWidget(QtWidgets.QLabel("Additional TIFF recordings:"))
        layout.addWidget(self.file_list)
        file_row = QtWidgets.QHBoxLayout()
        self.add_files_button = QtWidgets.QPushButton("Add TIFF files…", self)
        self.add_folder_button = QtWidgets.QPushButton("Add TIFF folder…", self)
        self.remove_files_button = QtWidgets.QPushButton("Remove selected", self)
        for button in (self.add_files_button, self.add_folder_button, self.remove_files_button):
            file_row.addWidget(button)
        layout.addLayout(file_row)
        self.add_files_button.clicked.connect(self._choose_files)
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.remove_files_button.clicked.connect(self._remove_files)

        source_row = QtWidgets.QHBoxLayout()
        self.dataset_source = QtWidgets.QComboBox(self)
        self.dataset_source.addItem("Import new recordings", "new")
        self.dataset_source.addItem("Reuse registered dataset", "existing")
        self.refresh_datasets = QtWidgets.QPushButton("Refresh datasets", self)
        source_row.addWidget(QtWidgets.QLabel("Dataset source:"))
        source_row.addWidget(self.dataset_source, 1)
        source_row.addWidget(self.refresh_datasets)
        layout.addLayout(source_row)
        form = QtWidgets.QFormLayout()
        self.existing_dataset = QtWidgets.QComboBox(self)
        self.data_format = QtWidgets.QComboBox(self)
        self.data_format.addItem("Timelapses (TYX)", "timelapse")
        self.data_format.addItem("2D images (YX)", "single")
        if self.recordings and self.recordings[0].axes == ("Y", "X"):
            self.data_format.setCurrentIndex(1)
        self.dataset_name = QtWidgets.QLineEdit("quick_hdn", self)
        self.experiment_name = QtWidgets.QLineEdit("quick_hdn_run", self)
        form.addRow("Existing dataset:", self.existing_dataset)
        form.addRow("Data format:", self.data_format)
        form.addRow("New dataset name:", self.dataset_name)
        form.addRow("Experiment name:", self.experiment_name)
        layout.addLayout(form)
        self.hdn = QuickHDNParametersWidget(list_hdn_noise_models(self.settings), self)
        layout.addWidget(self.hdn)
        layout.addStretch(1)
        self.refresh_datasets.clicked.connect(self._refresh_datasets)
        self.dataset_source.currentIndexChanged.connect(lambda *_: self.sourceChanged.emit())

    def _choose_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select TIFF recordings", "", "TIFF images (*.tif *.tiff)")
        self._add_paths(paths)

    def _choose_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select TIFF folder")
        if directory:
            self._add_paths(path for path in sorted(Path(directory).iterdir())
                            if path.suffix.lower() in {".tif", ".tiff"})

    def _add_paths(self, paths):
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and path not in self.files:
                self.files.append(path)
        self.file_list.clear()
        self.file_list.addItems([str(path) for path in self.files])

    def _remove_files(self):
        selected = {item.text() for item in self.file_list.selectedItems()}
        self.files = [path for path in self.files if str(path) not in selected]
        self.file_list.clear()
        self.file_list.addItems([str(path) for path in self.files])

    def _refresh_datasets(self):
        previous = self.existing_dataset.currentData()
        self.existing_dataset.clear()
        for entry in list_reusable_training_datasets(self.settings):
            self.existing_dataset.addItem(
                f"{entry['name']} — {entry['data_format']}, {entry['recording_count']} recordings",
                entry["name"],
            )
        if previous:
            index = self.existing_dataset.findData(previous)
            if index >= 0:
                self.existing_dataset.setCurrentIndex(index)

    def _arguments(self):
        if not self.hdn.noise_combo.currentText().strip():
            raise ValueError("No calibrated HDN noise model found in this workspace.")
        return dict(
            settings=self.settings,
            dataset_name=(self.dataset_name.text().strip() if self.dataset_source.currentData() == "new"
                          else str(self.existing_dataset.currentData() or "")),
            noise_model_name=self.hdn.noise_combo.currentText().strip(),
            experiment_name=self.experiment_name.text().strip(),
            epochs=self.hdn.epochs.value(), patch_size=self.hdn.patch_size.value(),
            seed=self.hdn.seed.value(), beta_kl=self.hdn.beta_kl.value(),
            split_ratios=({"train": 0.6, "val": 0.2, "test": 0.2}
                          if self.hdn.test_split.isChecked() else
                          {"train": 0.8, "val": 0.2, "test": 0.0}),
            learning_rate=(self.hdn.learning_rate.value()
                           if self.hdn.learning_rate_enabled.isChecked() else None),
        )

    def _selected_sources(self):
        recordings = [self.recordings[int(self.result_list.item(i).data(QtCore.Qt.UserRole))]
                      for i in range(self.result_list.count())
                      if self.result_list.item(i).checkState() == QtCore.Qt.Checked]
        return _training_sources(recordings, self.files, str(self.data_format.currentData()))


__all__ = ["QuickHDNParametersWidget", "TrainingRecording", "QuickHDNSetupWidget"]
