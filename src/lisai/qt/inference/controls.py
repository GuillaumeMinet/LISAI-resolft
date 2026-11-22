"""Host-neutral model/run selection and inference option controls.

The widget does not choose image files, execute models, or know about ImProcess.
Only ``lisai.application.resolve_inference_selection`` builds execution options.
"""
from __future__ import annotations

from qtpy import QtCore, QtWidgets

from lisai.application.catalogs import get_installed_model_names, list_usable_training_runs
from lisai.application.inference import InferenceSelection, resolve_inference_selection
from lisai.config.settings import Settings

_TILING_CHOICES = ("Model settings", "auto", "off", "256", "512", "1024")
_NORMALIZATION_CHOICES = (
    ("Model settings", ""), ("Training", "training"),
    ("Image", "image"), ("Manual", "manual"),
)
_EDGE_CHOICES = (
    ("Model settings", ""), ("Drop incomplete edges", "drop"),
    ("Black output edges", "black_output"), ("Black input context", "black_input"),
)


class InferenceControlsWidget(QtWidgets.QWidget):
    """Small reusable editor for model identity and *explicit* inference overrides.

    Selecting a model is always deliberate. Refreshing catalogs preserves a
    previously selected identity, even when it becomes unavailable; it never
    silently switches to another model in the workspace.
    """

    selectionChanged = QtCore.Signal()
    optionsChanged = QtCore.Signal()
    manageModelsRequested = QtCore.Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._updating = False
        self._requested = {"installed": "", "run": ""}
        self._available: set[str] = set()
        self._runs: dict[str, dict] = {}
        self._catalog_error: str | None = None

        layout = QtWidgets.QFormLayout(self)
        self.source_combo = QtWidgets.QComboBox(self)
        self.source_combo.addItem("Installed models", "installed")
        self.source_combo.addItem("Completed training runs", "run")
        layout.addRow("Model source:", self.source_combo)

        self.model_combo = QtWidgets.QComboBox(self)
        self.refresh_button = QtWidgets.QPushButton("Refresh", self)
        model_row = QtWidgets.QWidget(self)
        model_layout = QtWidgets.QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.refresh_button)
        layout.addRow("Model / run:", model_row)

        self.checkpoint_combo = QtWidgets.QComboBox(self)
        self.checkpoint_combo.addItem("Best checkpoint", "best")
        self.checkpoint_combo.addItem("Last checkpoint", "last")
        self.checkpoint_combo.setToolTip(
            "Select an existing checkpoint from a completed training run. "
            "Installed models use their bundled checkpoint."
        )
        layout.addRow("Run checkpoint:", self.checkpoint_combo)
        self.manage_button = QtWidgets.QPushButton("Browse / manage models…", self)
        layout.addRow("", self.manage_button)

        self.tiling_combo = QtWidgets.QComboBox(self)
        self.tiling_combo.setEditable(True)
        self.tiling_combo.addItems(_TILING_CHOICES)
        self.tiling_combo.setCurrentText("Model settings")
        self.tiling_combo.setToolTip(
            "Model settings leaves tiling unchanged; 'auto' uses the saved-model "
            "default (not automatic GPU memory planning); 'off' disables tiling. "
            "You can also enter a positive tile size."
        )
        layout.addRow("Tiling size:", self.tiling_combo)

        self.crop_spin = QtWidgets.QSpinBox(self)
        self.crop_spin.setRange(0, 100000)
        self.crop_spin.setSingleStep(16)
        self.crop_spin.setSpecialValueText("No override")
        self.crop_spin.setToolTip("0 preserves the saved model's crop policy.")
        layout.addRow("Crop size (px):", self.crop_spin)

        self.normalization_combo = QtWidgets.QComboBox(self)
        for title, code in _NORMALIZATION_CHOICES:
            self.normalization_combo.addItem(title, code)
        self.normalization_combo.setToolTip(
            "Model settings keeps saved input normalization. 'Image' matches input stack "
            "statistics; 'Manual' applies gain × raw intensity + offset first."
        )
        layout.addRow("Input normalization:", self.normalization_combo)

        self.gain_spin = QtWidgets.QDoubleSpinBox(self)
        self.gain_spin.setRange(0.000001, 1000000.0)
        self.gain_spin.setDecimals(6)
        self.gain_spin.setValue(1.0)
        self.offset_spin = QtWidgets.QDoubleSpinBox(self)
        self.offset_spin.setRange(-1000000000.0, 1000000000.0)
        self.offset_spin.setDecimals(5)
        self.offset_spin.setValue(0.0)
        layout.addRow("Manual gain:", self.gain_spin)
        layout.addRow("Manual offset:", self.offset_spin)

        self.edge_combo = QtWidgets.QComboBox(self)
        for title, code in _EDGE_CHOICES:
            self.edge_combo.addItem(title, code)
        self.edge_combo.setToolTip(
            "Leave model policy unchanged, drop edge predictions, fill output edges "
            "with black, or pad missing input context with black."
        )
        layout.addRow("Temporal edges:", self.edge_combo)

        self.status = QtWidgets.QLabel(self)
        self.status.setWordWrap(True)
        layout.addRow("", self.status)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.checkpoint_combo.currentIndexChanged.connect(self._checkpoint_changed)
        self.tiling_combo.currentTextChanged.connect(self._options_changed)
        self.crop_spin.valueChanged.connect(self._options_changed)
        self.normalization_combo.currentIndexChanged.connect(self._normalization_changed)
        self.gain_spin.valueChanged.connect(self._options_changed)
        self.offset_spin.valueChanged.connect(self._options_changed)
        self.edge_combo.currentIndexChanged.connect(self._options_changed)
        self.refresh_button.clicked.connect(lambda *_: self.refresh_catalog())
        self.manage_button.clicked.connect(lambda *_: self.manageModelsRequested.emit())
        self._update_manual_controls()
        self.refresh_catalog()

    def _source(self) -> str:
        return str(self.source_combo.currentData() or "installed")

    def _source_changed(self, *_):
        if self._updating:
            return
        self.refresh_catalog()

    def _model_changed(self, *_):
        if self._updating:
            return
        self._requested[self._source()] = str(self.model_combo.currentData() or "")
        self._update_checkpoints()
        self.selectionChanged.emit()

    def _update_checkpoints(self):
        source = self._source()
        run = self._runs.get(str(self.model_combo.currentData() or ""))
        checkpoints = run.get("checkpoints", {}) if run is not None else {}
        self.checkpoint_combo.setEnabled(source == "run" and bool(run))
        for i in range(self.checkpoint_combo.count()):
            item = self.checkpoint_combo.model().item(i)
            if item is not None:
                item.setEnabled(source != "run" or bool(checkpoints.get(
                    self.checkpoint_combo.itemData(i), False
                )))
        if source == "run" and run is not None:
            selected = str(self.checkpoint_combo.currentData() or "best")
            if not checkpoints.get(selected, False):
                self.status.setText(
                    f"This run has no {selected!r} checkpoint. Choose an available checkpoint."
                )
            elif self._catalog_error is None:
                self.status.setText("Completed training run selected.")

    def _checkpoint_changed(self, *_):
        if not self._updating:
            self._update_checkpoints()
            self._options_changed()

    def _normalization_changed(self, *_):
        self._update_manual_controls()
        self._options_changed()

    def _update_manual_controls(self):
        enabled = self.normalization_combo.currentData() == "manual"
        self.gain_spin.setEnabled(enabled)
        self.offset_spin.setEnabled(enabled)

    def _options_changed(self, *_):
        if not self._updating:
            self.optionsChanged.emit()

    def refresh_catalog(self):
        """Refresh workspace sources without changing the requested model identity."""
        source = self._source()
        requested = self._requested[source]
        self._updating = True
        try:
            try:
                if source == "installed":
                    names = get_installed_model_names(self.settings)
                    entries = [(name, name) for name in names]
                    self._runs = {}
                else:
                    runs = list_usable_training_runs(self.settings)
                    self._runs = {item["run_id"]: item for item in runs}
                    entries = [(item["label"], item["run_id"]) for item in runs]
                self._catalog_error = None
            except Exception as exc:
                self._catalog_error = f"Cannot load LISAI {source} catalog: {exc}"
                entries = []
                self._runs = {}
            self._available = {str(value) for _, value in entries}
            self.model_combo.clear()
            self.model_combo.addItem("Select a model…" if source == "installed"
                                     else "Select a completed run…", "")
            for label, identifier in entries:
                self.model_combo.addItem(label, identifier)
            if requested and requested not in self._available:
                self.model_combo.addItem(f"{requested} (unavailable)", requested)
            if requested:
                index = self.model_combo.findData(requested)
                self.model_combo.setCurrentIndex(max(index, 0))
            else:
                self.model_combo.setCurrentIndex(0)
            self.checkpoint_combo.setEnabled(source == "run")
            self.manage_button.setEnabled(True)
            if self._catalog_error is not None:
                self.status.setText(self._catalog_error)
            elif requested and requested not in self._available:
                self.status.setText(
                    f"Selected {'model' if source == 'installed' else 'run'} "
                    f"{requested!r} is no longer available in this workspace."
                )
            elif not entries:
                self.status.setText("No installed models. Use Browse / manage models."
                                    if source == "installed" else
                                    "No completed training runs with checkpoints were found.")
            elif not requested:
                self.status.setText("Choose a model or training run to enable inference.")
            else:
                self.status.setText("Model selected. Inference options are ready.")
            self._update_checkpoints()
        finally:
            self._updating = False
        self.selectionChanged.emit()

    def current_values(self) -> dict:
        """Host-neutral values using stable R4-4 ImProcess-compatible field names.

        These are *presentation* values, not a second inference configuration.
        ``inference_selection`` always performs authoritative validation.
        """
        source = self._source()
        identifier = str(self.model_combo.currentData() or "")
        return {
            "model_source": source,
            "model_name": identifier if source == "installed" else "",
            "run_id": identifier if source == "run" else "",
            "checkpoint_selector": str(self.checkpoint_combo.currentData() or "best"),
            "tiling_size": ("" if self.tiling_combo.currentText().strip().lower() == "model settings"
                            else self.tiling_combo.currentText().strip()),
            "crop_size": int(self.crop_spin.value()),
            "normalization_mode": str(self.normalization_combo.currentData() or ""),
            "manual_gain": float(self.gain_spin.value()),
            "manual_offset": float(self.offset_spin.value()),
            "temporal_edge_mode": str(self.edge_combo.currentData() or ""),
        }

    # Convenience names for host parameter adapters; neither serializes anything.
    get_values = current_values

    def set_values(self, values: dict):
        """Restore UI preferences; preserve unavailable identities for explicit review."""
        values = dict(values)
        source = str(values.get("model_source", self._source()))
        if source not in {"installed", "run"}:
            raise ValueError(f"Unsupported model source {source!r}")
        self._requested[source] = str(values.get(
            "run_id" if source == "run" else "model_name", self._requested[source]
        ) or "")
        self._updating = True
        try:
            self.source_combo.setCurrentIndex(0 if source == "installed" else 1)
            checkpoint = str(values.get("checkpoint_selector", "best") or "best")
            self.checkpoint_combo.setCurrentIndex(1 if checkpoint == "last" else 0)
            tiling = values.get("tiling_size", "")
            self.tiling_combo.setCurrentText("Model settings" if tiling in (None, "") else str(tiling))
            self.crop_spin.setValue(int(values.get("crop_size", 0)))
            norm = str(values.get("normalization_mode", "") or "")
            index = self.normalization_combo.findData(norm)
            if index < 0:
                raise ValueError(f"Unsupported normalization choice {norm!r}")
            self.normalization_combo.setCurrentIndex(index)
            self.gain_spin.setValue(float(values.get("manual_gain", 1.0)))
            self.offset_spin.setValue(float(values.get("manual_offset", 0.0)))
            edge = str(values.get("temporal_edge_mode", "") or "")
            index = self.edge_combo.findData(edge)
            if index < 0:
                raise ValueError(f"Unsupported temporal edge choice {edge!r}")
            self.edge_combo.setCurrentIndex(index)
            self._update_manual_controls()
        finally:
            self._updating = False
        self.refresh_catalog()
        self.optionsChanged.emit()

    def set_settings(self, settings: Settings):
        """Replace the selected workspace; never carry its model identity forward."""
        if not isinstance(settings, Settings):
            raise TypeError("Explicit LISAI Settings are required.")
        self.settings = settings
        self._requested = {"installed": "", "run": ""}
        self.refresh_catalog()

    def select_model(self, name: str, *, source: str = "installed"):
        """Select an installed model explicitly, e.g. after model installation."""
        if source not in {"installed", "run"}:
            raise ValueError(f"Unsupported model source {source!r}")
        self._requested[source] = name
        if self._source() != source:
            self.source_combo.setCurrentIndex(0 if source == "installed" else 1)
        else:
            self.refresh_catalog()

    def inference_selection(self) -> InferenceSelection:
        """Return an APP-1 selection; no inference, model weights or file I/O."""
        values = self.current_values()
        identifier = values["model_name"] if values["model_source"] == "installed" else values["run_id"]
        if self._catalog_error is not None:
            raise ValueError(self._catalog_error)
        if not identifier or identifier not in self._available:
            raise ValueError("Choose an available model or completed run first; refresh if necessary.")
        return resolve_inference_selection(
            settings=self.settings, source=values["model_source"], identifier=identifier,
            checkpoint=values["checkpoint_selector"], tiling_size=values["tiling_size"],
            crop_size=values["crop_size"], normalization_mode=values["normalization_mode"],
            manual_gain=values["manual_gain"], manual_offset=values["manual_offset"],
            temporal_edge_mode=values["temporal_edge_mode"],
        )


__all__ = ["InferenceControlsWidget"]
