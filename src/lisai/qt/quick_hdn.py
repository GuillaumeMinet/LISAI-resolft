"""First task-specific training form; the generic monitor is in training_panel."""
from __future__ import annotations

from qtpy import QtWidgets


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


__all__ = ["QuickHDNParametersWidget"]
