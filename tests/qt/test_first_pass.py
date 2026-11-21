"""Optional GUI smoke tests (run on a machine with QtPy and a Qt binding)."""
from __future__ import annotations

import pytest

try:
    from qtpy import QtWidgets
except Exception:
    pytest.skip("Qt binding not installed", allow_module_level=True)

from lisai.config.settings import Settings
from lisai.config.workspace import initialize_workspace
from lisai.qt.config_editor import TrainingConfigEditorWidget
from lisai.qt.quick_hdn import QuickHDNParametersWidget
from lisai.qt.training_panel import TrainingMonitorWidget
from lisai.qt.training_session import current_training_session, get_training_session


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_reusable_training_widgets_can_be_created_without_improcess(qapp):
    session = get_training_session()
    assert session is get_training_session()
    assert current_training_session() is session
    assert not session.is_active()
    monitor = TrainingMonitorWidget(session)
    assert not monitor.stop_button.isEnabled()
    assert not monitor.force_button.isEnabled()
    quick = QuickHDNParametersWidget(["calibrated"])
    assert quick.beta_kl.value() == 0.5
    assert quick.noise_combo.currentText() == "calibrated"
    quick.close()
    monitor.close()


def test_editor_has_guided_and_full_raw_yaml(qapp, tmp_path):
    root = tmp_path / "ws"
    initialize_workspace(root)
    settings = Settings.from_workspace(root)
    editor = TrainingConfigEditorWidget(settings)
    assert editor.tabs.tabText(0) == "Experiment parameters"
    assert "Raw YAML" in editor.tabs.tabText(1)
    raw = """# preserve this comment
experiment:
  task:
    name: denoising_hdn
    betaKL: 0.5
    supervised: false
  exp_name: demo
training:
  n_epochs: 1
"""
    editor.raw.setPlainText(raw)
    editor.reload_guided()
    editor.guided.fieldEdited.emit(("experiment", "task", "betaKL"), 0.4)
    assert "# preserve this comment" in editor.raw.toPlainText()
    assert "betaKL: 0.4" in editor.raw.toPlainText()
    editor.close()
