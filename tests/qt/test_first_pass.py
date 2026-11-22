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
from lisai.qt.training_panel import TrainingMonitorWidget, TrainingPanel, TrainingRecording
from lisai.qt.quick_hdn import _training_sources
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


def test_training_panel_composes_editor_quick_hdn_and_generic_saved_yaml(qapp, tmp_path):
    settings = Settings.from_workspace(_init_workspace(tmp_path / "r3-ws"))
    records = [TrainingRecording("frame1", __import__("numpy").ones((16, 16)), ("Y", "X"), True)]
    panel = TrainingPanel(settings, records)
    assert panel.setup.hdn.beta_kl.value() == 0.5
    assert panel.editor.tabs.tabText(0) == "Experiment parameters"
    assert "Raw YAML" in panel.editor.tabs.tabText(1)
    assert not panel.start_button.isEnabled()
    panel.workflow.setCurrentIndex(1)
    assert panel.tabs.currentWidget() is panel.editor
    assert not panel.start_button.isEnabled()
    panel.close()


def _init_workspace(root):
    initialize_workspace(root)
    return root


def test_training_recordings_validate_axes_without_improcess(qapp):
    import numpy as np
    records = [
        TrainingRecording("a", np.ones((4, 16, 16)), ("T", "Y", "X")),
        TrainingRecording("b", np.ones((4, 16, 16)), ("T", "Y", "X")),
    ]
    sources = _training_sources(records, [], "timelapse")
    assert [entry[0] for entry in sources] == ["a.tif", "b.tif"]
    with pytest.raises(ValueError, match="YX"):
        _training_sources(records, [], "single")


def test_r4_document_is_single_editor_state(qapp, tmp_path):
    """R4-A: raw and guided views update the same toolkit-independent object."""
    from lisai.application import TrainingConfigDocument

    settings = Settings.from_workspace(_init_workspace(tmp_path / "doc-ws"))
    editor = TrainingConfigEditorWidget(settings)
    draft = TrainingConfigDocument.from_text(
        "# retain this note\nexperiment:\n  exp_name: demo\n"
        "  task:\n    name: denoising_hdn\n    betaKL: 0.5\n"
    )
    assert editor.set_document(draft, confirm_discard=False)
    assert editor.document is draft
    assert editor.has_unsaved_changes
    editor.guided.fieldEdited.emit(("experiment", "task", "betaKL"), 0.4)
    assert "betaKL: 0.4" in editor.document.yaml_text
    assert editor.raw.toPlainText() == editor.document.yaml_text
    assert "# retain this note" in editor.document.yaml_text
    editor.raw.setPlainText("experiment: [broken")
    assert editor.document.yaml_text == "experiment: [broken"
    editor.reload_guided()
    assert not editor.guided.isEnabled()
    editor.raw.setPlainText("experiment:\n  exp_name: fixed\n")
    editor.reload_guided()
    assert editor.guided.isEnabled()
    editor.close()
