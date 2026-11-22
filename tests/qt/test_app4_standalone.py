"""APP-4 integration checks. Execute with Qt + pyqtgraph, ideally offscreen."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from qtpy import QtCore, QtWidgets
    import pyqtgraph  # noqa: F401
except (ImportError, RuntimeError):
    pytest.skip("Qt binding or pyqtgraph unavailable", allow_module_level=True)

from lisai.application.workspace import create_workspace
from lisai.app.inference_page import StandaloneInferencePage
from lisai.app.main_window import LisaiMainWindow, load_app_settings
from lisai.evaluation.inference.results import PredictionInfo
from lisai.infra.paths import Paths
from lisai.promoted_models.registry import PromotedModelRegistryEntry, register_promoted_model


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def settings(tmp_path):
    return create_workspace(tmp_path / "lisai_workspace")


def _register_model(settings, name="demo"):
    paths = Paths(settings)
    (paths.promoted_models_root() / name).mkdir(parents=True, exist_ok=True)
    register_promoted_model(
        name=name,
        entry=PromotedModelRegistryEntry(
            source_run_id="ABC123", path=name,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        paths=paths,
    )


def _run_events(qapp, thread):
    assert thread.wait(5000), "Prediction worker did not finish"
    for _ in range(10):
        qapp.processEvents()


def test_standalone_composes_reusable_training_and_inference(qapp, settings):
    window = LisaiMainWindow(settings)
    assert window.tabs.count() == 2
    assert [window.tabs.tabText(i) for i in range(2)] == ["Training", "Inference"]
    assert window.tabs.widget(0) is window.training
    assert window.tabs.widget(1) is window.inference
    assert str(settings.WORKSPACE_ROOT) in window.workspace_label.text()
    window.tabs.setCurrentIndex(1)
    window.tabs.setCurrentIndex(0)
    assert window.training.session is window.session
    window.close()


def test_load_predict_display_and_save_without_reexecution(qapp, settings, tmp_path, monkeypatch):
    import lisai

    _register_model(settings)
    raw = np.arange(6 * 13 * 17, dtype=np.float32).reshape(6, 13, 17)
    file = tmp_path / "time_series.tiff"
    tifffile.imwrite(file, raw, metadata={"axes": "TYX"}, photometric="minisblack")
    calls = []

    def fake_predict(model, data, **kwargs):
        calls.append((model, kwargs))
        np.testing.assert_array_equal(data, raw)
        assert kwargs["return_info"] is True
        return np.full((2, 26, 34), 8.0, dtype=np.float32), PredictionInfo((2, 3))

    monkeypatch.setattr(lisai, "predict", fake_predict)
    page = StandaloneInferencePage(settings)
    assert not page.save_button.isEnabled()
    page.load_tiff(file)
    page.controls.select_model("demo")
    assert not page.comparison.has_prediction
    page.run_inference()
    worker = page._worker
    assert worker is not None
    assert not page.run_button.isEnabled()
    _run_events(qapp, worker)
    assert page._worker is None
    assert len(calls) == 1
    assert page.save_button.isEnabled()
    assert page.comparison.current_prediction_index() is None
    page.comparison.set_frame_index(2)
    assert page.comparison.current_prediction_index() == 0
    assert page.comparison.prediction_viewer.image.shape == (26, 34)

    result = tmp_path / "pred.tif"
    page.save_prediction_to(result)
    with tifffile.TiffFile(result) as tif:
        assert tif.series[0].axes == "TYX"
        np.testing.assert_array_equal(tif.asarray(), np.full((2, 26, 34), 8.0))
    assert len(calls) == 1  # saving is never a second inference call
    with pytest.raises(FileExistsError):
        page.save_prediction_to(result)
    page.controls.normalization_combo.setCurrentIndex(
        page.controls.normalization_combo.findData("image")
    )
    assert not page.save_button.isEnabled()
    assert not page.comparison.has_prediction
    page.close()


def test_failed_prediction_keeps_input_but_cannot_save(qapp, settings, tmp_path, monkeypatch):
    import lisai

    _register_model(settings)
    file = tmp_path / "raw.tif"
    tifffile.imwrite(file, np.ones((9, 11), np.float32),
                     metadata={"axes": "YX"}, photometric="minisblack")

    def fails(*args, **kwargs):
        raise RuntimeError("GPU out of memory")

    monkeypatch.setattr(lisai, "predict", fails)
    page = StandaloneInferencePage(settings)
    page.load_tiff(file)
    page.controls.select_model("demo")
    page.run_inference()
    worker = page._worker
    _run_events(qapp, worker)
    assert "GPU out of memory" in page.status.text()
    assert page.comparison.input_viewer.image is not None
    assert not page.comparison.has_prediction
    assert not page.save_button.isEnabled()
    page.close()



def test_model_manager_navigation_does_not_invalidate_prediction(qapp, settings, tmp_path):
    _register_model(settings)
    source = tmp_path / "image.tif"
    tifffile.imwrite(source, np.ones((7, 9), np.float32),
                     metadata={"axes": "YX"}, photometric="minisblack")
    page = StandaloneInferencePage(settings)
    page.load_tiff(source)
    page.controls.select_model("demo")
    page.comparison.set_prediction(np.ones((7, 9), np.float32) * 2,
                                   info=PredictionInfo((0,)))
    assert page.comparison.has_prediction
    page.show_model_manager()
    page.show_inference()
    assert page.comparison.has_prediction
    assert page.controls.current_values()["model_name"] == "demo"
    page.close()


def test_internal_model_manager_and_workspace_switch_clear_state(qapp, settings, tmp_path):
    _register_model(settings)
    page = StandaloneInferencePage(settings)
    page.controls.select_model("demo")
    page.show_model_manager()
    assert not page.model_manager.is_busy()
    assert page.stack.currentIndex() == 1
    page.show_inference()
    assert page.stack.currentIndex() == 0
    other = create_workspace(tmp_path / "other")
    assert load_app_settings(other.WORKSPACE_ROOT).WORKSPACE_ROOT == other.WORKSPACE_ROOT
    page.set_settings(other)
    assert page.controls.current_values()["model_name"] == ""
    assert not page.save_button.isEnabled()
    page.close()


def test_workspace_switch_constructs_training_with_widget_parent(qapp, settings, tmp_path, monkeypatch):
    other = create_workspace(tmp_path / "next_workspace")
    window = LisaiMainWindow(settings)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory",
        lambda *args, **kwargs: str(other.WORKSPACE_ROOT),
    )
    window.tabs.setCurrentIndex(1)
    window.change_workspace()
    assert window.settings.WORKSPACE_ROOT == other.WORKSPACE_ROOT
    assert window.tabs.currentIndex() == 1
    assert window.training.parent() is window.tabs
    assert window.training.data_page.recordings == ()
    window.close()
