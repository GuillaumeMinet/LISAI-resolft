"""APP-2 standalone Qt integration checks; skipped only without a Qt binding."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from qtpy import QtWidgets
except Exception:
    pytest.skip("Qt binding not installed", allow_module_level=True)

from lisai.application.workspace import create_workspace
from lisai.config.settings import Settings
from lisai.qt.inference.controls import InferenceControlsWidget
from lisai.qt.model_manager import ModelManagerWidget


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


@pytest.fixture
def settings(tmp_path):
    return create_workspace(tmp_path / "ws")


def _register(settings):
    from datetime import datetime, timezone

    from lisai.infra.paths import Paths
    from lisai.promoted_models.registry import (
        PromotedModelRegistryEntry,
        register_promoted_model,
    )

    paths = Paths(settings)
    (paths.promoted_models_root() / "demo").mkdir(parents=True, exist_ok=True)
    register_promoted_model(
        name="demo", entry=PromotedModelRegistryEntry(
            source_run_id="ABC123", path="demo",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ), paths=paths,
    )


def test_control_is_standalone_and_overrides_are_sparse(qapp, settings):
    _register(settings)
    widget = InferenceControlsWidget(settings)
    assert widget.get_values()["model_name"] == ""
    with pytest.raises(ValueError, match="Choose"):
        widget.inference_selection()
    events = []
    widget.selectionChanged.connect(lambda: events.append("selection"))
    widget.optionsChanged.connect(lambda: events.append("options"))
    widget.select_model("demo")
    assert widget.inference_selection().model == "demo"
    assert widget.inference_selection().overrides.model_dump(exclude_unset=True) == {}
    widget.normalization_combo.setCurrentIndex(widget.normalization_combo.findData("manual"))
    widget.gain_spin.setValue(9.94)
    widget.offset_spin.setValue(-7.35)
    widget.tiling_combo.setCurrentText("512")
    widget.edge_combo.setCurrentIndex(widget.edge_combo.findData("black_output"))
    selected = widget.inference_selection()
    assert selected.overrides.model_dump(exclude_unset=True) == {
        "input": {"normalization": {"mode": "manual", "gain": 9.94, "offset": -7.35}},
        "inference": {"tiling_size": 512, "temporal_edge_mode": "black_output"},
    }
    assert events.count("options") >= 3
    widget.set_values({"model_source": "installed", "model_name": "demo",
                       "normalization_mode": "", "tiling_size": ""})
    assert widget.get_values()["model_name"] == "demo"
    assert widget.inference_selection().overrides.model_dump(exclude_unset=True) == {
        "inference": {"temporal_edge_mode": "black_output"},
    }
    widget.close()


def test_run_checkpoints_and_disappearing_selections(qapp, settings, monkeypatch):
    import lisai.application.catalogs as catalogs
    import lisai.qt.inference.controls as controls_module

    runs = [{"run_id": "ABC123", "label": "Example run [demo] (ABC123)",
             "checkpoints": {"best": False, "last": True}}]
    monkeypatch.setattr(catalogs, "list_usable_training_runs", lambda _: runs)
    monkeypatch.setattr(controls_module, "list_usable_training_runs", lambda _: runs)
    widget = InferenceControlsWidget(settings)
    widget.source_combo.setCurrentIndex(1)
    widget.select_model("ABC123", source="run")
    with pytest.raises(FileNotFoundError, match="best"):
        widget.inference_selection()
    widget.checkpoint_combo.setCurrentIndex(1)
    assert widget.inference_selection().identifier == "ABC123"
    assert widget.inference_selection().overrides.model_dump(exclude_unset=True) == {
        "checkpoint": {"best_or_last": "last"},
    }
    runs.clear()
    widget.refresh_catalog()
    assert "unavailable" in widget.model_combo.currentText()
    with pytest.raises(ValueError, match="available"):
        widget.inference_selection()
    widget.close()


def test_workspace_switch_does_not_carry_model_identity(qapp, settings, tmp_path):
    _register(settings)
    widget = InferenceControlsWidget(settings)
    widget.select_model("demo")
    assert widget.inference_selection().identifier == "demo"
    other = create_workspace(tmp_path / "other")
    widget.set_settings(other)
    assert widget.get_values()["model_name"] == ""
    with pytest.raises(ValueError, match="Choose"):
        widget.inference_selection()
    widget.close()


def test_model_manager_local_first_and_internal_navigation(qapp, settings, monkeypatch):
    import lisai.qt.model_manager as manager_module

    _register(settings)
    monkeypatch.setattr(manager_module, "get_installed_model_card", lambda *_: "# Demo card")
    # Constructing a manager must never fetch an online catalog or weights.
    monkeypatch.setattr(manager_module, "get_installed_model_names", lambda _: ["demo"])
    widget = ModelManagerWidget(settings)
    assert widget.model_list.count() == 1
    assert widget.selected_model_name() == "demo"
    assert "Demo card" in widget.details.toPlainText()
    assert not widget.install_button.isEnabled()
    events = []
    widget.backRequested.connect(lambda: events.append("back"))
    widget.back_button.click()
    assert events == ["back"]
    with pytest.raises(RuntimeError, match="active model job"):
        widget._job = object()
        widget.set_settings(create_workspace(settings.WORKSPACE_ROOT.parent / "other"))
    widget._job = None
    widget.close()


def test_model_manager_async_catalog_and_install(qapp, settings, monkeypatch):
    """No network or weights: verify worker signals and install refresh."""
    from pathlib import Path
    from types import SimpleNamespace

    import lisai.promoted_models.catalog as catalog
    import lisai.promoted_models.download as download
    import lisai.promoted_models.install as install

    monkeypatch.setattr(catalog, "list_models", lambda: [SimpleNamespace(
        name="online_demo", task="denoising_hdn", description="Test model"
    )])
    monkeypatch.setattr(download, "download_model", lambda *args, **kwargs: SimpleNamespace(
        archive_path=Path("online_demo.lisai.zip")
    ))
    monkeypatch.setattr(install, "install_model_archive", lambda *args, **kwargs: None)
    import lisai.qt.model_manager as manager_module
    monkeypatch.setattr(manager_module, "get_installed_model_card", lambda *_: "Test card")

    widget = ModelManagerWidget(settings)
    widget.refresh_online()
    worker = widget._job
    assert worker is not None and worker.wait(5000)
    qapp.processEvents()
    qapp.processEvents()
    assert "online_demo" in widget._catalog
    assert widget.model_list.count() >= 1
    assert widget._job is None

    for i in range(widget.model_list.count()):
        if widget.model_list.item(i).data(manager_module.QtCore.Qt.UserRole) == "online_demo":
            widget.model_list.setCurrentRow(i)
            break
    installed = []
    widget.modelInstalled.connect(installed.append)
    widget.install_selected()
    worker = widget._job
    assert worker is not None and worker.wait(5000)
    qapp.processEvents()
    qapp.processEvents()
    assert installed == ["online_demo"]
    assert "online_demo" in widget._installed
    assert widget._job is None
    widget.close()
