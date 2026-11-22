"""APP-3 offscreen Qt tests; run with a Qt binding and pyqtgraph installed."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from qtpy import QtWidgets
    import pyqtgraph  # noqa: F401
except (ImportError, RuntimeError):
    pytest.skip("Qt binding or pyqtgraph not installed", allow_module_level=True)

from lisai.evaluation.inference.results import PredictionInfo
from lisai.qt.inference import ImageComparisonWidget, ImageViewer2D


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_viewer_row_major_geometry_independent_levels_and_original_array(qapp):
    # Asymmetric shape with unique pixel at x=9/y=1 detects accidental transpose.
    array = np.zeros((7, 11), dtype=np.float32)
    array[1, 9] = 500
    original = array.copy()
    view = ImageViewer2D()
    view.set_image(array)
    assert view.image is array
    np.testing.assert_array_equal(view.image_item.image, array)
    assert view.image_item.axisOrder == "row-major"
    assert view.view_box.state["yInverted"] is True
    assert view.image_item.width() == 11
    assert view.image_item.height() == 7
    view.set_levels(-20, 550)
    assert view.levels == (-20, 550)
    assert not view.auto_checkbox.isChecked()
    np.testing.assert_array_equal(array, original)
    view.auto_checkbox.setChecked(True)
    assert view.levels[1] > view.levels[0]
    view.reset_view()
    view.clear("Empty")
    assert view.image is None
    assert view.empty_label.text() == "Empty"
    view.close()


def test_comparison_drop_missing_edges_and_canonical_input_slider(qapp):
    input_array = np.stack([np.full((9, 13), t, np.float32) for t in range(10)])
    prediction = np.stack([np.full((18, 26), 100 + t, np.float32) for t in range(6)])
    comparison = ImageComparisonWidget()
    comparison.set_input(input_array, axes="TYX")
    comparison.set_prediction(prediction, info=PredictionInfo(tuple(range(2, 8))))
    assert comparison.current_frame_index == 0
    assert comparison.current_prediction_index() is None
    assert "No prediction" in comparison.prediction_status.text()
    comparison.set_frame_index(2)
    assert comparison.current_prediction_index() == 0
    assert comparison.prediction_viewer.image[0, 0] == 100
    assert comparison.input_viewer.image[0, 0] == 2
    assert comparison.frame_spin.value() == 3
    comparison.frame_slider.setValue(7)
    assert comparison.current_prediction_index() == 5
    assert comparison.prediction_viewer.image[0, 0] == 105
    comparison.set_frame_index(9)
    assert comparison.current_prediction_index() is None
    assert comparison.prediction_viewer.image is None
    comparison.close()


def test_comparison_black_output_and_black_input_and_no_array_mutation(qapp):
    raw = np.arange(10 * 3 * 5, dtype=np.float32).reshape(10, 3, 5)
    pred = raw.copy()
    pred[0] = 0
    pred[9] = 0
    original = pred.copy()
    comparison = ImageComparisonWidget()
    comparison.set_input(raw, axes="TYX")
    comparison.set_prediction(pred, info=PredictionInfo(tuple(range(10)), (0, 9)))
    assert comparison.current_prediction_index() == 0
    assert "black output" in comparison.prediction_status.text().lower()
    np.testing.assert_array_equal(comparison.prediction_viewer.image, pred[0])
    comparison.set_frame_index(1)
    assert "black output" not in comparison.prediction_status.text().lower()
    comparison.set_prediction(pred, info=PredictionInfo(tuple(range(10))))
    comparison.set_frame_index(9)
    assert "black output" not in comparison.prediction_status.text().lower()
    np.testing.assert_array_equal(pred, original)
    comparison.close()


def test_comparison_yx_rejects_bad_axes_and_alignment_without_losing_valid_result(qapp):
    raw = np.arange(40, dtype=np.uint16).reshape(5, 8)
    prediction = raw * 2
    comparison = ImageComparisonWidget()
    with pytest.raises(ValueError, match="YX or TYX"):
        comparison.set_input(raw[..., None], axes="YXC")
    comparison.set_input(raw, axes="YX")
    comparison.set_prediction(prediction, info=PredictionInfo((0,)))
    assert comparison.current_prediction_index() == 0
    with pytest.raises(ValueError, match="source_frame_indices"):
        comparison.set_prediction(prediction, info=PredictionInfo((1,)))
    assert comparison.has_prediction
    assert comparison.current_prediction_index() == 0
    with pytest.raises(ValueError, match="placeholder"):
        comparison.set_prediction(prediction, info=PredictionInfo((0,), (0,)))
    assert comparison.has_prediction
    comparison.invalidate_prediction("Model settings changed")
    assert comparison.prediction_viewer.image is None
    assert "Model settings changed" in comparison.prediction_status.text()
    comparison.close()


def test_new_input_invalidates_old_prediction_and_independent_manual_contrast(qapp):
    raw = np.ones((4, 7), np.float32)
    widget = ImageComparisonWidget()
    widget.set_input(raw, axes="YX")
    widget.set_prediction(raw * 20, info=PredictionInfo((0,)))
    widget.input_viewer.set_levels(0, 1)
    widget.prediction_viewer.set_levels(2, 24)
    assert widget.input_viewer.levels != widget.prediction_viewer.levels
    widget.set_input(raw + 1, axes="YX")
    assert not widget.has_prediction
    assert widget.prediction_viewer.image is None
    widget.close()
