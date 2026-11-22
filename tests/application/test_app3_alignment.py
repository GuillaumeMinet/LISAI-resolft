"""APP-3: frame mapping is verifiable even without Qt or pyqtgraph."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lisai.evaluation.inference.results import PredictionInfo
from lisai.qt.inference.alignment import FrameAlignment


def test_single_yx_frame_alignment():
    mapping = FrameAlignment.from_info(
        PredictionInfo((0,)), input_frames=1, output_frames=1,
    )
    assert mapping.source_to_output == (0,)
    assert not mapping.is_black_placeholder(0)


def test_drop_edges_maps_to_original_input_time_not_output_position():
    mapping = FrameAlignment.from_info(
        PredictionInfo((2, 3, 4, 5, 6, 7)), input_frames=10, output_frames=6,
    )
    assert mapping.source_to_output == (None, None, 0, 1, 2, 3, 4, 5, None, None)
    assert mapping.prediction_index(2) == 0
    assert mapping.prediction_index(7) == 5
    assert mapping.prediction_index(0) is None


def test_black_output_and_black_input_are_different():
    full = tuple(range(10))
    black_output = FrameAlignment.from_info(
        PredictionInfo(full, (0, 1, 8, 9)), input_frames=10, output_frames=10,
    )
    assert black_output.prediction_index(0) == 0
    assert black_output.is_black_placeholder(0)
    assert black_output.is_black_placeholder(9)
    assert not black_output.is_black_placeholder(3)
    black_input = FrameAlignment.from_info(
        PredictionInfo(full), input_frames=10, output_frames=10,
    )
    assert not any(black_input.is_black_placeholder(t) for t in full)


@pytest.mark.parametrize("info,input_frames,output_frames,expected", [
    (PredictionInfo((0,)), 10, 2, "count"),
    (PredictionInfo((0, 0)), 10, 2, "duplicate"),
    (PredictionInfo((0, 10)), 10, 2, "out of bounds"),
    (PredictionInfo((-1, 1)), 10, 2, "out of bounds"),
    (PredictionInfo((0, True)), 10, 2, "integers"),
    (PredictionInfo((0, 1), (2,)), 10, 2, "out of bounds"),
    (PredictionInfo((0, 1), (1, 1)), 10, 2, "duplicate"),
    (PredictionInfo((0, 1), (True,)), 10, 2, "integers"),
])
def test_invalid_mapping_fails_explicitly(info, input_frames, output_frames, expected):
    with pytest.raises(ValueError, match=expected):
        FrameAlignment.from_info(
            info, input_frames=input_frames, output_frames=output_frames,
        )


def test_frame_index_count_validation():
    with pytest.raises(ValueError, match="positive"):
        FrameAlignment.from_info(PredictionInfo((0,)), input_frames=0, output_frames=1)
    with pytest.raises(TypeError, match="PredictionInfo"):
        FrameAlignment.from_info(None, input_frames=1, output_frames=1)


def test_optional_image_viewer_imports_are_lazy():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-c", (
            "import sys, lisai, lisai.application, lisai.qt.inference; "
            "from lisai.qt.inference.alignment import FrameAlignment; "
            "assert 'qtpy' not in sys.modules; "
            "assert 'pyqtgraph' not in sys.modules"
        )],
        check=True, cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
