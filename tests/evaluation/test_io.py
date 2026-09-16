from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread

from lisai.evaluation.data import EvalItem
from lisai.evaluation.io import EvalItemOutputWriter


def _make_timelapse_item(tmp_path: Path) -> EvalItem:
    return EvalItem(
        name="stack_a",
        inp_path=tmp_path / "stack_a.tif",
        gt_path=None,
        split="test",
        file_index=0,
        data_format="timelapse",
        sample_count=2,
        time_indices=(0, 1),
    )


def test_eval_item_output_writer_stacks_timelapse_outputs(tmp_path: Path):
    item = _make_timelapse_item(tmp_path)
    writer = EvalItemOutputWriter(item=item, save_folder=tmp_path)

    for sample_index, value in ((1, 7.0), (0, 3.0)):
        writer.add(
            sample_index=sample_index,
            tosave={"pred": np.full((1, 1, 2, 3), value, dtype=np.float32)},
        )

    writer.flush()

    saved = imread(tmp_path / "stack_a_pred.tif")
    assert saved.shape == (2, 2, 3)
    assert np.all(saved[0] == 3.0)
    assert np.all(saved[1] == 7.0)


def test_eval_item_output_writer_stacks_lvae_timelapse_outputs(tmp_path: Path):
    item = _make_timelapse_item(tmp_path)
    writer = EvalItemOutputWriter(item=item, save_folder=tmp_path)

    for sample_index, value in ((1, 7.0), (0, 3.0)):
        writer.add(
            sample_index=sample_index,
            tosave={
                "pred": np.full((2, 3), value, dtype=np.float32),
                "samples": np.full((4, 2, 3), value, dtype=np.float32),
            },
        )

    writer.flush()

    pred = imread(tmp_path / "stack_a_pred.tif")
    assert pred.shape == (2, 2, 3)
    assert np.all(pred[0] == 3.0)
    assert np.all(pred[1] == 7.0)

    samples = imread(tmp_path / "stack_a_samples.tif")
    assert samples.shape == (4, 2, 2, 3)
    assert np.all(samples[:, 0] == 3.0)
    assert np.all(samples[:, 1] == 7.0)
