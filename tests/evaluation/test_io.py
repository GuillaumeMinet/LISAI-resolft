from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread

from lisai.evaluation.data import EvalItem
from lisai.evaluation.io import EvalItemOutputWriter, load_outputs_manifest, save_outputs_manifest


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
        input_id="inp/test/stack_a.tif",
        source_axis="time",
        source_indices=(0, 1),
    )


def test_eval_item_output_writer_stacks_timelapse_outputs(tmp_path: Path):
    item = _make_timelapse_item(tmp_path)
    writer = EvalItemOutputWriter(item=item, save_folder=tmp_path)

    for sample_index, value in ((1, 7.0), (0, 3.0)):
        writer.add(
            sample_index=sample_index,
            tosave={"pred": np.full((1, 1, 2, 3), value, dtype=np.float32)},
        )

    record = writer.flush()

    saved = imread(tmp_path / "stack_a_pred.tif")
    assert saved.shape == (2, 2, 3)
    assert np.all(saved[0] == 3.0)
    assert np.all(saved[1] == 7.0)
    assert record == {
        "name": "stack_a",
        "input_id": "inp/test/stack_a.tif",
        "gt_id": None,
        "source_axis": "time",
        "source_indices": (0, 1),
        "outputs": {
            "pred": {
                "files": ["stack_a_pred.tif"],
                "sample_axis": 0,
            }
        },
    }


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


def test_eval_item_output_writer_keeps_unknown_source_indices(tmp_path: Path):
    item = EvalItem(
        name="stack_a",
        inp_path=tmp_path / "stack_a.tif",
        gt_path=None,
        split="test",
        file_index=0,
        data_format="timelapse",
        sample_count=2,
        time_indices=(None, None),
        input_id="inp/test/stack_a.tif",
        source_axis="time",
        source_indices="unknown",
    )
    writer = EvalItemOutputWriter(item=item, save_folder=tmp_path)
    writer.add(sample_index=0, tosave={"pred": np.ones((2, 3), dtype=np.float32)})
    writer.add(sample_index=1, tosave={"pred": np.ones((2, 3), dtype=np.float32)})

    record = writer.flush()

    assert record["source_indices"] == "unknown"


def test_eval_item_output_writer_records_only_persisted_time_indices(tmp_path: Path):
    item = EvalItem(
        name="stack_a",
        inp_path=tmp_path / "stack_a.tif",
        gt_path=None,
        split="test",
        file_index=0,
        data_format="timelapse",
        sample_count=3,
        time_indices=(1, 2, 3),
        input_id="inp/test/stack_a.tif",
        source_axis="time",
        source_indices=(1, 2, 3),
    )
    writer = EvalItemOutputWriter(item=item, save_folder=tmp_path)
    writer.add(sample_index=0, tosave={"pred": np.ones((2, 3), dtype=np.float32)})
    writer.add(sample_index=1, tosave={"pred": np.ones((2, 3), dtype=np.float32)})

    record = writer.flush()

    assert record["source_indices"] == (1, 2)


def test_eval_item_output_writer_records_nonstacked_snr_files_in_sample_order(tmp_path: Path):
    item = EvalItem(
        name="img_a",
        inp_path=tmp_path / "img_a.tif",
        gt_path=None,
        split="test",
        file_index=0,
        data_format="mltpl_snr",
        sample_count=2,
        time_indices=(None, None),
        input_id="inp/test/img_a.tif",
        source_axis="snr",
        source_indices=(4, 1),
    )
    writer = EvalItemOutputWriter(item=item, save_folder=tmp_path)
    writer.add(sample_index=0, tosave={"pred": np.ones((2, 3), dtype=np.float32)})
    writer.add(sample_index=1, tosave={"pred": np.ones((2, 3), dtype=np.float32) * 2})

    record = writer.flush()

    assert record["source_indices"] == (4, 1)
    assert record["outputs"]["pred"] == {
        "files": ["img_a_0_pred.tif", "img_a_1_pred.tif"],
        "sample_axis": None,
    }


def test_outputs_manifest_roundtrip(tmp_path: Path):
    items = [
        {
            "name": "stack_a",
            "input_id": "inp/test/stack_a.tif",
            "gt_id": None,
            "source_axis": "time",
            "source_indices": (2, 3),
            "outputs": {
                "pred": {
                    "files": ["stack_a_pred.tif"],
                    "sample_axis": 0,
                }
            },
        }
    ]

    manifest_path = save_outputs_manifest(tmp_path, items)
    loaded = load_outputs_manifest(tmp_path)

    assert manifest_path == tmp_path / "outputs_manifest.yaml"
    assert loaded["version"] == 1
    assert loaded["items"][0]["source_indices"] == [2, 3]

    manifest_text = manifest_path.read_text()
    assert "source_indices: [2, 3]" in manifest_text


def test_outputs_manifest_compacts_long_contiguous_source_indices(tmp_path: Path):
    items = [
        {
            "name": "stack_a",
            "input_id": "inp/test/stack_a.tif",
            "gt_id": None,
            "source_axis": "time",
            "source_indices": tuple(range(2, 42)),
            "outputs": {},
        }
    ]

    manifest_path = save_outputs_manifest(tmp_path, items)
    loaded = load_outputs_manifest(manifest_path)

    manifest_text = manifest_path.read_text()
    assert "source_indices: {start: 2, end: 41}" in manifest_text
    assert loaded["items"][0]["source_indices"] == list(range(2, 42))


def test_outputs_manifest_keeps_irregular_source_indices_explicit(tmp_path: Path):
    items = [
        {
            "name": "stack_a",
            "input_id": "inp/test/stack_a.tif",
            "gt_id": None,
            "source_axis": "snr",
            "source_indices": (0, 2, 4, 6, 8),
            "outputs": {},
        }
    ]

    manifest_path = save_outputs_manifest(tmp_path, items)
    loaded = load_outputs_manifest(tmp_path)

    manifest_text = manifest_path.read_text()
    assert "source_indices: [0, 2, 4, 6, 8]" in manifest_text
    assert loaded["items"][0]["source_indices"] == [0, 2, 4, 6, 8]
