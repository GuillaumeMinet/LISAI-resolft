from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tifffile import imwrite

from graphs.utils.eval_folder import RunEvaluation
import graphs.utils.eval_outputs as eval_outputs_mod
from graphs.utils.eval_outputs import (
    UnalignedComparisonWarning,
    prepare_comparison_outputs,
)
from lisai.evaluation.io import save_outputs_manifest


def _evaluation(tmp_path, name: str) -> RunEvaluation:
    run_dir = tmp_path / name
    folder = run_dir / "evaluations" / "training_test" / "best_epoch_1"
    folder.mkdir(parents=True)
    run = SimpleNamespace(run_dir=run_dir, name=name)
    return RunEvaluation(run=run, folder=folder)


def _write_timelapse_item(evaluation, *, indices, values, input_id="cell.tif", source_axis="time"):
    folder = evaluation.folder
    values = np.asarray(values, dtype=np.float32)
    gt = values[:, None, None]
    pred = (values + 100)[:, None, None]
    imwrite(folder / "cell_gt.tif", gt, photometric="minisblack")
    imwrite(folder / "cell_pred.tif", pred, photometric="minisblack")
    save_outputs_manifest(
        folder,
        [
            {
                "name": "cell",
                "input_id": input_id,
                "gt_id": input_id,
                "source_axis": source_axis,
                "source_indices": indices,
                "outputs": {
                    "gt": {"files": ["cell_gt.tif"], "sample_axis": 0},
                    "pred": {"files": ["cell_pred.tif"], "sample_axis": 0},
                },
            }
        ],
    )


def _use_canonical_data_dir(monkeypatch, data_dir):
    monkeypatch.setattr(
        eval_outputs_mod,
        "_canonical_dataset_data_dir",
        lambda **_kwargs: data_dir,
    )


def test_comparison_aligns_common_timepoints(tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")
    _write_timelapse_item(first, indices=[0, 1, 2, 3, 4], values=[0, 1, 2, 3, 4])
    _write_timelapse_item(second, indices=[1, 2, 3], values=[11, 12, 13])

    comparison = prepare_comparison_outputs([first, second])

    first_item = comparison.items_for(first)[0]
    second_item = comparison.items_for(second)[0]
    assert first_item.source_indices == (1, 2, 3)
    assert second_item.source_indices == (1, 2, 3)

    first_gt, _ = first_item.load_gt_pred()
    second_gt, _ = second_item.load_gt_pred()
    assert [float(sample) for sample in first_gt] == [1.0, 2.0, 3.0]
    assert [float(sample) for sample in second_gt] == [11.0, 12.0, 13.0]


def test_unaligned_comparison_warns_and_keeps_all_samples(tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")
    _write_timelapse_item(first, indices=[0, 1, 2], values=[0, 1, 2])
    _write_timelapse_item(second, indices=[1, 2], values=[11, 12])

    with pytest.warns(UnalignedComparisonWarning, match="alignment is disabled"):
        comparison = prepare_comparison_outputs([first, second], align=False)

    assert comparison.items_for(first)[0].positions == (0, 1, 2)
    assert comparison.items_for(second)[0].positions == (0, 1)


def test_unknown_source_indices_require_explicit_opt_out(tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")
    _write_timelapse_item(first, indices=[0, 1, 2], values=[0, 1, 2])
    _write_timelapse_item(second, indices="unknown", values=[11, 12, 13])

    with pytest.raises(ValueError, match="time source indices are unknown"):
        prepare_comparison_outputs([first, second])


def test_alignment_uses_input_and_gt_identity(tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")
    _write_timelapse_item(first, indices=[0, 1], values=[0, 1], input_id="cell_a.tif")
    _write_timelapse_item(second, indices=[0, 1], values=[10, 11], input_id="cell_b.tif")

    with pytest.raises(ValueError, match="no common input/GT items"):
        prepare_comparison_outputs([first, second])



def test_load_aux_data_uses_registered_auxiliary_and_source_indices(monkeypatch, tmp_path):
    evaluation = _evaluation(tmp_path, "run_aux")
    _write_timelapse_item(evaluation, indices=[1, 3], values=[1, 3], input_id="low/cell.tif")

    data_dir = tmp_path / "evaluation_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    (data_dir / "conf").mkdir(parents=True)
    imwrite(data_dir / "conf" / "cell.tif", np.arange(5, dtype=np.float32)[:, None, None], photometric="minisblack")

    (evaluation.folder / "evaluation.yaml").write_text(
        f"""version: 1
dataset:
  name: gag_live_upsamp_eval
  usage: evaluation
  data_type: recon
  data_dir: {data_dir.as_posix()}
  input: low
"""
    )
    registry_info = {
        "outputs": {
            "recon": [
                {"key": "inp", "path": "low", "role": "inp", "axes": "YX"},
                {"key": "gt", "path": "high", "role": "gt", "axes": "YX"},
                {"key": "conf", "path": "conf", "role": "aux", "axes": "TYX"},
            ]
        }
    }
    monkeypatch.setattr(eval_outputs_mod, "load_dataset_info", lambda *_args, **_kwargs: registry_info)

    comparison = prepare_comparison_outputs([evaluation])
    aux_frames = comparison.items_for(evaluation)[0].load_aux_data("conf")

    assert [float(frame) for frame in aux_frames] == [1.0, 3.0]


def test_load_aux_data_lists_registered_auxiliary_names(monkeypatch, tmp_path):
    evaluation = _evaluation(tmp_path, "run_aux")
    _write_timelapse_item(evaluation, indices=[0], values=[0], input_id="low/cell.tif")

    data_dir = tmp_path / "evaluation_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    (evaluation.folder / "evaluation.yaml").write_text(
        f"""version: 1
dataset:
  name: gag_live_upsamp_eval
  usage: evaluation
  data_type: recon
  data_dir: {data_dir.as_posix()}
  input: low
"""
    )
    registry_info = {"outputs": {"recon": [{"key": "conf", "path": "conf", "role": "aux", "axes": "YX"}]}}
    monkeypatch.setattr(eval_outputs_mod, "load_dataset_info", lambda *_args, **_kwargs: registry_info)

    item = prepare_comparison_outputs([evaluation]).items_for(evaluation)[0]
    with pytest.raises(KeyError, match="Available auxiliary outputs: conf"):
        item.load_aux_data("missing")


def test_external_prediction_manifest_can_load_gt_from_registered_dataset(monkeypatch, tmp_path):
    evaluation = _evaluation(tmp_path, "external_run")
    folder = evaluation.folder
    data_dir = tmp_path / "evaluation_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    (data_dir / "low").mkdir(parents=True)
    (data_dir / "high").mkdir(parents=True)
    imwrite(data_dir / "low" / "cell.tif", np.ones((4, 5), dtype=np.float32))
    imwrite(data_dir / "high" / "cell.tif", np.full((4, 5), 7, dtype=np.float32))
    imwrite(folder / "cell_pred.tif", np.full((4, 5), 9, dtype=np.float32))
    save_outputs_manifest(
        folder,
        [
            {
                "name": "cell",
                "input_id": "low/cell.tif",
                "gt_id": "high/cell.tif",
                "outputs": {"pred": {"files": ["cell_pred.tif"], "sample_axis": None}},
            }
        ],
    )
    (folder / "evaluation.yaml").write_text(
        f"""version: 1
dataset:
  name: eval_ds
  usage: evaluation
  data_type: recon
  data_dir: {data_dir.as_posix()}
  input: low
  eval_gt: high
"""
    )
    monkeypatch.setattr(
        eval_outputs_mod,
        "load_dataset_info",
        lambda *_args, **_kwargs: {
            "outputs": {
                "recon": [
                    {"key": "inp", "path": "low", "role": "inp", "axes": "YX"},
                    {"key": "gt", "path": "high", "role": "gt", "axes": "YX"},
                ]
            }
        },
    )

    comparison = prepare_comparison_outputs([evaluation])
    gt, pred = comparison.items_for(evaluation)[0].load_gt_pred()
    assert np.all(gt[0] == 7)
    assert np.all(pred[0] == 9)


def test_shared_dataset_gt_is_identical_for_lisai_and_external_runs(monkeypatch, tmp_path):
    lisai_eval = _evaluation(tmp_path, "lisai_run")
    external_eval = _evaluation(tmp_path, "external_run")

    data_dir = tmp_path / "training_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    (data_dir / "inp").mkdir(parents=True)
    (data_dir / "gt").mkdir(parents=True)
    imwrite(data_dir / "inp" / "cell.tif", np.ones((4, 5), dtype=np.float32))
    imwrite(data_dir / "gt" / "cell.tif", np.full((4, 5), 7, dtype=np.float32))

    imwrite(lisai_eval.folder / "cell_gt.tif", np.full((4, 5), -3, dtype=np.float32))
    imwrite(lisai_eval.folder / "cell_pred.tif", np.full((4, 5), 10, dtype=np.float32))
    save_outputs_manifest(
        lisai_eval.folder,
        [{
            "name": "cell",
            "input_id": "inp/cell.tif",
            "gt_id": "gt/cell.tif",
            "outputs": {
                "gt": {"files": ["cell_gt.tif"], "sample_axis": None},
                "pred": {"files": ["cell_pred.tif"], "sample_axis": None},
            },
        }],
    )
    imwrite(external_eval.folder / "cell_pred.tif", np.full((4, 5), 11, dtype=np.float32))
    save_outputs_manifest(
        external_eval.folder,
        [{
            "name": "cell",
            "input_id": "inp/cell.tif",
            "gt_id": "gt/cell.tif",
            "outputs": {"pred": {"files": ["cell_pred.tif"], "sample_axis": None}},
        }],
    )

    for evaluation in (lisai_eval, external_eval):
        (evaluation.folder / "evaluation.yaml").write_text(
            f"""version: 1
dataset:
  name: vim_fixed_multi_snr
  usage: training
  data_type: null
  split: test
  data_dir: {data_dir.as_posix()}
  input: inp
  eval_gt: gt
"""
        )

    registry_info = {
        "outputs": {
            "recon": [
                {"key": "inp", "path": "inp", "role": "inp", "axes": "YX"},
                {"key": "gt", "path": "gt", "role": "gt", "axes": "YX"},
            ]
        }
    }
    monkeypatch.setattr(eval_outputs_mod, "load_dataset_info", lambda *_args, **_kwargs: registry_info)

    per_run = prepare_comparison_outputs([lisai_eval, external_eval])
    lisai_gt, _ = per_run.items_for(lisai_eval)[0].load_gt_pred()
    external_gt, _ = per_run.items_for(external_eval)[0].load_gt_pred()
    assert np.all(lisai_gt[0] == -3)
    assert np.all(external_gt[0] == 7)

    shared = prepare_comparison_outputs([lisai_eval, external_eval], gt_reference="dataset")
    lisai_gt, _ = shared.load_gt_pred(shared.items_for(lisai_eval)[0])
    external_gt, _ = shared.load_gt_pred(shared.items_for(external_eval)[0])
    assert np.all(lisai_gt[0] == 7)
    assert np.all(external_gt[0] == 7)


def test_shared_dataset_gt_ignores_stale_recorded_roots(monkeypatch, tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")

    data_dir = tmp_path / "current_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    (data_dir / "inp").mkdir(parents=True)
    (data_dir / "gt").mkdir(parents=True)
    imwrite(data_dir / "inp" / "cell.tif", np.ones((4, 5), dtype=np.float32))
    imwrite(data_dir / "gt" / "cell.tif", np.full((4, 5), 7, dtype=np.float32))

    for evaluation, recorded_root, pred_value in (
        (first, tmp_path / "old_root_a" / "preprocess" / "recon", 10),
        (second, tmp_path / "old_root_b" / "preprocess" / "recon", 11),
    ):
        imwrite(evaluation.folder / "cell_pred.tif", np.full((4, 5), pred_value, dtype=np.float32))
        save_outputs_manifest(
            evaluation.folder,
            [{
                "name": "cell",
                "input_id": "inp/cell.tif",
                "gt_id": "gt/cell.tif",
                "outputs": {"pred": {"files": ["cell_pred.tif"], "sample_axis": None}},
            }],
        )
        (evaluation.folder / "evaluation.yaml").write_text(
            f"""version: 1
dataset:
  name: vim_fixed_multi_snr
  usage: training
  data_type: recon
  split: test
  data_dir: {recorded_root.as_posix()}
  input: inp
  eval_gt: gt
"""
        )

    registry_info = {
        "outputs": {
            "recon": [
                {"key": "inp", "path": "inp", "role": "inp", "axes": "YX"},
                {"key": "gt", "path": "gt", "role": "gt", "axes": "YX"},
            ]
        }
    }
    monkeypatch.setattr(eval_outputs_mod, "load_dataset_info", lambda *_args, **_kwargs: registry_info)

    comparison = prepare_comparison_outputs([first, second], gt_reference="dataset")

    first_gt, first_pred = comparison.load_gt_pred(comparison.items_for(first)[0])
    second_gt, second_pred = comparison.load_gt_pred(comparison.items_for(second)[0])
    assert np.all(first_gt[0] == 7)
    assert np.all(second_gt[0] == 7)
    assert np.all(first_pred[0] == 10)
    assert np.all(second_pred[0] == 11)


def test_reference_evaluation_gt_can_be_shared_across_runs(monkeypatch, tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")
    _write_timelapse_item(first, indices=[1, 2], values=[5, 6])
    _write_timelapse_item(second, indices=[1, 2], values=[15, 16])

    comparison = prepare_comparison_outputs([first, second], gt_reference=first)
    first_item = comparison.items_for(first)[0]
    second_item = comparison.items_for(second)[0]
    first_gt, _ = comparison.load_gt_pred(first_item)
    second_gt, _ = comparison.load_gt_pred(second_item)

    assert [float(sample) for sample in first_gt] == [5.0, 6.0]
    assert [float(sample) for sample in second_gt] == [5.0, 6.0]


def test_dataset_gt_without_source_axis_is_reused_for_selected_snr_samples(monkeypatch, tmp_path):
    evaluation = _evaluation(tmp_path, "run_snr")
    _write_timelapse_item(
        evaluation,
        indices=[0, 2],
        values=[1, 3],
        input_id="inp_mltpl_snr/cell.tif",
        source_axis="snr",
    )

    data_dir = tmp_path / "training_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    (data_dir / "inp_mltpl_snr").mkdir(parents=True)
    (data_dir / "gt_avg").mkdir(parents=True)
    imwrite(
        data_dir / "inp_mltpl_snr" / "cell.tif",
        np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5),
        photometric="minisblack",
    )
    imwrite(data_dir / "gt_avg" / "cell.tif", np.full((4, 5), 7, dtype=np.float32))
    (evaluation.folder / "evaluation.yaml").write_text(
        f"""version: 1
dataset:
  name: vim_fixed_multi_snr
  usage: training
  data_type: null
  split: test
  data_dir: {data_dir.as_posix()}
  input: inp_mltpl_snr
  eval_gt: gt_avg
"""
    )
    registry_info = {
        "outputs": {
            "recon": [
                {"key": "inp_mltpl_snr", "path": "inp_mltpl_snr", "role": "inp", "axes": "TYX"},
                {"key": "gt_avg", "path": "gt_avg", "role": "gt", "axes": "YX"},
            ]
        }
    }
    monkeypatch.setattr(eval_outputs_mod, "load_dataset_info", lambda *_args, **_kwargs: registry_info)

    comparison = prepare_comparison_outputs([evaluation], gt_reference="dataset")
    gt_frames, _ = comparison.load_gt_pred(comparison.items_for(evaluation)[0])

    assert len(gt_frames) == 2
    assert np.all(gt_frames[0] == 7)
    assert np.all(gt_frames[1] == 7)


def test_shared_dataset_gt_rejects_different_registered_targets(monkeypatch, tmp_path):
    first = _evaluation(tmp_path, "run_1")
    second = _evaluation(tmp_path, "run_2")
    _write_timelapse_item(first, indices=[0], values=[1], input_id="inp/cell.tif")
    _write_timelapse_item(second, indices=[0], values=[2], input_id="inp/cell.tif")

    data_dir = tmp_path / "training_data" / "preprocess" / "recon"
    _use_canonical_data_dir(monkeypatch, data_dir)
    for evaluation, gt_name in ((first, "gt_avg"), (second, "gt_snr0")):
        (evaluation.folder / "evaluation.yaml").write_text(
            f"""version: 1
dataset:
  name: vim_fixed_multi_snr
  usage: training
  data_type: recon
  split: test
  data_dir: {data_dir.as_posix()}
  input: inp
  eval_gt: {gt_name}
"""
        )

    registry_info = {
        "outputs": {
            "recon": [
                {"key": "inp", "path": "inp", "role": "inp", "axes": "YX"},
                {"key": "gt_avg", "path": "gt_avg", "role": "gt", "axes": "YX"},
                {"key": "gt_snr0", "path": "gt_snr0", "role": "gt", "axes": "YX"},
            ]
        }
    }
    monkeypatch.setattr(eval_outputs_mod, "load_dataset_info", lambda *_args, **_kwargs: registry_info)

    with pytest.raises(ValueError, match="different dataset references"):
        prepare_comparison_outputs([first, second], gt_reference="dataset")
