from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lisai.preprocess.core.dataset_registry import DatasetRegistry


def _write_registry_entry(path: Path, *, usage: str) -> dict:
    registry = DatasetRegistry(path)
    registry.update_after_preprocess(
        dataset_name="demo",
        data_type="recon",
        data_format="single",
        structure=["inp", "gt"],
        outputs=[
            {"key": "inp", "path": "inp", "role": "inp", "axes": "YX"},
            {"key": "gt", "path": "gt", "role": "gt", "axes": "YX"},
        ],
        result=SimpleNamespace(n_files=2, n_frames=None, snr_levels=None, timepoints=None),
        usage=usage,
        split_summary={"counts": {"train": 1, "val": 0, "test": 1}},
    )
    registry.save()
    return registry.data["demo"]


def test_training_registry_defaults_include_target_and_eval_gt(tmp_path: Path):
    entry = _write_registry_entry(tmp_path / "dataset_registry.yml", usage="training")

    assert entry["usage"] == "training"
    assert entry["for_training"] is True
    assert entry["defaults"]["recon"] == {
        "input": "inp",
        "target": "gt",
        "eval_gt": "gt",
    }


def test_evaluation_registry_defaults_include_eval_gt_but_not_target(tmp_path: Path):
    entry = _write_registry_entry(tmp_path / "dataset_registry.yml", usage="evaluation")

    assert entry["usage"] == "evaluation"
    assert entry["for_training"] is False
    assert entry["defaults"]["recon"] == {
        "input": "inp",
        "target": None,
        "eval_gt": "gt",
    }


def test_registry_defaults_stay_null_when_multiple_outputs_match_role(tmp_path: Path):
    registry = DatasetRegistry(tmp_path / "dataset_registry.yml")
    registry.update_after_preprocess(
        dataset_name="demo",
        data_type="recon",
        data_format="mltpl_snr",
        structure=["inp_mltpl_snr", "inp_single", "gt_snr0", "gt_avg"],
        outputs=[
            {"key": "inp_mltpl_snr", "path": "inp_mltpl_snr", "role": "inp", "axes": "TYX"},
            {"key": "inp_single", "path": "inp_single", "role": "inp", "axes": "YX"},
            {"key": "gt_snr0", "path": "gt_snr0", "role": "gt", "axes": "YX"},
            {"key": "gt_avg", "path": "gt_avg", "role": "gt", "axes": "YX"},
        ],
        result=SimpleNamespace(n_files=2, n_frames=4, snr_levels=[0, 1], timepoints=None),
        usage="training",
        split_summary={"counts": {"train": 1, "val": 0, "test": 1}},
    )

    assert registry.data["demo"]["defaults"]["recon"] == {
        "input": None,
        "target": None,
        "eval_gt": None,
    }
    assert registry.data["demo"]["size"]["recon"]["snr_levels"] == {"min": 0, "max": 1}


def test_registry_default_overrides_can_set_and_remove_defaults(tmp_path: Path):
    registry = DatasetRegistry(tmp_path / "dataset_registry.yml")
    registry.update_after_preprocess(
        dataset_name="demo",
        data_type="recon",
        data_format="mltpl_snr",
        structure=["inp_mltpl_snr", "inp_single", "gt_snr0", "gt_avg"],
        outputs=[
            {"key": "inp_mltpl_snr", "path": "inp_mltpl_snr", "role": "inp", "axes": "TYX"},
            {"key": "inp_single", "path": "inp_single", "role": "inp", "axes": "YX"},
            {"key": "gt_snr0", "path": "gt_snr0", "role": "gt", "axes": "YX"},
            {"key": "gt_avg", "path": "gt_avg", "role": "gt", "axes": "YX"},
        ],
        result=SimpleNamespace(n_files=2, n_frames=4, snr_levels=[0, 1], timepoints=None),
        usage="training",
        default_overrides={"input": "inp_mltpl_snr", "target": None, "eval_gt": "gt_snr0"},
        split_summary={"counts": {"train": 1, "val": 0, "test": 1}},
    )

    assert registry.data["demo"]["defaults"]["recon"] == {
        "input": "inp_mltpl_snr",
        "target": None,
        "eval_gt": "gt_snr0",
    }
